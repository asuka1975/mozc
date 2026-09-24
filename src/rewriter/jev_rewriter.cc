// Copyright 2026 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the LICENSE
// file are met.

#include "rewriter/jev_rewriter.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "base/environ.h"
#include "base/util.h"
#include "converter/candidate.h"
#include "converter/segments.h"
#include "curl/curl.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"

namespace mozc {
namespace {

constexpr size_t kDefaultMaxCandidates = 9;
constexpr size_t kMaxContextCharacters = 512;
constexpr int kDefaultTimeoutMillis = 1000;
constexpr size_t kMaxResponseBytes = 64 * 1024;
constexpr char kJevEndpoint[] = "https://api.typesafe.ai/v1/systemone";

bool IsEnabled(absl::string_view value) {
  return value == "1" || absl::AsciiStrToLower(value) == "true";
}

const google::protobuf::Value* FindField(const google::protobuf::Struct& object,
                                         absl::string_view name) {
  const auto it = object.fields().find(std::string(name));
  return it == object.fields().end() ? nullptr : &it->second;
}

std::optional<std::string> MakeRequestJson(
    absl::string_view preceding_context, absl::string_view composition,
    absl::Span<const std::string> candidates, absl::string_view model) {
  google::protobuf::Struct request;
  (*request.mutable_fields())["model"].set_string_value(std::string(model));

  google::protobuf::Struct* state =
      (*request.mutable_fields())["state"].mutable_struct_value();
  (*state->mutable_fields())["preceding_text"].set_string_value(
      std::string(preceding_context));
  (*state->mutable_fields())["reading"].set_string_value(
      std::string(composition));
  google::protobuf::ListValue* values =
      (*state->mutable_fields())["candidates"].mutable_list_value();
  for (const std::string& candidate : candidates) {
    values->add_values()->set_string_value(candidate);
  }

  google::protobuf::Struct* questions =
      (*request.mutable_fields())["questions"].mutable_struct_value();
  for (size_t i = 0; i < candidates.size(); ++i) {
    google::protobuf::Struct* question =
        (*questions->mutable_fields())[absl::StrCat("c", i)]
            .mutable_struct_value();
    (*question->mutable_fields())["type"].set_string_value("score");
    (*question->mutable_fields())["instructions"].set_string_value(
        absl::StrCat("候補[", i,
                     "]が直前の文章に自然に続く度合いを、意味・文法・語調から評価する。"
                     "候補の順位ではなく文脈との適合性を判定する。"));
    google::protobuf::ListValue* criteria =
        (*question->mutable_fields())["criteria"].mutable_list_value();
    for (absl::string_view label : {"文脈に合わない", "やや不自然", "自然",
                                    "最も自然"}) {
      criteria->add_values()->set_string_value(std::string(label));
    }
  }

  std::string json;
  if (!google::protobuf::util::MessageToJsonString(request, &json).ok()) {
    return std::nullopt;
  }
  return json;
}

std::optional<std::vector<double>> ParseScores(absl::string_view response,
                                                size_t count) {
  google::protobuf::Struct root;
  if (!google::protobuf::util::JsonStringToMessage(response, &root).ok()) {
    return std::nullopt;
  }
  const google::protobuf::Value* answers = FindField(root, "answers");
  if (answers == nullptr || answers->kind_case() !=
                                google::protobuf::Value::kStructValue) {
    return std::nullopt;
  }

  std::vector<double> scores;
  scores.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    const google::protobuf::Value* answer =
        FindField(answers->struct_value(), absl::StrCat("c", i));
    if (answer == nullptr || answer->kind_case() !=
                                 google::protobuf::Value::kStructValue) {
      return std::nullopt;
    }
    const google::protobuf::Struct& fields = answer->struct_value();
    const google::protobuf::Value* type = FindField(fields, "type");
    if (type != nullptr &&
        (type->kind_case() != google::protobuf::Value::kStringValue ||
         type->string_value() != "score")) {
      return std::nullopt;
    }
    const google::protobuf::Value* score = FindField(fields, "score");
    if (score == nullptr || score->kind_case() !=
                                google::protobuf::Value::kNumberValue ||
        !std::isfinite(score->number_value())) {
      return std::nullopt;
    }
    scores.push_back(score->number_value());
  }
  return scores;
}

size_t ReceiveResponse(char* data, size_t size, size_t count, void* output) {
  if (count != 0 && size > kMaxResponseBytes / count) return 0;
  const size_t bytes = size * count;
  std::string* response = static_cast<std::string*>(output);
  if (bytes > kMaxResponseBytes - response->size()) return 0;
  response->append(data, bytes);
  return bytes;
}

class CurlJevTransport final : public JevHttpTransportInterface {
 public:
  std::optional<std::string> Post(absl::string_view api_key,
                                  absl::string_view body,
                                  int timeout_millis) const override {
    if (api_key.empty() || api_key.find_first_of("\r\n") !=
                               absl::string_view::npos) {
      return std::nullopt;
    }
    static const CURLcode init_result = ::curl_global_init(CURL_GLOBAL_DEFAULT);
    if (init_result != CURLE_OK) return std::nullopt;

    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(
        ::curl_easy_init(), &curl_easy_cleanup);
    if (curl == nullptr) return std::nullopt;

    curl_slist* raw_headers =
        ::curl_slist_append(nullptr, "Content-Type: application/json");
    if (raw_headers == nullptr) return std::nullopt;
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(
        raw_headers, &curl_slist_free_all);
    const std::string authorization =
        absl::StrCat("Authorization: Bearer ", api_key);
    raw_headers = ::curl_slist_append(headers.get(), authorization.c_str());
    if (raw_headers == nullptr) return std::nullopt;
    headers.release();
    headers.reset(raw_headers);

    std::string request_body(body);
    std::string response;
    if (::curl_easy_setopt(curl.get(), CURLOPT_URL, kJevEndpoint) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https") !=
            CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get()) !=
            CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_POST, 1L) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDS,
                           request_body.c_str()) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_POSTFIELDSIZE_LARGE,
                           static_cast<curl_off_t>(request_body.size())) !=
            CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS,
                           static_cast<long>(timeout_millis)) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS,
                           static_cast<long>(timeout_millis)) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION,
                           &ReceiveResponse) != CURLE_OK ||
        ::curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &response) !=
            CURLE_OK) {
      return std::nullopt;
    }
    const CURLcode result = ::curl_easy_perform(curl.get());
    long status = 0;
    const CURLcode status_result =
        ::curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    curl.reset();
    if (result != CURLE_OK || status_result != CURLE_OK || status != 200) {
      return std::nullopt;
    }
    return response;
  }
};

std::unique_ptr<const JevEvaluatorInterface> CreateEvaluatorFromEnvironment() {
  if (!IsEnabled(Environ::GetEnv("MOZC_JEV_RERANKER"))) return nullptr;
  std::string api_key = Environ::GetEnv("TYPESAFE_API_KEY");
  if (api_key.empty()) return nullptr;
  std::string model = Environ::GetEnv("MOZC_JEV_MODEL");
  if (model.empty()) model = "jev-latest";
  int timeout_millis = kDefaultTimeoutMillis;
  int configured_timeout = 0;
  if (absl::SimpleAtoi(Environ::GetEnv("MOZC_JEV_TIMEOUT_MS"),
                       &configured_timeout)) {
    timeout_millis = std::clamp(configured_timeout, 50, 5000);
  }
  return std::make_unique<JevApiEvaluator>(
      std::make_unique<CurlJevTransport>(), std::move(api_key),
      std::move(model), timeout_millis);
}

size_t GetMaxCandidatesFromEnvironment() {
  int configured = 0;
  if (absl::SimpleAtoi(Environ::GetEnv("MOZC_JEV_MAX_CANDIDATES"),
                       &configured)) {
    return static_cast<size_t>(std::clamp(configured, 2, 20));
  }
  return kDefaultMaxCandidates;
}

std::string TruncateContext(absl::string_view context) {
  const size_t length = Util::CharsLen(context);
  if (length <= kMaxContextCharacters) return std::string(context);
  return std::string(
      Util::Utf8SubString(context, length - kMaxContextCharacters));
}

}  // namespace

JevApiEvaluator::JevApiEvaluator(
    std::unique_ptr<const JevHttpTransportInterface> transport,
    std::string api_key, std::string model, int timeout_millis)
    : transport_(std::move(transport)),
      api_key_(std::move(api_key)),
      model_(std::move(model)),
      timeout_millis_(timeout_millis) {}

std::optional<std::vector<double>> JevApiEvaluator::Evaluate(
    absl::string_view preceding_context, absl::string_view composition,
    absl::Span<const std::string> candidates) const {
  if (transport_ == nullptr || api_key_.empty() || candidates.empty()) {
    return std::nullopt;
  }
  const std::optional<std::string> request = MakeRequestJson(
      preceding_context, composition, candidates, model_);
  if (!request.has_value()) return std::nullopt;
  const std::optional<std::string> response =
      transport_->Post(api_key_, *request, timeout_millis_);
  if (!response.has_value()) return std::nullopt;
  return ParseScores(*response, candidates.size());
}

JevRewriter::JevRewriter()
    : JevRewriter(CreateEvaluatorFromEnvironment(),
                  GetMaxCandidatesFromEnvironment()) {}

JevRewriter::JevRewriter(
    std::unique_ptr<const JevEvaluatorInterface> evaluator,
    size_t max_candidates)
    : evaluator_(std::move(evaluator)),
      max_candidates_(std::max<size_t>(2, max_candidates)) {}

bool JevRewriter::Rewrite(const ConversionRequest& request,
                          Segments* segments) const {
  if (evaluator_ == nullptr || segments == nullptr || request.incognito_mode() ||
      request.config().history_learning_level() == config::Config::NO_HISTORY ||
      request.context().input_field_type() == commands::Context::PASSWORD) {
    return false;
  }

  // Keep prior lines: an input at the start of a new line still has useful
  // context, even though GetSurroundingContext() discards everything before
  // the last newline.
  std::string preceding_context = request.context().preceding_text();
  if (preceding_context.empty()) {
    preceding_context = std::string(request.converter_history_value());
  }
  if (preceding_context.empty()) return false;
  preceding_context = TruncateContext(preceding_context);

  bool modified = false;
  for (Segment& segment : segments->conversion_segments()) {
    const size_t count = std::min(segment.candidates_size(), max_candidates_);
    if (count < 2) continue;

    std::vector<std::string> values;
    std::vector<const converter::Candidate*> original_candidates;
    values.reserve(count);
    original_candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      const converter::Candidate& candidate = segment.candidate(i);
      values.push_back(candidate.value);
      original_candidates.push_back(&candidate);
    }

    const std::optional<std::vector<double>> scores = evaluator_->Evaluate(
        preceding_context, segment.key(), absl::MakeConstSpan(values));
    if (!scores.has_value() || scores->size() != count ||
        !std::all_of(scores->begin(), scores->end(),
                     [](double score) { return std::isfinite(score); })) {
      continue;
    }

    std::vector<size_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
      return (*scores)[lhs] > (*scores)[rhs];
    });

    for (size_t target = 0; target < count; ++target) {
      const converter::Candidate* desired = original_candidates[order[target]];
      size_t current = target;
      while (current < segment.candidates_size() &&
             &segment.candidate(current) != desired) {
        ++current;
      }
      if (current < segment.candidates_size() && current != target) {
        segment.move_candidate(static_cast<int>(current),
                               static_cast<int>(target));
        modified = true;
      }
    }
  }
  return modified;
}

}  // namespace mozc
