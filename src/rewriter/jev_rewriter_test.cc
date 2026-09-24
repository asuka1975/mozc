// Copyright 2026 Google LLC

#include "rewriter/jev_rewriter.h"

#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "converter/segments.h"
#include "google/protobuf/struct.pb.h"
#include "google/protobuf/util/json_util.h"
#include "protocol/commands.pb.h"
#include "protocol/config.pb.h"
#include "request/conversion_request.h"
#include "testing/gunit.h"

namespace mozc {
namespace {

class FakeJevEvaluator final : public JevEvaluatorInterface {
 public:
  explicit FakeJevEvaluator(std::optional<std::vector<double>> scores)
      : scores_(std::move(scores)) {}

  std::optional<std::vector<double>> Evaluate(
      absl::string_view preceding_context, absl::string_view composition,
      absl::Span<const std::string> candidates) const override {
    ++call_count;
    context = std::string(preceding_context);
    key = std::string(composition);
    values.assign(candidates.begin(), candidates.end());
    return scores_;
  }

  mutable int call_count = 0;
  mutable std::string context;
  mutable std::string key;
  mutable std::vector<std::string> values;

 private:
  std::optional<std::vector<double>> scores_;
};

class FakeHttpTransport final : public JevHttpTransportInterface {
 public:
  explicit FakeHttpTransport(std::optional<std::string> response)
      : response_(std::move(response)) {}

  std::optional<std::string> Post(absl::string_view api_key,
                                  absl::string_view body,
                                  int timeout_millis) const override {
    ++call_count;
    received_key = std::string(api_key);
    received_body = std::string(body);
    received_timeout = timeout_millis;
    return response_;
  }

  mutable int call_count = 0;
  mutable std::string received_key;
  mutable std::string received_body;
  mutable int received_timeout = 0;

 private:
  std::optional<std::string> response_;
};

Segment* AddCandidates(std::initializer_list<absl::string_view> values,
                       Segments* segments) {
  Segment* segment = segments->add_segment();
  segment->set_key("こうほ");
  for (absl::string_view value : values) {
    converter::Candidate* candidate = segment->add_candidate();
    candidate->key = "こうほ";
    candidate->value = value;
  }
  return segment;
}

ConversionRequest MakeRequest(const commands::Context& context,
                              bool incognito = false,
                              bool no_history = false) {
  config::Config config;
  config.set_incognito_mode(incognito);
  if (no_history) {
    config.set_history_learning_level(config::Config::NO_HISTORY);
  }
  return ConversionRequestBuilder()
      .SetContext(context)
      .SetConfig(config)
      .SetKey("こうほ")
      .SetRequestType(ConversionRequest::PREDICTION)
      .Build();
}

TEST(JevRewriterTest, ReranksStablyByJevScore) {
  auto evaluator =
      std::make_unique<FakeJevEvaluator>(std::vector<double>{0.2, 0.9, 0.9});
  FakeJevEvaluator* evaluator_ptr = evaluator.get();
  JevRewriter rewriter(std::move(evaluator));
  commands::Context context;
  context.set_preceding_text("昼食には");
  Segments segments;
  Segment* segment = AddCandidates({"コーヒー", "カレー", "うどん"}, &segments);

  EXPECT_TRUE(rewriter.Rewrite(MakeRequest(context), &segments));
  EXPECT_EQ(segment->candidate(0).value, "カレー");
  EXPECT_EQ(segment->candidate(1).value, "うどん");
  EXPECT_EQ(segment->candidate(2).value, "コーヒー");
  EXPECT_EQ(evaluator_ptr->call_count, 1);
  EXPECT_EQ(evaluator_ptr->context, "昼食には");
  EXPECT_EQ(evaluator_ptr->key, "こうほ");
  EXPECT_EQ(evaluator_ptr->values,
            std::vector<std::string>({"コーヒー", "カレー", "うどん"}));
}

TEST(JevRewriterTest, PreservesOrderWhenEvaluationFails) {
  auto evaluator = std::make_unique<FakeJevEvaluator>(std::nullopt);
  JevRewriter rewriter(std::move(evaluator));
  commands::Context context;
  context.set_preceding_text("文脈");
  Segments segments;
  Segment* segment = AddCandidates({"A", "B"}, &segments);

  EXPECT_FALSE(rewriter.Rewrite(MakeRequest(context), &segments));
  EXPECT_EQ(segment->candidate(0).value, "A");
  EXPECT_EQ(segment->candidate(1).value, "B");
}

TEST(JevRewriterTest, UsesContextAcrossLineBreaks) {
  auto evaluator =
      std::make_unique<FakeJevEvaluator>(std::vector<double>{0.0, 1.0});
  FakeJevEvaluator* evaluator_ptr = evaluator.get();
  JevRewriter rewriter(std::move(evaluator));
  commands::Context context;
  context.set_preceding_text("今日は晴れです。\n");
  Segments segments;
  AddCandidates({"天気", "転機"}, &segments);

  EXPECT_TRUE(rewriter.Rewrite(MakeRequest(context), &segments));
  EXPECT_EQ(evaluator_ptr->call_count, 1);
  EXPECT_EQ(evaluator_ptr->context, "今日は晴れです。\n");
}

TEST(JevRewriterTest, DoesNotSendPrivacySensitiveContext) {
  for (int privacy_mode = 0; privacy_mode < 3; ++privacy_mode) {
    auto evaluator = std::make_unique<FakeJevEvaluator>(
        std::vector<double>{0.0, 1.0});
    FakeJevEvaluator* evaluator_ptr = evaluator.get();
    JevRewriter rewriter(std::move(evaluator));
    commands::Context context;
    context.set_preceding_text("secret");
    if (privacy_mode == 0) {
      context.set_input_field_type(commands::Context::PASSWORD);
    }
    Segments segments;
    AddCandidates({"A", "B"}, &segments);

    const ConversionRequest request =
        MakeRequest(context, privacy_mode == 1, privacy_mode == 2);
    EXPECT_FALSE(rewriter.Rewrite(request, &segments));
    EXPECT_EQ(evaluator_ptr->call_count, 0);
  }
}

TEST(JevRewriterTest, LimitsTheNumberOfCandidatesSentToJev) {
  auto evaluator =
      std::make_unique<FakeJevEvaluator>(std::vector<double>{0.0, 1.0});
  FakeJevEvaluator* evaluator_ptr = evaluator.get();
  JevRewriter rewriter(std::move(evaluator), 2);
  commands::Context context;
  context.set_preceding_text("文脈");
  Segments segments;
  Segment* segment = AddCandidates({"A", "B", "C"}, &segments);

  EXPECT_TRUE(rewriter.Rewrite(MakeRequest(context), &segments));
  EXPECT_EQ(evaluator_ptr->values, std::vector<std::string>({"A", "B"}));
  EXPECT_EQ(segment->candidate(0).value, "B");
  EXPECT_EQ(segment->candidate(1).value, "A");
  EXPECT_EQ(segment->candidate(2).value, "C");
}

TEST(JevRewriterTest, SupportsConversionSuggestionAndPrediction) {
  JevRewriter rewriter(nullptr);
  const ConversionRequest request;
  EXPECT_EQ(rewriter.capability(request),
            RewriterInterface::CONVERSION | RewriterInterface::SUGGESTION |
                RewriterInterface::PREDICTION);
}

TEST(JevRewriterTest, SendsStructuredJsonAndReadsTypedScores) {
  auto transport = std::make_unique<FakeHttpTransport>(
      R"({"answers":{"c0":{"type":"score","score":0.25},"c1":{"type":"score","score":2.75}}})");
  FakeHttpTransport* transport_ptr = transport.get();
  JevApiEvaluator evaluator(std::move(transport), "test-key", "jev-latest", 250);
  const std::vector<std::string> candidates = {"雪\"雨", "晴れ"};

  EXPECT_EQ(evaluator.Evaluate("明日は\n", "てんき", absl::MakeConstSpan(candidates)),
            std::optional<std::vector<double>>({0.25, 2.75}));
  EXPECT_EQ(transport_ptr->call_count, 1);
  EXPECT_EQ(transport_ptr->received_key, "test-key");
  EXPECT_EQ(transport_ptr->received_timeout, 250);

  google::protobuf::Struct request;
  ASSERT_TRUE(google::protobuf::util::JsonStringToMessage(
                  transport_ptr->received_body, &request)
                  .ok());
  EXPECT_EQ(request.fields().at("model").string_value(), "jev-latest");
  const auto& state = request.fields().at("state").struct_value().fields();
  EXPECT_EQ(state.at("preceding_text").string_value(), "明日は\n");
  EXPECT_EQ(state.at("reading").string_value(), "てんき");
  EXPECT_EQ(state.at("candidates").list_value().values(0).string_value(),
            "雪\"雨");
  const auto& questions =
      request.fields().at("questions").struct_value().fields();
  EXPECT_EQ(questions.size(), 2);
  EXPECT_EQ(questions.at("c0").struct_value().fields().at("type").string_value(),
            "score");
}

TEST(JevRewriterTest, RejectsMalformedApiResponse) {
  for (const char* response : {
           R"({"answers":{"c0":{"score":1}}})",
           R"({"answers":{"c0":{"score":"1"},"c1":{"score":2}}})",
           R"({"answers":{"c0":{"type":"choice","score":1},"c1":{"score":2}}})",
           "not json",
       }) {
    JevApiEvaluator evaluator(
        std::make_unique<FakeHttpTransport>(std::string(response)),
                              "test-key", "jev-latest", 250);
    const std::vector<std::string> candidates = {"A", "B"};
    EXPECT_FALSE(evaluator.Evaluate("文脈", "よみ",
                                    absl::MakeConstSpan(candidates))
                     .has_value());
  }
}

}  // namespace
}  // namespace mozc
