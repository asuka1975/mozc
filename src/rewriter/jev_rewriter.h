// Copyright 2026 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the conditions in the LICENSE
// file are met.

#ifndef MOZC_REWRITER_JEV_REWRITER_H_
#define MOZC_REWRITER_JEV_REWRITER_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "rewriter/rewriter_interface.h"

namespace mozc {

// Interface to the Jev decision model. Kept separate from the rewriter so
// tests do not make network requests and another transport can be substituted.
class JevEvaluatorInterface {
 public:
  virtual ~JevEvaluatorInterface() = default;

  // Returns one contextual-naturalness score per candidate. A failure is
  // represented by nullopt; the caller then preserves Mozc's original order.
  virtual std::optional<std::vector<double>> Evaluate(
      absl::string_view preceding_context, absl::string_view composition,
      absl::Span<const std::string> candidates) const = 0;
};

// HTTP boundary for the Jev API. Tests can supply a transport without making
// network requests.
class JevHttpTransportInterface {
 public:
  virtual ~JevHttpTransportInterface() = default;
  virtual std::optional<std::string> Post(absl::string_view api_key,
                                          absl::string_view body,
                                          int timeout_millis) const = 0;
};

class JevApiEvaluator final : public JevEvaluatorInterface {
 public:
  JevApiEvaluator(std::unique_ptr<const JevHttpTransportInterface> transport,
                  std::string api_key, std::string model,
                  int timeout_millis);

  std::optional<std::vector<double>> Evaluate(
      absl::string_view preceding_context, absl::string_view composition,
      absl::Span<const std::string> candidates) const override;

 private:
  std::unique_ptr<const JevHttpTransportInterface> transport_;
  std::string api_key_;
  std::string model_;
  int timeout_millis_;
};

// Reranks generated conversion, suggestion, and prediction candidates by
// asking Jev how naturally each candidate continues the preceding text.
class JevRewriter final : public RewriterInterface {
 public:
  JevRewriter();
  explicit JevRewriter(std::unique_ptr<const JevEvaluatorInterface> evaluator,
                       size_t max_candidates = 9);
  ~JevRewriter() override = default;

  int capability(const ConversionRequest&) const override {
    return RewriterInterface::CONVERSION | RewriterInterface::PREDICTION |
           RewriterInterface::SUGGESTION;
  }

  bool Rewrite(const ConversionRequest& request,
               Segments* segments) const override;

 private:
  std::unique_ptr<const JevEvaluatorInterface> evaluator_;
  size_t max_candidates_;
};

}  // namespace mozc

#endif  // MOZC_REWRITER_JEV_REWRITER_H_
