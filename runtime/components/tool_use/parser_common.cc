// Copyright 2025 The Google AI Edge Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/components/tool_use/parser_common.h"

#include <cstddef>
#include <exception>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "ANTLRErrorListener.h"
#include "Parser.h"
#include "Recognizer.h"
#include "Token.h"
#include "atn/ATNConfigSet.h"
#include "dfa/DFA.h"

namespace litert::lm {

void DefaultErrorListener::syntaxError(antlr4::Recognizer* recognizer,
                                       antlr4::Token* offendingSymbol,
                                       size_t line, size_t charPositionInLine,
                                       const std::string& msg,
                                       std::exception_ptr e) {
  status_ = false;
}

void DefaultErrorListener::reportAmbiguity(antlr4::Parser* recognizer,
                                           const antlr4::dfa::DFA& dfa,
                                           size_t startIndex, size_t stopIndex,
                                           bool exact,
                                           const antlrcpp::BitSet& ambigAlts,
                                           antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

void DefaultErrorListener::reportAttemptingFullContext(
    antlr4::Parser* recognizer, const antlr4::dfa::DFA& dfa, size_t startIndex,
    size_t stopIndex, const antlrcpp::BitSet& conflictingAlts,
    antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

void DefaultErrorListener::reportContextSensitivity(
    antlr4::Parser* recognizer, const antlr4::dfa::DFA& dfa, size_t startIndex,
    size_t stopIndex, size_t prediction, antlr4::atn::ATNConfigSet* configs) {
  status_ = false;
}

absl::string_view StripQuotes(absl::string_view text) {
  if (text.size() < 2 || (text.front() != '"' && text.front() != '\'') ||
      text.back() != text.front()) {
    return text;
  }
  return text.substr(1, text.size() - 2);
}

}  // namespace litert::lm
