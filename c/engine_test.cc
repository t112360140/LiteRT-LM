#include "c/engine.h"

#include <cstring>
#include <algorithm>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_matchers.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl

namespace {

std::string GetTestdataPath(const std::string& filename) {
  std::string srcdir = ::testing::SrcDir();
  // On Windows, SrcDir() may return paths with backslashes. The LiteRT LM C API
  // expects forward slashes.
  std::replace(srcdir.begin(), srcdir.end(), '\\', '/');
  return srcdir + "/" + filename;
}

// Use unique_ptr for automatic resource management of C API objects.
using EngineSettingsPtr =
    std::unique_ptr<LiteRtLmEngineSettings,
                    decltype(&litert_lm_engine_settings_delete)>;
using EnginePtr =
    std::unique_ptr<LiteRtLmEngine, decltype(&litert_lm_engine_delete)>;
using SessionPtr =
    std::unique_ptr<LiteRtLmSession, decltype(&litert_lm_session_delete)>;
using ResponsesPtr =
    std::unique_ptr<LiteRtLmResponses, decltype(&litert_lm_responses_delete)>;
using ConversationPtr =
    std::unique_ptr<LiteRtLmConversation,
                    decltype(&litert_lm_conversation_delete)>;
using JsonResponsePtr =
    std::unique_ptr<LiteRtLmJsonResponse,
                    decltype(&litert_lm_json_response_delete)>;

TEST(EngineCTest, GenerateContent) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(engine.get()),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), &input_data, 1),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  EXPECT_EQ(litert_lm_responses_get_num_candidates(responses.get()), 1);
  const char* response_text =
      litert_lm_responses_get_response_text_at(responses.get(), 0);
  ASSERT_NE(response_text, nullptr);
  EXPECT_GT(strlen(response_text), 0);
}

TEST(EngineCTest, ConversationSendMessage) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(litert_lm_conversation_create(engine.get()),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  JsonResponsePtr response(
      litert_lm_conversation_send_message(conversation.get(), message_json),
      &litert_lm_json_response_delete);
  ASSERT_NE(response, nullptr);

  const char* response_str = litert_lm_json_response_get_string(response.get());
  ASSERT_NE(response_str, nullptr);
  EXPECT_GT(strlen(response_str), 0);
}

struct StreamCallbackData {
  std::string response;
  absl::Notification done;
  absl::Status status;
};

void StreamCallback(void* callback_data, const char* chunk, bool is_final,
                    const char* error_msg) {
  auto* data = static_cast<StreamCallbackData*>(callback_data);
  if (error_msg) {
    data->status = absl::InternalError(error_msg);
  }
  if (chunk) {
    data->response.append(chunk);
  }
  if (is_final) {
    data->done.Notify();
  }
}

TEST(EngineCTest, GenerateContentStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(engine.get()),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  StreamCallbackData callback_data;
  int result = litert_lm_session_generate_content_stream(
      session.get(), &input_data, 1, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();

  // This model is too small and generate random output, so the result may be
  // either success or failure due to maximum kv-cache size reached.
  EXPECT_THAT(
      callback_data.status,
      testing::AnyOf(absl_testing::IsOk(),
                     absl_testing::StatusIs(
                     absl::StatusCode::kInternal,
                     testing::HasSubstr("Maximum kv-cache size reached."))));
  EXPECT_GT(callback_data.response.length(), 0);
}

TEST(EngineCTest, ConversationSendMessageStream) {
  const std::string task_path = GetTestdataPath(
      "litert_lm/runtime/testdata/test_lm_new_metadata.task");

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  ConversationPtr conversation(litert_lm_conversation_create(engine.get()),
                               &litert_lm_conversation_delete);
  ASSERT_NE(conversation, nullptr);

  const char* message_json =
      R"({"role": "user", "content": [{"type": "text", "text": "Hello"}]})";
  StreamCallbackData callback_data;
  int result = litert_lm_conversation_send_message_stream(
      conversation.get(), message_json, &StreamCallback, &callback_data);
  ASSERT_EQ(result, 0);

  callback_data.done.WaitForNotification();
  EXPECT_GT(callback_data.response.length(), 0);
}

using BenchmarkInfoPtr =
    std::unique_ptr<LiteRtLmBenchmarkInfo,
                    decltype(&litert_lm_benchmark_info_delete)>;

TEST(EngineCTest, Benchmark) {
  auto task_path =
      std::filesystem::path(::testing::SrcDir()) /
      "litert_lm/runtime/testdata/test_lm_new_metadata.task";

  EngineSettingsPtr settings(
      litert_lm_engine_settings_create(task_path.c_str(), "cpu"),
      &litert_lm_engine_settings_delete);
  ASSERT_NE(settings, nullptr);
  litert_lm_engine_settings_set_max_num_tokens(settings.get(), 16);
  litert_lm_engine_settings_enable_benchmark(settings.get());

  EnginePtr engine(litert_lm_engine_create(settings.get()),
                   &litert_lm_engine_delete);
  ASSERT_NE(engine, nullptr);

  SessionPtr session(litert_lm_engine_create_session(engine.get()),
                     &litert_lm_session_delete);
  ASSERT_NE(session, nullptr);

  const char* prompt = "Hello world!";
  InputData input_data;
  input_data.type = kInputText;
  input_data.data = prompt;
  input_data.size = strlen(prompt);
  ResponsesPtr responses(
      litert_lm_session_generate_content(session.get(), &input_data, 1),
      &litert_lm_responses_delete);
  ASSERT_NE(responses, nullptr);

  BenchmarkInfoPtr benchmark_info(
      litert_lm_session_get_benchmark_info(session.get()),
      &litert_lm_benchmark_info_delete);
  ASSERT_NE(benchmark_info, nullptr);

  EXPECT_GT(litert_lm_benchmark_info_get_time_to_first_token(
    benchmark_info.get()), 0.0);
  int num_prefill_turns =
      litert_lm_benchmark_info_get_num_prefill_turns(benchmark_info.get());
  EXPECT_GT(num_prefill_turns, 0);
  for (int i = 0; i < num_prefill_turns; ++i) {
    EXPECT_GT(litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(
                  benchmark_info.get(), i),
              0.0);
  }
  int num_decode_turns =
      litert_lm_benchmark_info_get_num_decode_turns(benchmark_info.get());
  EXPECT_GT(num_decode_turns, 0);
  for (int i = 0; i < num_decode_turns; ++i) {
    EXPECT_GT(litert_lm_benchmark_info_get_decode_tokens_per_sec_at(
                  benchmark_info.get(), i),
              0.0);
  }
}
}  // namespace
