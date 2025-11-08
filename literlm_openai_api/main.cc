#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <variant>

#define _WIN32_WINNT 0x0A00
#include <windows.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"

// FIX: Do not use a 'using' alias for 'json' to avoid redefinition conflicts
// with dependencies like minja. Use nlohmann::json directly.
namespace lm = litert::lm;

ABSL_FLAG(std::string, model_path, "", "Path to the LiteRT-LM model file.");
ABSL_FLAG(std::string, host, "0.0.0.0", "Host address to bind the server to.");
ABSL_FLAG(int, port, 8080, "Port for the server to listen on.");

absl::StatusOr<lm::JsonMessage> ConvertToLiteRtJsonMessage(const nlohmann::json& messages) {
  if (!messages.is_array() || messages.empty()) {
    return absl::InvalidArgumentError("'messages' must be a non-empty array.");
  }

  const auto& last_message = messages.back();
  if (!last_message.contains("role") || !last_message.contains("content")) {
    return absl::InvalidArgumentError("Last message must have 'role' and 'content'.");
  }

  lm::JsonMessage output_message;
  output_message["role"] = last_message["role"];
  const auto& content = last_message["content"];

  if (content.is_string()) {
    output_message["content"] = content.get<std::string>();
  } else if (content.is_array()) {
    nlohmann::json content_parts = nlohmann::json::array();
    for (const auto& item : content) {
      if (!item.contains("type")) continue;
      std::string type = item["type"];
      if (type == "text" && item.contains("text")) {
        content_parts.push_back({{"type", "text"}, {"text", item["text"]}});
      } else if (type == "image" && item.contains("blob")) {
        content_parts.push_back({{"type", "image"}, {"blob", item["blob"]}});
      } else if (type == "audio" && item.contains("blob")) {
        content_parts.push_back({{"type", "audio"}, {"blob", item["blob"]}});
      }
    }
    output_message["content"] = content_parts;
  } else {
    return absl::InvalidArgumentError("'content' must be a string or an array.");
  }

  return output_message;
}

std::string format_sse_chunk(const std::string& id, const std::string& model_name,
                           const std::string& content_delta) {
  nlohmann::json chunk = {
      {"id", id},
      {"object", "chat.completion.chunk"},
      {"created", std::time(nullptr)},
      {"model", model_name},
      {"choices",
       {{{"index", 0},
         {"delta", {{"role", "assistant"}, {"content", content_delta}}},
         {"finish_reason", nullptr}}}}};
  return absl::StrCat("data: ", chunk.dump(), "\n\n");
}

class ApiServer {
 public:
  explicit ApiServer(std::unique_ptr<lm::Engine> engine)
      : engine_(std::move(engine)) {}

  void Start(const std::string& host, int port) {
    svr_.Post("/v1/chat/completions",
              [this](const httplib::Request& req, httplib::Response& res) {
                this->HandleChatCompletions(req, res);
              });
    std::cout << "Server starting on " << host << ":" << port << std::endl;
    svr_.listen(host.c_str(), port);
  }

 private:
  void HandleChatCompletions(const httplib::Request& req,
                             httplib::Response& res) {
    try {
      nlohmann::json request_json = nlohmann::json::parse(req.body);
      bool is_streaming = request_json.value("stream", false);

      auto conversation_config_or = lm::ConversationConfig::CreateDefault(*engine_);
      if (!conversation_config_or.ok()) throw std::runtime_error(conversation_config_or.status().ToString());
      auto conversation_or = lm::Conversation::Create(*engine_, *conversation_config_or);
      if (!conversation_or.ok()) throw std::runtime_error(conversation_or.status().ToString());
      auto conversation = std::move(*conversation_or);

      auto input_message_or = ConvertToLiteRtJsonMessage(request_json["messages"]);
      if (!input_message_or.ok()) {
        throw std::runtime_error(input_message_or.status().ToString());
      }
      
      const std::string model_name = absl::GetFlag(FLAGS_model_path);

      if (is_streaming) {
        HandleStreamingRequest(res, *conversation, *input_message_or, model_name);
      } else {
        HandleBlockingRequest(res, *conversation, *input_message_or, model_name);
      }

    } catch (const nlohmann::json::parse_error& e) {
      res.status = 400;
      res.set_content(nlohmann::json{{"error", "Invalid JSON format"}}.dump(), "application/json");
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
    }
  }

  void HandleBlockingRequest(httplib::Response& res, lm::Conversation& conversation,
                             const lm::JsonMessage& input_message,
                             const std::string& model_name) {
    absl::StatusOr<lm::Message> response_message_or = conversation.SendMessage(input_message);
    if (!response_message_or.ok()) {
      throw std::runtime_error("Model inference failed: " + response_message_or.status().ToString());
    }
    
    const auto& json_message = std::get<lm::JsonMessage>(*response_message_or);
    std::string model_reply = json_message["content"][0]["text"];

    nlohmann::json response_json = {
        {"id", "chatcmpl-local-blocking"},
        {"object", "chat.completion"},
        {"created", std::time(nullptr)},
        {"model", model_name},
        {"choices", {{
             {"index", 0},
             {"message", {{"role", "assistant"}, {"content", model_reply}}},
             {"finish_reason", "stop"},
         }}},
        {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}};
    
    res.set_content(response_json.dump(), "application/json");
  }

  void HandleStreamingRequest(httplib::Response& res, lm::Conversation& conversation,
                              const lm::JsonMessage& input_message,
                              const std::string& model_name) {
    res.set_chunked_content_provider("text/event-stream", 
      [&](size_t offset, httplib::DataSink& sink) -> bool {
        
        std::mutex mtx;
        std::condition_variable cv;
        bool stream_finished = false;

        auto callback = 
            [&](absl::StatusOr<lm::Message> message_or) {
          std::lock_guard<std::mutex> lock(mtx);
          if (message_or.ok()) {
              const auto& json_message = std::get<lm::JsonMessage>(*message_or);
              if (json_message.is_null()) {
                  stream_finished = true;
              } else {
                  const std::string& delta = json_message["content"][0]["text"];
                  std::string sse_chunk = format_sse_chunk("chatcmpl-local-streaming", model_name, delta);
                  sink.write(sse_chunk.c_str(), sse_chunk.length());
              }
          } else {
            std::cerr << "Streaming error: " << message_or.status() << std::endl;
            stream_finished = true;
          }
          if (stream_finished) {
            cv.notify_one();
          }
        };
        
        absl::Status status = conversation.SendMessageAsync(input_message, callback);
        if (!status.ok()) {
            std::cerr << "Failed to start streaming generation: " << status << std::endl;
            sink.done();
            return false;
        }
        
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return stream_finished; });

        sink.write("data: [DONE]\n\n", 15);
        sink.done();
        return false;
      });
  }

  std::unique_ptr<lm::Engine> engine_;
  httplib::Server svr_;
};

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "Error: --model_path is required." << std::endl;
    return 1;
  }

  auto model_assets_or = lm::ModelAssets::Create(model_path);
  if (!model_assets_or.ok()) {
    std::cerr << "Failed to create model assets: " << model_assets_or.status() << std::endl;
    return 1;
  }

  auto engine_settings_or = lm::EngineSettings::CreateDefault(*model_assets_or, lm::Backend::CPU);
  if (!engine_settings_or.ok()) {
      std::cerr << "Failed to create engine settings: " << engine_settings_or.status() << std::endl;
      return 1;
  }
  
  absl::StatusOr<std::unique_ptr<lm::Engine>> engine_or = lm::Engine::CreateEngine(*engine_settings_or);
  if (!engine_or.ok()) {
    std::cerr << "Failed to create engine: " << engine_or.status() << std::endl;
    return 1;
  }
  std::cout << "LiteRT-LM engine initialized successfully." << std::endl;
  
  ApiServer server(std::move(*engine_or));
  server.Start(absl::GetFlag(FLAGS_host), absl::GetFlag(FLAGS_port));

  return 0;
}
