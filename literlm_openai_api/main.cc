#include <chrono>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <variant>

#ifdef _WIN32
  #define _WIN32_WINNT 0x0A00
  #include <windows.h>
#endif

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/log/globals.h"
#include "httplib.h"
#include "nlohmann/json.hpp"
#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/io_types.h"

namespace lm = litert::lm;

ABSL_FLAG(std::string, model_path, "", "Path to the LiteRT-LM model file.");
ABSL_FLAG(std::string, model_name, "", "The name of the model to be served. If empty, it's derived from model_path.");
ABSL_FLAG(std::string, host, "0.0.0.0", "Host address to bind the server to.");
ABSL_FLAG(int, port, 8080, "Port for the server to listen on.");
ABSL_FLAG(bool, verbose, false, "Set the logging verbosity level.");
ABSL_FLAG(bool, use_gpu, false, "Set the backend to GPU.");
ABSL_FLAG(bool, image, false, "Input with Image.");
ABSL_FLAG(bool, audio, false, "Input with Audio.");

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
      } else if (type == "image" && item.contains("image_url")) {
          const auto& image_url_obj = item["image_url"];
          if (image_url_obj.contains("url")) {
              std::string url_data = image_url_obj["url"];
              size_t comma_pos = url_data.find(',');
              if (comma_pos != std::string::npos) {
                  std::string base64_data = url_data.substr(comma_pos + 1);
                  content_parts.push_back({{"type", "image"}, {"blob", base64_data}});
              }
          }
      } else if (type == "audio_url" && item.contains("audio_url")) {
          const auto& audio_url_obj = item["audio_url"];
          if (audio_url_obj.contains("url")) {
              std::string url_data = audio_url_obj["url"];
              size_t comma_pos = url_data.find(',');
              if (comma_pos != std::string::npos) {
                  std::string base64_data = url_data.substr(comma_pos + 1);
                  content_parts.push_back({{"type", "audio"}, {"blob", base64_data}});
              }
          }
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
  // MODIFIED: Accept model_name in the constructor
  explicit ApiServer(std::unique_ptr<lm::Engine> engine, const std::string& model_name)
      : engine_(std::move(engine)), model_name_(model_name) {}

  void Start(const std::string& host, int port) {
    // ADDED: CORS pre-flight and header middleware to fix cross-origin issues
    svr_.Options(R"(/.*)", [](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        if (req.has_header("Access-Control-Request-Headers"))
            res.set_header("Access-Control-Allow-Headers", req.get_header_value("Access-Control-Request-Headers"));
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.status = 204;
    });
    
    svr_.Get("/v1/models",
             [this](const httplib::Request& req, httplib::Response& res) {
               res.set_header("Access-Control-Allow-Origin", "*");
               this->HandleGetModels(req, res);
             });

    svr_.Post("/v1/chat/completions",
              [this](const httplib::Request& req, httplib::Response& res) {
                res.set_header("Access-Control-Allow-Origin", "*");
                this->HandleChatCompletions(req, res);
              });
              
    std::cout << "Server starting on " << host << ":" << port << std::endl;
    svr_.listen(host.c_str(), port);
  }

 private:
  // ADDED: Handler for the /v1/models endpoint
  void HandleGetModels(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json response_json = {
      {"object", "list"},
      {"data", {{
        {"id", model_name_},
        {"object", "model"},
        {"created", std::time(nullptr)},
        {"owned_by", "user"}
      }}}
    };
    res.set_content(response_json.dump(), "application/json");
  }

  void HandleChatCompletions(const httplib::Request& req,
                             httplib::Response& res) {
    try {
      nlohmann::json request_json = nlohmann::json::parse(req.body);
      bool is_streaming = request_json.value("stream", false);

      auto conversation_config_or = lm::ConversationConfig::CreateDefault(*engine_);
      if (!conversation_config_or.ok()) throw std::runtime_error(conversation_config_or.status().ToString());
      
      auto conversation_or = lm::Conversation::Create(*engine_, *conversation_config_or);
      if (!conversation_or.ok()) throw std::runtime_error(conversation_or.status().ToString());
      
      auto conversation = std::shared_ptr<lm::Conversation>(std::move(*conversation_or));

      auto input_message_or = ConvertToLiteRtJsonMessage(request_json["messages"]);
      if (!input_message_or.ok()) {
        throw std::runtime_error(input_message_or.status().ToString());
      }
      
      const std::string& model_name = model_name_;
      
      auto input_message = *input_message_or;

      if (is_streaming) {
        HandleStreamingRequest(res, conversation, input_message, model_name);
      } else {
        HandleBlockingRequest(res, conversation, input_message, model_name);
      }

    } catch (const nlohmann::json::parse_error& e) {
      res.status = 400;
      res.set_content(nlohmann::json{{"error", "Invalid JSON format"}}.dump(), "application/json");
    } catch (const std::exception& e) {
      res.status = 500;
      res.set_content(nlohmann::json{{"error", e.what()}}.dump(), "application/json");
    }
  }

  void HandleBlockingRequest(httplib::Response& res,
                             std::shared_ptr<lm::Conversation> conversation,
                             const lm::JsonMessage& input_message,
                             const std::string& model_name) {
    std::mutex mtx;
    std::condition_variable cv;
    bool stream_finished = false;
    std::string full_reply_content;
    absl::Status error_status;

    auto callback = 
        [&](absl::StatusOr<lm::Message> message_or) {
      std::lock_guard<std::mutex> lock(mtx);
      if (message_or.ok()) {
          const auto& json_message = std::get<lm::JsonMessage>(*message_or);
          if (json_message.is_null()) {
              // is_null() 表示串流結束
              stream_finished = true;
          } else {
              if (json_message.contains("content") && 
                  json_message["content"].is_array() && 
                  !json_message["content"].empty() &&
                  json_message["content"][0].contains("text")) {
                full_reply_content += json_message["content"][0]["text"].get<std::string>();
              }
          }
      } else {
        error_status = message_or.status();
        stream_finished = true;
      }
      
      if (stream_finished) {
        cv.notify_one();
      }
    };

    absl::Status status = conversation->SendMessageAsync(input_message, callback);
    if (!status.ok()) {
        throw std::runtime_error("Failed to start generation: " + status.ToString());
    }
    
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [&] { return stream_finished; });

    if (!error_status.ok()) {
        throw std::runtime_error("Model inference failed: " + error_status.ToString());
    }

    nlohmann::json response_json = {
        {"id", "chatcmpl-local-blocking"},
        {"object", "chat.completion"},
        {"created", std::time(nullptr)},
        {"model", model_name},
        {"choices", {{
             {"index", 0},
             {"message", {{"role", "assistant"}, {"content", full_reply_content}}},
             {"finish_reason", "stop"},
         }}},
        {"usage", {{"prompt_tokens", 0}, {"completion_tokens", 0}, {"total_tokens", 0}}}};
    
    res.set_content(response_json.dump(), "application/json");
}

  void HandleStreamingRequest(httplib::Response& res, 
                              std::shared_ptr<lm::Conversation> conversation,
                              const lm::JsonMessage& input_message,
                              const std::string& model_name) {
    res.set_chunked_content_provider("text/event-stream", 
      [conversation, input_message, model_name](size_t offset, httplib::DataSink& sink) -> bool {
        
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
        
        absl::Status status = conversation->SendMessageAsync(input_message, callback);
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
  std::string model_name_; // ADDED: Member to store model name
};

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  const bool verbose = absl::GetFlag(FLAGS_verbose);
  if(!verbose) absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    std::cerr << "Error: --model_path is required." << std::endl;
    return 1;
  }

  // ADDED: Logic to determine the model name from flag or file path
  std::string model_name = absl::GetFlag(FLAGS_model_name);
  if (model_name.empty()) {
    size_t last_slash_idx = model_path.find_last_of("/\\");
    if (std::string::npos != last_slash_idx) {
      model_name = model_path.substr(last_slash_idx + 1);
    } else {
      model_name = model_path;
    }
  }

  auto model_assets_or = lm::ModelAssets::Create(model_path);
  if (!model_assets_or.ok()) {
    std::cerr << "Failed to create model assets: " << model_assets_or.status() << std::endl;
    return 1;
  }

  const bool use_gpu = absl::GetFlag(FLAGS_use_gpu);
  const bool image = absl::GetFlag(FLAGS_image);
  const bool audio = absl::GetFlag(FLAGS_audio);
  auto engine_settings_or = lm::EngineSettings::CreateDefault(*model_assets_or,
                use_gpu ? lm::Backend::GPU : lm::Backend::CPU,
                image ? std::optional<lm::Backend>(lm::Backend::CPU/*lm::Backend::GPU*/) : std::nullopt,
                audio ? std::optional<lm::Backend>(lm::Backend::CPU) : std::nullopt);
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
  std::cout << "Serving model: " << model_name << std::endl; // ADDED: Log the model name being served
  
  // MODIFIED: Pass the determined model name to the server
  ApiServer server(std::move(*engine_or), model_name);
  server.Start(absl::GetFlag(FLAGS_host), absl::GetFlag(FLAGS_port));

  return 0;
}
