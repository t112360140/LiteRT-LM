cc_binary(
    name = "literlm_api_server",
    srcs = ["main.cc"],
    copts = ["/D_WIN32_WINNT=0x0A00"],
    deps = [
        "//runtime/engine:litert_lm_lib",
        "@com_github_yhirose_cpp_httplib//:httplib",
        "@nlohmann_json//:json",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "literlm_api_server_linux",
    srcs = ["main.cc"],
    deps = [
        "//runtime/engine:litert_lm_lib",
        "@com_github_yhirose_cpp_httplib//:httplib",
        "@nlohmann_json//:json",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)
