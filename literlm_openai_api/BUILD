cc_binary(
    name = "literlm_api_server",
    srcs = ["main.cc"],
    copts = select({
        "@bazel_platforms//os:windows": [
            # This macro tells the compiler (MSVC) to target Windows 10 or later,
            # which is required by the cpp-httplib library.
            "/D_WIN32_WINNT=0x0A00",
        ],
        "//conditions:default": [],
    }),
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
