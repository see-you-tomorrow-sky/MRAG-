cmake_minimum_required(VERSION 3.14)
project(mrag LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# 引入外部依赖
message(STATUS "Fetching nlohmann/json...")
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(json)

# 只构建本项目需要的 llama.cpp 库，避免拉取后额外编译示例和测试。
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)

message(STATUS "Fetching llama.cpp...")
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    FetchContent_Declare(
        llama
        URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9703.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
else()
    FetchContent_Declare(
        llama
        URL https://github.com/ggml-org/llama.cpp/archive/refs/tags/b9703.zip
    )
endif()
FetchContent_MakeAvailable(llama)

# 设置编译目标
set(MRAG_SOURCES
    src/main.cc
    src/config.cc
    src/chunk.cc
    src/document.cc
    src/vectordatabase.cc
    src/llamamodel.cc
    src/embeddingengine.cc
    src/generationengine.cc
    src/mragapp.cc
)

add_executable(mrag ${MRAG_SOURCES})
target_include_directories(mrag PRIVATE src)
target_link_libraries(mrag PRIVATE nlohmann_json::nlohmann_json)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mrag PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(NOT TARGET llama)
    message(FATAL_ERROR "llama.cpp target 'llama' was not created")
endif()

if(TARGET ggml)
    target_link_libraries(mrag PRIVATE llama ggml)
elseif(TARGET ggml::ggml)
    target_link_libraries(mrag PRIVATE llama ggml::ggml)
else()
    message(FATAL_ERROR "llama.cpp target 'ggml' was not created")
endif()
