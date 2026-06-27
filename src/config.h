#pragma once

#include <cstddef>
#include <string>

struct AppConfig {
    std::string emb_model_path = "./models/bge-small-zh-v1.5-f16.gguf";
    std::string gen_model_path = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    int n_gpu_layers = 99;

    std::size_t chunk_size = 1500;
    std::size_t overlap_size = 150;
    std::size_t top_k = 5;
    bool show_recalled_chunks = true;

    int gen_n_ctx = 8192;
    int gen_n_batch = 8192;
    int gen_n_ubatch = 512;
    int max_output_tokens = 1024;

    float temperature = 0.3F;
    int top_k_sampler = 40;
    float top_p = 0.9F;

    int emb_n_ctx = 1024;
    int emb_n_batch = 1024;
    int emb_n_ubatch = 1024;
};

// 从 JSON 文件加载配置；文件不存在时保留 AppConfig 默认值。
AppConfig loadConfig(const std::string &path = "config.json");
