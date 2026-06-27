#pragma once

#include "config.h"
#include "llamamodel.h"

#include <string>
#include <vector>

class EmbeddingEngine : public LlamaModelBase {
public:
    explicit EmbeddingEngine(const AppConfig &config);

    // 将文本编码为 embedding 向量，返回维度为 llama_model_n_embd(model_)。
    std::vector<float> generateEmbedding(const std::string &text);
};
