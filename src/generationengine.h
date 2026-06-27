#pragma once

#include "chunk.h"
#include "config.h"
#include "llamamodel.h"

#include <string>
#include <vector>

class GenerationEngine : public LlamaModelBase {
public:
    explicit GenerationEngine(const AppConfig &config);

    // 基于检索片段组装 RAG Prompt，并将生成结果流式输出到 stdout。
    bool generateStream(const std::string &query, const std::vector<Chunk> &chunks);
};
