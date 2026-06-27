#pragma once

#include "config.h"

#include <string>
#include <vector>

#include <llama.h>

enum class ModelMode {
    Embedding,
    Generation
};

class LlamaModelBase {
public:
    LlamaModelBase(const AppConfig &config, ModelMode mode);
    virtual ~LlamaModelBase();

    LlamaModelBase(const LlamaModelBase &) = delete;
    LlamaModelBase &operator=(const LlamaModelBase &) = delete;
    LlamaModelBase(LlamaModelBase &&) = delete;
    LlamaModelBase &operator=(LlamaModelBase &&) = delete;

protected:
    std::vector<llama_token> tokenize(const std::string &text) const;

    AppConfig config_;
    llama_model *model_{};
    llama_context *ctx_{};
};
