#include "llamamodel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string modeName(ModelMode mode) {
    return mode == ModelMode::Embedding ? "embedding" : "generation";
}

std::string modelPathForMode(const AppConfig &config, ModelMode mode) {
    return mode == ModelMode::Embedding ? config.emb_model_path : config.gen_model_path;
}

uint32_t checkedUint32(int value, const char *field_name) {
    if (value <= 0) {
        throw std::runtime_error(std::string(field_name) + " must be positive");
    }
    return static_cast<uint32_t>(value);
}

llama_context_params makeContextParams(const AppConfig &config, ModelMode mode) {
    llama_context_params params = llama_context_default_params();
    params.n_seq_max = 1;

    if (mode == ModelMode::Embedding) {
        params.n_ctx = checkedUint32(config.emb_n_ctx, "emb_n_ctx");
        params.n_batch = checkedUint32(config.emb_n_batch, "emb_n_batch");
        params.n_ubatch = checkedUint32(config.emb_n_ubatch, "emb_n_ubatch");
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    } else {
        params.n_ctx = checkedUint32(config.gen_n_ctx, "gen_n_ctx");
        params.n_batch = checkedUint32(config.gen_n_batch, "gen_n_batch");
        params.n_ubatch = checkedUint32(config.gen_n_ubatch, "gen_n_ubatch");
        params.embeddings = false;
    }

    return params;
}

}  // namespace

LlamaModelBase::LlamaModelBase(const AppConfig &config, ModelMode mode)
    : config_(config) {
    const std::string path = modelPathForMode(config_, mode);
    if (path.empty()) {
        throw std::runtime_error(modeName(mode) + " model path is empty");
    }

    const llama_context_params ctx_params = makeContextParams(config_, mode);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;

    model_ = llama_model_load_from_file(path.c_str(), model_params);
    if (model_ == nullptr) {
        throw std::runtime_error("failed to load " + modeName(mode) + " model: " + path);
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (ctx_ == nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("failed to create " + modeName(mode) +
                                 " llama context for model: " + path);
    }
}

LlamaModelBase::~LlamaModelBase() {
    if (ctx_ != nullptr) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}

std::vector<llama_token> LlamaModelBase::tokenize(const std::string &text) const {
    if (model_ == nullptr) {
        throw std::runtime_error("cannot tokenize without a loaded llama model");
    }
    if (text.empty()) {
        return {};
    }

    const llama_vocab *vocab = llama_model_get_vocab(model_);
    if (vocab == nullptr) {
        throw std::runtime_error("failed to get llama vocabulary");
    }

    const auto text_len = static_cast<int32_t>(
        std::min<std::size_t>(text.size(),
                              static_cast<std::size_t>(std::numeric_limits<int32_t>::max())));

    // 第一次调用只获取精确 token 数，第二次按该容量真正写入 token。
    const int32_t needed =
        llama_tokenize(vocab, text.data(), text_len, nullptr, 0, true, true);
    if (needed == 0) {
        return {};
    }
    if (needed == std::numeric_limits<int32_t>::min()) {
        throw std::runtime_error("tokenization overflow");
    }

    const int32_t capacity = needed < 0 ? -needed : needed;
    std::vector<llama_token> tokens(static_cast<std::size_t>(capacity));
    const int32_t written =
        llama_tokenize(vocab, text.data(), text_len, tokens.data(), capacity, true, true);
    if (written < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }

    tokens.resize(static_cast<std::size_t>(written));
    return tokens;
}
