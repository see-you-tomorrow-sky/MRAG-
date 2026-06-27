#include "embeddingengine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool isContinuationByte(unsigned char ch) {
    return (ch & 0xC0) == 0x80;
}

std::size_t utf8CharLength(unsigned char ch) {
    if ((ch & 0x80) == 0x00) {
        return 1;
    }
    if ((ch & 0xE0) == 0xC0) {
        return 2;
    }
    if ((ch & 0xF0) == 0xE0) {
        return 3;
    }
    if ((ch & 0xF8) == 0xF0) {
        return 4;
    }
    return 0;
}

// 删除残缺或非法 UTF-8 字节，避免 tokenizer 在坏输入上失败。
std::string sanitizeUtf8(const std::string &text) {
    std::string result;
    result.reserve(text.size());

    for (std::size_t i = 0; i < text.size();) {
        const auto first = static_cast<unsigned char>(text[i]);
        const std::size_t length = utf8CharLength(first);
        if (length == 0 || i + length > text.size()) {
            ++i;
            continue;
        }

        bool valid = true;
        for (std::size_t j = 1; j < length; ++j) {
            if (!isContinuationByte(static_cast<unsigned char>(text[i + j]))) {
                valid = false;
                break;
            }
        }

        if (valid) {
            result.append(text, i, length);
            i += length;
        } else {
            ++i;
        }
    }

    return result;
}

struct BatchGuard {
    explicit BatchGuard(llama_batch batch_value) : batch(batch_value) {}
    ~BatchGuard() {
        llama_batch_free(batch);
    }

    BatchGuard(const BatchGuard &) = delete;
    BatchGuard &operator=(const BatchGuard &) = delete;

    llama_batch batch;
};

}  // namespace

EmbeddingEngine::EmbeddingEngine(const AppConfig &config)
    : LlamaModelBase(config, ModelMode::Embedding) {}

std::vector<float> EmbeddingEngine::generateEmbedding(const std::string &text) {
    const std::string clean_text = sanitizeUtf8(text);
    auto tokens = tokenize(clean_text);
    if (tokens.empty()) {
        return {};
    }

    const int token_limit_config = std::min(config_.emb_n_ctx, config_.emb_n_batch);
    if (token_limit_config <= 0) {
        throw std::runtime_error("embedding token limit is zero");
    }
    const auto token_limit = static_cast<std::size_t>(token_limit_config);
    if (tokens.size() > token_limit) {
        // 截断发生在 token 边界；长文本语义边界主要由 DocumentProcessor 控制。
        tokens.resize(token_limit);
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    BatchGuard guard(llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1));
    auto &batch = guard.batch;
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr ||
        batch.seq_id == nullptr || batch.logits == nullptr) {
        throw std::runtime_error("failed to allocate llama embedding batch");
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<std::size_t>(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1) ? 1 : 0;
    }

    if (llama_decode(ctx_, batch) != 0) {
        throw std::runtime_error("llama_decode failed while generating embedding");
    }

    const int embedding_size = llama_model_n_embd(model_);
    if (embedding_size <= 0) {
        throw std::runtime_error("embedding model returned invalid embedding size");
    }

    const float *embedding = llama_get_embeddings_seq(ctx_, 0);
    if (embedding == nullptr) {
        embedding = llama_get_embeddings(ctx_);
    }
    if (embedding == nullptr) {
        throw std::runtime_error("failed to get embedding from llama context");
    }

    std::vector<float> result(static_cast<std::size_t>(embedding_size));
    std::memcpy(result.data(),
                embedding,
                static_cast<std::size_t>(embedding_size) * sizeof(float));
    return result;
}
