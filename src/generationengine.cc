#include "generationengine.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BatchGuard {
    explicit BatchGuard(llama_batch batch_value) : batch(batch_value) {}
    ~BatchGuard() {
        llama_batch_free(batch);
    }

    BatchGuard(const BatchGuard &) = delete;
    BatchGuard &operator=(const BatchGuard &) = delete;

    llama_batch batch;
};

struct SamplerGuard {
    explicit SamplerGuard(llama_sampler *sampler_value) : sampler(sampler_value) {}
    ~SamplerGuard() {
        if (sampler != nullptr) {
            llama_sampler_free(sampler);
        }
    }

    SamplerGuard(const SamplerGuard &) = delete;
    SamplerGuard &operator=(const SamplerGuard &) = delete;

    llama_sampler *sampler{};
};

void addSampler(llama_sampler *chain, llama_sampler *sampler, const char *name) {
    if (sampler == nullptr) {
        throw std::runtime_error(std::string("failed to allocate sampler: ") + name);
    }
    llama_sampler_chain_add(chain, sampler);
}

void fillBatch(llama_batch &batch,
               const std::vector<llama_token> &tokens,
               int32_t start_pos,
               bool logits_last_only) {
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<std::size_t>(i)];
        batch.pos[i] = start_pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (!logits_last_only || i == batch.n_tokens - 1) ? 1 : 0;
    }
}

std::string formatContext(const std::vector<Chunk> &chunks, std::size_t count) {
    std::ostringstream out;
    count = std::min(count, chunks.size());
    for (std::size_t i = 0; i < count; ++i) {
        out << "[片段 " << (i + 1) << " | " << chunks[i].metadata << " | #"
            << chunks[i].id << "]\n"
            << chunks[i].text << "\n\n";
    }
    return out.str();
}

std::string buildPrompt(const std::string &query,
                        const std::vector<Chunk> &chunks,
                        std::size_t chunk_count) {
    std::ostringstream prompt;
    prompt << "<|im_start|>system\n"
           << "你是一个专业的小说阅读助手，擅长根据原文片段准确回答读者问题。\n\n"
           << "回答规则：\n"
           << "1. 【必须】只能依据下面 [参考上下文] 中提供的片段作答，不得使用你自己的知识。\n"
           << "2. 回答时先给出结论，再引用原文片段中的具体细节作为证据。\n"
           << "3. 引用时请标注片段编号，例如「根据[片段1]…」。\n"
           << "4. 如果参考上下文不足以回答问题，请明确说「根据当前片段无法得到答案」，"
           << "不要猜测或编造。\n"
           << "5. 答案尽量完整详尽，不要过度精简。<|im_end|>\n"
           << "<|im_start|>user\n"
           << "[参考上下文]\n"
           << formatContext(chunks, chunk_count)
           << "用户问题：" << query << "<|im_end|>\n"
           << "<|im_start|>assistant\n";
    return prompt.str();
}

std::string tokenToPiece(const llama_vocab *vocab, llama_token token) {
    std::string piece(32, '\0');
    int32_t written =
        llama_token_to_piece(vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
    if (written < 0) {
        piece.resize(static_cast<std::size_t>(-written));
        written = llama_token_to_piece(
            vocab, token, piece.data(), static_cast<int32_t>(piece.size()), 0, false);
    }
    if (written <= 0) {
        return {};
    }
    piece.resize(static_cast<std::size_t>(written));
    return piece;
}

}  // namespace

GenerationEngine::GenerationEngine(const AppConfig &config)
    : LlamaModelBase(config, ModelMode::Generation) {}

bool GenerationEngine::generateStream(const std::string &query,
                                      const std::vector<Chunk> &chunks) {
    if (chunks.empty()) {
        std::cout << "根据当前片段无法得到答案\n";
        return true;
    }

    llama_memory_clear(llama_get_memory(ctx_), true);

    const uint32_t n_ctx = llama_n_ctx(ctx_);
    const uint32_t n_batch = llama_n_batch(ctx_);
    std::size_t chunk_count = chunks.size();
    std::vector<llama_token> prompt_tokens;
    while (chunk_count > 0) {
        prompt_tokens = tokenize(buildPrompt(query, chunks, chunk_count));
        if (!prompt_tokens.empty() && prompt_tokens.size() < n_ctx) {
            break;
        }
        --chunk_count;
    }

    if (prompt_tokens.empty()) {
        std::cerr << "empty prompt after tokenization\n";
        return false;
    }
    if (prompt_tokens.size() >= n_ctx) {
        std::cerr << "warning: prompt exceeds context window, skip this query\n";
        return false;
    }
    if (prompt_tokens.size() > static_cast<std::size_t>(n_batch)) {
        std::cerr << "warning: prompt exceeds generation batch limit, skip this query\n";
        return false;
    }

    BatchGuard prefill_guard(
        llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1));
    if (prefill_guard.batch.token == nullptr || prefill_guard.batch.pos == nullptr ||
        prefill_guard.batch.n_seq_id == nullptr || prefill_guard.batch.seq_id == nullptr ||
        prefill_guard.batch.logits == nullptr) {
        throw std::runtime_error("failed to allocate generation prefill batch");
    }

    fillBatch(prefill_guard.batch, prompt_tokens, 0, true);
    if (llama_decode(ctx_, prefill_guard.batch) != 0) {
        std::cerr << "llama_decode failed during prompt prefill\n";
        return false;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    SamplerGuard sampler(llama_sampler_chain_init(sampler_params));
    if (sampler.sampler == nullptr) {
        throw std::runtime_error("failed to allocate llama sampler chain");
    }
    addSampler(sampler.sampler, llama_sampler_init_top_k(config_.top_k_sampler), "top_k");
    addSampler(sampler.sampler, llama_sampler_init_top_p(config_.top_p, 1), "top_p");
    addSampler(sampler.sampler, llama_sampler_init_temp(config_.temperature), "temperature");
    addSampler(sampler.sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED), "dist");

    const llama_vocab *vocab = llama_model_get_vocab(model_);
    if (vocab == nullptr) {
        throw std::runtime_error("failed to get llama vocabulary");
    }

    BatchGuard decode_guard(llama_batch_init(1, 0, 1));
    if (decode_guard.batch.token == nullptr || decode_guard.batch.pos == nullptr ||
        decode_guard.batch.n_seq_id == nullptr || decode_guard.batch.seq_id == nullptr ||
        decode_guard.batch.logits == nullptr) {
        throw std::runtime_error("failed to allocate generation decode batch");
    }

    int32_t position = static_cast<int32_t>(prompt_tokens.size());
    const int max_tokens = std::max(0, config_.max_output_tokens);
    for (int i = 0; i < max_tokens && static_cast<uint32_t>(position) < n_ctx; ++i) {
        const llama_token token = llama_sampler_sample(sampler.sampler, ctx_, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        llama_sampler_accept(sampler.sampler, token);
        std::cout << tokenToPiece(vocab, token) << std::flush;

        std::vector<llama_token> next_token{token};
        fillBatch(decode_guard.batch, next_token, position, true);
        if (llama_decode(ctx_, decode_guard.batch) != 0) {
            std::cerr << "\nllama_decode failed during generation\n";
            return false;
        }
        ++position;
    }

    std::cout << '\n';
    return true;
}
