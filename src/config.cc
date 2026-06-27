#include "config.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

template <typename T>
void assignIfPresent(const nlohmann::json &object, const char *key, T &target) {
    if (object.contains(key) && !object.at(key).is_null()) {
        target = object.at(key).get<T>();
    }
}

void assignSizeIfPresent(const nlohmann::json &object, const char *key, std::size_t &target) {
    if (!object.contains(key) || object.at(key).is_null()) {
        return;
    }

    const auto &value = object.at(key);
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error(std::string("config field '") + key + "' is too large");
        }
        target = static_cast<std::size_t>(number);
        return;
    }

    if (value.is_number_integer()) {
        const auto number = value.get<int64_t>();
        if (number < 0) {
            throw std::runtime_error(std::string("config field '") + key +
                                     "' must not be negative");
        }
        target = static_cast<std::size_t>(number);
        return;
    }

    throw std::runtime_error(std::string("config field '") + key +
                             "' must be an integer");
}

const nlohmann::json *findObject(const nlohmann::json &root, const char *key) {
    if (!root.contains(key) || !root.at(key).is_object()) {
        return nullptr;
    }
    return &root.at(key);
}

void requirePositive(int value, const char *field_name) {
    if (value <= 0) {
        throw std::runtime_error(std::string("config field '") + field_name +
                                 "' must be positive");
    }
}

void validateConfig(const AppConfig &config) {
    if (config.chunk_size == 0) {
        throw std::runtime_error("config field 'document.chunk_size' must be positive");
    }
    if (config.overlap_size >= config.chunk_size) {
        throw std::runtime_error(
            "config field 'document.overlap_size' must be smaller than chunk_size");
    }
    if (config.top_k == 0) {
        throw std::runtime_error("config field 'retrieval.top_k' must be positive");
    }

    requirePositive(config.gen_n_ctx, "generation.n_ctx");
    requirePositive(config.gen_n_batch, "generation.n_batch");
    requirePositive(config.gen_n_ubatch, "generation.n_ubatch");
    if (config.gen_n_ubatch > config.gen_n_batch) {
        throw std::runtime_error(
            "config field 'generation.n_ubatch' must not exceed generation.n_batch");
    }
    requirePositive(config.max_output_tokens, "generation.max_output_tokens");
    requirePositive(config.top_k_sampler, "generation.top_k");
    if (config.temperature < 0.0F) {
        throw std::runtime_error("config field 'generation.temperature' must not be negative");
    }
    if (config.top_p <= 0.0F || config.top_p > 1.0F) {
        throw std::runtime_error("config field 'generation.top_p' must be in (0, 1]");
    }

    requirePositive(config.emb_n_ctx, "embedding.n_ctx");
    requirePositive(config.emb_n_batch, "embedding.n_batch");
    requirePositive(config.emb_n_ubatch, "embedding.n_ubatch");
    if (config.emb_n_ubatch > config.emb_n_batch) {
        throw std::runtime_error(
            "config field 'embedding.n_ubatch' must not exceed embedding.n_batch");
    }
}

}  // namespace

AppConfig loadConfig(const std::string &path) {
    AppConfig config;

    std::ifstream input(path);
    if (!input) {
        return config;
    }
    if (input.peek() == std::ifstream::traits_type::eof()) {
        return config;
    }

    nlohmann::json root;
    try {
        input >> root;
    } catch (const nlohmann::json::exception &ex) {
        throw std::runtime_error("failed to parse config file '" + path + "': " + ex.what());
    }

    if (const auto *models = findObject(root, "models")) {
        assignIfPresent(*models, "embedding", config.emb_model_path);
        assignIfPresent(*models, "generation", config.gen_model_path);
        assignIfPresent(*models, "n_gpu_layers", config.n_gpu_layers);
    }

    if (const auto *document = findObject(root, "document")) {
        assignSizeIfPresent(*document, "chunk_size", config.chunk_size);
        assignSizeIfPresent(*document, "overlap_size", config.overlap_size);
    }

    if (const auto *retrieval = findObject(root, "retrieval")) {
        assignSizeIfPresent(*retrieval, "top_k", config.top_k);
        assignIfPresent(*retrieval, "show_recalled_chunks", config.show_recalled_chunks);
    }

    if (const auto *generation = findObject(root, "generation")) {
        assignIfPresent(*generation, "n_ctx", config.gen_n_ctx);
        assignIfPresent(*generation, "n_batch", config.gen_n_batch);
        assignIfPresent(*generation, "n_ubatch", config.gen_n_ubatch);
        assignIfPresent(*generation, "max_output_tokens", config.max_output_tokens);
        assignIfPresent(*generation, "temperature", config.temperature);
        assignIfPresent(*generation, "top_k", config.top_k_sampler);
        assignIfPresent(*generation, "top_p", config.top_p);
    }

    if (const auto *embedding = findObject(root, "embedding")) {
        assignIfPresent(*embedding, "n_ctx", config.emb_n_ctx);
        assignIfPresent(*embedding, "n_batch", config.emb_n_batch);
        assignIfPresent(*embedding, "n_ubatch", config.emb_n_ubatch);
    }

    validateConfig(config);
    return config;
}
