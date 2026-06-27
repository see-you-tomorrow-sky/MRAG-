#include "vectordatabase.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace {

// 数据库文件头：魔数用于识别 MRAG 文件，版本号用于后续格式升级。
constexpr uint32_t kMagic = 0x4D524147;  // "MRAG"
constexpr uint32_t kVersion = 1;

void writeBytes(std::ofstream &out, const char *data, std::streamsize size) {
    out.write(data, size);
    if (!out) {
        throw std::runtime_error("failed to write vector database");
    }
}

void readBytes(std::ifstream &in, char *data, std::streamsize size) {
    in.read(data, size);
    if (!in) {
        throw std::runtime_error("failed to read vector database");
    }
}

void writeUint32(std::ofstream &out, uint32_t value) {
    writeBytes(out, reinterpret_cast<const char *>(&value), sizeof(value));
}

uint32_t readUint32(std::ifstream &in) {
    uint32_t value{};
    readBytes(in, reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}

void writeUint64(std::ofstream &out, uint64_t value) {
    writeBytes(out, reinterpret_cast<const char *>(&value), sizeof(value));
}

uint64_t readUint64(std::ifstream &in) {
    uint64_t value{};
    readBytes(in, reinterpret_cast<char *>(&value), sizeof(value));
    return value;
}

std::size_t checkedSize(uint64_t value, const char *field_name) {
    if (value > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("vector database ") + field_name +
                                 " is too large");
    }
    return static_cast<std::size_t>(value);
}

// 返回合法余弦相似度 [-1, 1]；-2 表示维度不匹配或零向量。
float cosineSimilarity(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    if (lhs.empty() || lhs.size() != rhs.size()) {
        return -2.0F;
    }

    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        dot += static_cast<double>(lhs[i]) * rhs[i];
        lhs_norm += static_cast<double>(lhs[i]) * lhs[i];
        rhs_norm += static_cast<double>(rhs[i]) * rhs[i];
    }

    if (lhs_norm == 0.0 || rhs_norm == 0.0) {
        return -2.0F;
    }

    return static_cast<float>(dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm)));
}

struct SearchResult {
    float score{};
    std::size_t index{};
};

// priority_queue 默认大顶堆，这里反过来让最低分排在堆顶。
struct WorseFirst {
    bool operator()(const SearchResult &lhs, const SearchResult &rhs) const {
        return lhs.score > rhs.score;
    }
};

}  // namespace

void VectorDatabase::clear() {
    chunks_.clear();
}

bool VectorDatabase::empty() const {
    return chunks_.empty();
}

std::size_t VectorDatabase::size() const {
    return chunks_.size();
}

void VectorDatabase::insert(const Chunk &chunk) {
    chunks_.push_back(chunk);
}

void VectorDatabase::insert(Chunk &&chunk) {
    chunks_.push_back(std::move(chunk));
}

void VectorDatabase::insert(const std::vector<Chunk> &chunks) {
    chunks_.insert(chunks_.end(), chunks.begin(), chunks.end());
}

void VectorDatabase::insert(std::vector<Chunk> &&chunks) {
    chunks_.reserve(chunks_.size() + chunks.size());
    for (auto &chunk : chunks) {
        chunks_.push_back(std::move(chunk));
    }
}

std::vector<Chunk> VectorDatabase::search(const std::vector<float> &query_emb,
                                          std::size_t top_k) const {
    if (query_emb.empty() || top_k == 0) {
        return {};
    }

    // 用最小堆保存当前 top-k，堆顶永远是候选集合里分数最低的项。
    std::priority_queue<SearchResult, std::vector<SearchResult>, WorseFirst> heap;
    for (std::size_t i = 0; i < chunks_.size(); ++i) {
        const float score = cosineSimilarity(query_emb, chunks_[i].embedding);
        if (score < -1.0F) {
            // 跳过无效向量，合法余弦分数最小也只会到 -1。
            continue;
        }

        if (heap.size() < top_k) {
            heap.push({score, i});
        } else if (score > heap.top().score) {
            heap.pop();
            heap.push({score, i});
        }
    }

    std::vector<SearchResult> ranked;
    ranked.reserve(heap.size());
    while (!heap.empty()) {
        ranked.push_back(heap.top());
        heap.pop();
    }
    // 堆弹出顺序是从低到高，返回前按相似度从高到低排列。
    std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<Chunk> results;
    results.reserve(ranked.size());
    for (const auto &item : ranked) {
        results.push_back(chunks_[item.index]);
    }
    return results;
}

bool VectorDatabase::saveToDisk(const std::string &path) const {
    try {
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            std::cerr << "failed to open database for writing: " << path << '\n';
            return false;
        }

        writeUint32(out, kMagic);
        writeUint32(out, kVersion);
        // 文件结构：magic/version/chunk_count/chunk...
        writeUint64(out, static_cast<uint64_t>(chunks_.size()));
        for (const auto &chunk : chunks_) {
            chunk.serialize(out);
        }
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "failed to save database '" << path << "': " << ex.what() << '\n';
        return false;
    }
}

bool VectorDatabase::loadFromDisk(const std::string &path) {
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            std::cerr << "failed to open database for reading: " << path << '\n';
            return false;
        }

        const auto magic = readUint32(in);
        if (magic != kMagic) {
            std::cerr << "invalid database magic: " << path << '\n';
            return false;
        }

        const auto version = readUint32(in);
        if (version != kVersion) {
            std::cerr << "unsupported database version " << version << ": " << path
                      << '\n';
            return false;
        }

        // 先读入临时 vector，全部成功后再替换原数据库。
        const auto count = checkedSize(readUint64(in), "chunk count");
        std::vector<Chunk> loaded;
        loaded.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            loaded.push_back(Chunk::deserialize(in));
        }

        chunks_ = std::move(loaded);
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "failed to load database '" << path << "': " << ex.what() << '\n';
        return false;
    }
}

uint64_t VectorDatabase::maxId() const {
    if (chunks_.empty()) {
        return 0;
    }
    uint64_t max = 0;
    for (const auto &chunk : chunks_) {
        if (chunk.id > max) {
            max = chunk.id;
        }
    }
    return max;
}

const std::vector<Chunk> &VectorDatabase::chunks() const {
    return chunks_;
}
