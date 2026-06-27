#pragma once

#include "chunk.h"

#include <cstddef>
#include <string>
#include <vector>

class VectorDatabase {
public:
    void clear();
    bool empty() const;
    std::size_t size() const;

    // 插入单个或多个 Chunk；右值版本用于建库阶段减少拷贝。
    void insert(const Chunk &chunk);
    void insert(Chunk &&chunk);
    void insert(const std::vector<Chunk> &chunks);
    void insert(std::vector<Chunk> &&chunks);

    // 返回与 query_emb 余弦相似度最高的 top_k 个 Chunk。
    std::vector<Chunk> search(const std::vector<float> &query_emb,
                              std::size_t top_k) const;

    // 二进制保存/加载数据库，文件头包含魔数和版本号。
    bool saveToDisk(const std::string &path) const;
    bool loadFromDisk(const std::string &path);

    const std::vector<Chunk> &chunks() const;

    // 返回已存储 Chunk 中的最大 id，用于增量追加时确定起始编号；
    // 数据库为空时返回 0。
    uint64_t maxId() const;

private:
    std::vector<Chunk> chunks_;
};
