#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

struct Chunk {
    uint64_t id{};                 // 全局唯一递增 ID
    std::string text;              // 文本内容
    std::string metadata;          // 所属章节名
    std::vector<float> embedding;  // 向量，建库阶段填充

    Chunk();
    Chunk(uint64_t id,
          std::string text,
          std::string metadata,
          std::vector<float> embedding = {});

    // 二进制序列化接口
    void serialize(std::ofstream &out) const;

    // 二进制反序列化接口
    static Chunk deserialize(std::ifstream &in);
};
