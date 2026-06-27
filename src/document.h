#pragma once

#include "chunk.h"

#include <cstddef>
#include <string>
#include <vector>

class DocumentProcessor {
public:
    // chunk_size/overlap_size 均按 UTF-8 字节数计算。
    DocumentProcessor(std::size_t chunk_size = 500, std::size_t overlap_size = 50);

    // 读取 UTF-8 小说文件，按章节和字节阈值切分为 Chunk。
    std::vector<Chunk> processNovel(const std::string &filepath) const;

private:
    std::size_t chunk_size_;
    std::size_t overlap_size_;
};
