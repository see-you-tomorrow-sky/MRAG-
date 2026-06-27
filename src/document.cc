#include "document.h"

#include <algorithm>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr const char *kDefaultChapter = "未命名章节";

// UTF-8 后续字节形如 10xxxxxx，不能作为切割起点。
bool isUtf8Continuation(unsigned char ch) {
    return (ch & 0xC0) == 0x80;
}

// 向前回退到合法 UTF-8 字符边界，避免截断中文字符。
std::size_t previousUtf8Boundary(const std::string &text, std::size_t pos) {
    pos = std::min(pos, text.size());
    while (pos > 0 && pos < text.size() &&
           isUtf8Continuation(static_cast<unsigned char>(text[pos]))) {
        --pos;
    }
    return pos;
}

// 向后推进到合法 UTF-8 字符边界，用于计算 overlap 起点。
std::size_t nextUtf8Boundary(const std::string &text, std::size_t pos) {
    pos = std::min(pos, text.size());
    while (pos < text.size() && isUtf8Continuation(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

bool endsWithAny(const std::string &text,
                 std::size_t pos,
                 const std::vector<std::string> &markers) {
    for (const auto &marker : markers) {
        if (pos >= marker.size() &&
            text.compare(pos - marker.size(), marker.size(), marker) == 0) {
            return true;
        }
    }
    return false;
}

std::size_t previousCutPosition(const std::string &text, std::size_t pos) {
    if (pos == 0) {
        return 0;
    }
    return previousUtf8Boundary(text, pos - 1);
}

// 在目标大小附近优先按句末标点或换行切分，避免生成过短块。
std::size_t findPreferredCut(const std::string &text, std::size_t limit) {
    static const std::vector<std::string> markers = {"。", "！", "？", "\n"};
    limit = previousUtf8Boundary(text, limit);
    const auto min_cut = previousUtf8Boundary(text, limit / 2);

    std::size_t pos = limit;
    while (pos > min_cut) {
        pos = previousUtf8Boundary(text, pos);
        if (pos <= min_cut) {
            break;
        }
        if (endsWithAny(text, pos, markers)) {
            return pos;
        }
        pos = previousCutPosition(text, pos);
    }

    return limit;
}

// 保留上一块末尾 overlap_size 字节，起点同样对齐到 UTF-8 边界。
std::size_t overlapStart(const std::string &chunk_text, std::size_t overlap_size) {
    if (overlap_size == 0 || chunk_text.size() <= overlap_size) {
        return chunk_text.size();
    }
    return nextUtf8Boundary(chunk_text, chunk_text.size() - overlap_size);
}

void removeCarriageReturn(std::string &line) {
    line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
}

void appendLine(std::string &buffer, const std::string &line) {
    buffer += line;
    buffer += '\n';
}

// 将缓冲区尽量切成 Chunk；flush_all=true 时会把尾部残余也写出。
void flushBuffer(std::vector<Chunk> &chunks,
                 std::string &buffer,
                 const std::string &metadata,
                 std::size_t &next_id,
                 std::size_t chunk_size,
                 std::size_t overlap_size,
                 bool flush_all) {
    while (buffer.size() >= chunk_size || (flush_all && !buffer.empty())) {
        const auto limit = flush_all ? buffer.size() : chunk_size;
        std::size_t cut = findPreferredCut(buffer, limit);
        if (cut == 0) {
            cut = previousUtf8Boundary(buffer, limit);
        }
        if (cut == 0) {
            break;
        }

        std::string chunk_text = buffer.substr(0, cut);
        chunks.emplace_back(next_id++, std::move(chunk_text), metadata);

        // 未消费文本前拼接 overlap，保证相邻块之间有上下文连续性。
        if (cut >= buffer.size()) {
            buffer.clear();
            break;
        }

        const auto keep_from = overlapStart(chunks.back().text, overlap_size);
        buffer = chunks.back().text.substr(keep_from) + buffer.substr(cut);

        if (!flush_all && buffer.size() < chunk_size) {
            break;
        }
    }
}

}  // namespace

DocumentProcessor::DocumentProcessor(std::size_t chunk_size, std::size_t overlap_size)
    : chunk_size_(chunk_size == 0 ? 500 : chunk_size),
      overlap_size_(std::min(overlap_size, chunk_size_ - 1)) {}

std::vector<Chunk> DocumentProcessor::processNovel(const std::string &filepath) const {
    std::ifstream input(filepath);
    if (!input) {
        throw std::runtime_error("failed to open novel file: " + filepath);
    }

    static const std::regex chapter_pattern(R"(^(第.*[章节回卷]|Chapter\s+\d+))",
                                            std::regex::ECMAScript);

    std::vector<Chunk> chunks;
    std::string current_metadata = kDefaultChapter;
    std::string buffer;
    std::string line;
    std::size_t next_id = 0;

    while (std::getline(input, line)) {
        // 兼容 Windows 文本文件，std::getline 会保留行尾的 '\r'。
        removeCarriageReturn(line);

        // 新章节出现前，先把上一章节缓冲区剩余内容落成 Chunk。
        if (std::regex_search(line, chapter_pattern)) {
            flushBuffer(chunks,
                        buffer,
                        current_metadata,
                        next_id,
                        chunk_size_,
                        overlap_size_,
                        true);
            current_metadata = line;
        }

        appendLine(buffer, line);
        flushBuffer(chunks,
                    buffer,
                    current_metadata,
                    next_id,
                    chunk_size_,
                    overlap_size_,
                    false);
    }

    flushBuffer(chunks,
                buffer,
                current_metadata,
                next_id,
                chunk_size_,
                overlap_size_,
                true);

    return chunks;
}
