#include "chunk.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace {

void writeBytes(std::ofstream &out, const char *data, std::streamsize size) {
    out.write(data, size);
    if (!out) {
        throw std::runtime_error("failed to write chunk data");
    }
} // 写入数据

void readBytes(std::ifstream &in, char *data, std::streamsize size) {
    in.read(data, size);
    if (!in) {
        throw std::runtime_error("failed to read chunk data");
    }
} // 读取数据

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
        throw std::runtime_error(std::string("chunk ") + field_name + " is too large");
    }
    return static_cast<std::size_t>(value);
}

}  // namespace

Chunk::Chunk() = default;

Chunk::Chunk(uint64_t id_value,
             std::string text_value,
             std::string metadata_value,
             std::vector<float> embedding_value)
    : id(id_value),
      text(std::move(text_value)),
      metadata(std::move(metadata_value)),
      embedding(std::move(embedding_value)) {}

void Chunk::serialize(std::ofstream &out) const {
    writeUint64(out, id);

    writeUint64(out, static_cast<uint64_t>(text.size()));
    if (!text.empty()) {
        writeBytes(out, text.data(), static_cast<std::streamsize>(text.size()));
    }

    writeUint64(out, static_cast<uint64_t>(metadata.size()));
    if (!metadata.empty()) {
        writeBytes(out, metadata.data(), static_cast<std::streamsize>(metadata.size()));
    }

    writeUint64(out, static_cast<uint64_t>(embedding.size()));
    if (!embedding.empty()) {
        const auto byte_count =
            static_cast<std::streamsize>(embedding.size() * sizeof(float));
        writeBytes(out, reinterpret_cast<const char *>(embedding.data()), byte_count);
    }
}

Chunk Chunk::deserialize(std::ifstream &in) {
    Chunk chunk;
    chunk.id = readUint64(in);

    const auto text_size = checkedSize(readUint64(in), "text size");
    chunk.text.resize(text_size);
    if (!chunk.text.empty()) {
        readBytes(in, chunk.text.data(), static_cast<std::streamsize>(chunk.text.size()));
    }

    const auto metadata_size = checkedSize(readUint64(in), "metadata size");
    chunk.metadata.resize(metadata_size);
    if (!chunk.metadata.empty()) {
        readBytes(in,
                  chunk.metadata.data(),
                  static_cast<std::streamsize>(chunk.metadata.size()));
    }

    const auto embedding_size = checkedSize(readUint64(in), "embedding size");
    chunk.embedding.resize(embedding_size);
    if (!chunk.embedding.empty()) {
        const auto byte_count =
            static_cast<std::streamsize>(chunk.embedding.size() * sizeof(float));
        readBytes(in, reinterpret_cast<char *>(chunk.embedding.data()), byte_count);
    }

    return chunk;
}
