#include "mragapp.h"

#include "chunk.h"
#include "document.h"
#include "embeddingengine.h"
#include "generationengine.h"
#include "vectordatabase.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <llama.h>

namespace {

std::string baseName(const std::string &path) {
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string sourceMetadata(const std::string &path, const std::string &metadata) {
    const std::string source = baseName(path);
    if (metadata.empty()) {
        return source;
    }
    return source + " / " + metadata;
}

}  // namespace

MragApp::BackendGuard::BackendGuard() {
    llama_backend_init();
}

MragApp::BackendGuard::~BackendGuard() {
    llama_backend_free();
}

MragApp::MragApp(AppConfig config)
    : backend_(),
      config_(std::move(config)) {}

bool MragApp::buildKnowledgeBase(const std::string &txt_path, const std::string &db_path) {
    return buildKnowledgeBase(std::vector<std::string>{txt_path}, db_path);
}

bool MragApp::buildKnowledgeBase(const std::vector<std::string> &txt_paths,
                                 const std::string &db_path) {
    if (txt_paths.empty()) {
        std::cerr << "no input documents provided\n";
        return false;
    }

    DocumentProcessor processor(config_.chunk_size, config_.overlap_size);
    VectorDatabase database;
    EmbeddingEngine embedding_engine(config_);
    uint64_t next_id = 0;

    for (std::size_t doc_index = 0; doc_index < txt_paths.size(); ++doc_index) {
        const std::string &txt_path = txt_paths[doc_index];
        std::vector<Chunk> chunks = processor.processNovel(txt_path);
        if (chunks.empty()) {
            std::cerr << "warning: no chunks generated from: " << txt_path << '\n';
            continue;
        }

        std::cerr << "document " << (doc_index + 1) << "/" << txt_paths.size()
                  << ": " << txt_path << '\n';
        std::cerr << "generated " << chunks.size() << " chunks\n";

        for (std::size_t i = 0; i < chunks.size(); ++i) {
            chunks[i].metadata = sourceMetadata(txt_path, chunks[i].metadata);
            chunks[i].embedding = embedding_engine.generateEmbedding(chunks[i].text);
            if (chunks[i].embedding.empty()) {
                std::cerr << "warning: empty embedding for chunk from "
                          << chunks[i].metadata << '\n';
                continue;
            }

            chunks[i].id = next_id++;
            database.insert(std::move(chunks[i]));
            std::cerr << "\rembedding chunks: " << (i + 1) << "/" << chunks.size()
                      << " | database chunks: " << database.size();
        }
        std::cerr << '\n';
    }

    if (database.empty()) {
        std::cerr << "no valid embeddings were generated\n";
        return false;
    }

    if (!database.saveToDisk(db_path)) {
        return false;
    }

    std::cerr << "saved " << database.size() << " chunks from " << txt_paths.size()
              << " document(s) to: " << db_path << '\n';
    return true;
}

bool MragApp::appendKnowledgeBase(const std::vector<std::string> &txt_paths,
                                   const std::string &db_path) {
    if (txt_paths.empty()) {
        std::cerr << "no input documents provided\n";
        return false;
    }

    // 先加载已有数据库
    VectorDatabase database;
    if (!database.loadFromDisk(db_path)) {
        std::cerr << "failed to load existing database: " << db_path
                  << "\nuse --ingest to create a new database first\n";
        return false;
    }

    const uint64_t start_id = database.maxId() + 1;
    const std::size_t existing_count = database.size();
    std::cerr << "loaded " << existing_count << " existing chunks from: " << db_path << '\n';
    std::cerr << "new chunks will start from id " << start_id << '\n';

    DocumentProcessor processor(config_.chunk_size, config_.overlap_size);
    EmbeddingEngine embedding_engine(config_);
    uint64_t next_id = start_id;
    std::size_t appended = 0;

    for (std::size_t doc_index = 0; doc_index < txt_paths.size(); ++doc_index) {
        const std::string &txt_path = txt_paths[doc_index];
        std::vector<Chunk> chunks = processor.processNovel(txt_path);
        if (chunks.empty()) {
            std::cerr << "warning: no chunks generated from: " << txt_path << '\n';
            continue;
        }

        std::cerr << "document " << (doc_index + 1) << "/" << txt_paths.size()
                  << ": " << txt_path << '\n';
        std::cerr << "generated " << chunks.size() << " chunks\n";

        for (std::size_t i = 0; i < chunks.size(); ++i) {
            chunks[i].metadata = sourceMetadata(txt_path, chunks[i].metadata);
            chunks[i].embedding = embedding_engine.generateEmbedding(chunks[i].text);
            if (chunks[i].embedding.empty()) {
                std::cerr << "warning: empty embedding for chunk from "
                          << chunks[i].metadata << '\n';
                continue;
            }

            chunks[i].id = next_id++;
            database.insert(std::move(chunks[i]));
            ++appended;
            std::cerr << "\rembedding chunks: " << (i + 1) << "/" << chunks.size()
                      << " | database chunks: " << database.size();
        }
        std::cerr << '\n';
    }

    if (appended == 0) {
        std::cerr << "no new chunks were appended\n";
        return false;
    }

    if (!database.saveToDisk(db_path)) {
        return false;
    }

    std::cerr << "appended " << appended << " new chunks from " << txt_paths.size()
              << " document(s)\n";
    std::cerr << "database now contains " << database.size() << " chunks total: "
              << db_path << '\n';
    return true;
}

namespace {

void printRecalledChunks(const std::vector<Chunk> &results) {
    std::cout << "\n━━━━━ 召回文本块 ━━━━━\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto &chunk = results[i];
        std::cout << "\n[块 " << (i + 1) << "] 来源: " << chunk.metadata
                  << "  |  ID: " << chunk.id << '\n';

        // 显示文本内容，截取前300字节用于快速浏览
        const std::size_t preview_len = std::min<std::size_t>(300, chunk.text.size());
        std::cout << "内容: ";
        std::cout.write(chunk.text.data(), static_cast<std::streamsize>(preview_len));
        if (chunk.text.size() > preview_len) {
            std::cout << "…";
        }
        std::cout << '\n';
    }
    std::cout << "━━━━━━━━━━━━━━━━━━━━\n\n";
}

}  // namespace

bool MragApp::chatLoop(const std::string &db_path) {
    VectorDatabase database;
    if (!database.loadFromDisk(db_path)) {
        return false;
    }
    if (database.empty()) {
        std::cerr << "database is empty: " << db_path << '\n';
        return false;
    }

    std::cerr << "loaded " << database.size() << " chunks from: " << db_path << '\n';
    std::cerr << "type an empty line to exit\n";

    EmbeddingEngine embedding_engine(config_);
    GenerationEngine generation_engine(config_);

    std::string query;
    while (true) {
        std::cout << "\n> ";
        if (!std::getline(std::cin, query) || query.empty()) {
            break;
        }

        const std::vector<float> query_embedding = embedding_engine.generateEmbedding(query);
        const std::vector<Chunk> results = database.search(query_embedding, config_.top_k);
        if (results.empty()) {
            std::cout << "没有检索到相关片段。\n";
            continue;
        }

        // 根据配置决定是否打印召回的文本块详情
        if (config_.show_recalled_chunks) {
            printRecalledChunks(results);
        }

        std::cout << "回答：";
        if (!generation_engine.generateStream(query, results)) {
            std::cerr << "failed to generate answer\n";
        }
    }

    return true;
}
