#pragma once

#include "config.h"

#include <string>
#include <vector>

class MragApp {
public:
    explicit MragApp(AppConfig config);

    // 离线建库：切分文本、生成向量并写入二进制数据库。
    bool buildKnowledgeBase(const std::string &txt_path, const std::string &db_path);

    // 多文档建库：多个 TXT 会写入同一个向量数据库。
    bool buildKnowledgeBase(const std::vector<std::string> &txt_paths,
                            const std::string &db_path);

    // 增量追加：在已有 .mrag 数据库上追加新文档，不重建已有数据。
    bool appendKnowledgeBase(const std::vector<std::string> &txt_paths,
                             const std::string &db_path);

    // 在线检索循环：读取问题、检索片段并生成回答。
    bool chatLoop(const std::string &db_path);

private:
    struct BackendGuard {
        BackendGuard();
        ~BackendGuard();

        BackendGuard(const BackendGuard &) = delete;
        BackendGuard &operator=(const BackendGuard &) = delete;
    };

    BackendGuard backend_;
    AppConfig config_;
};
