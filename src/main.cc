#include "config.h"
#include "mragapp.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <llama.h>

namespace {

void silentLlamaLog(ggml_log_level, const char *, void *) {}

void printUsage(const char *program) {
    std::cerr << "Usage:\n"
              << "  " << program << " --ingest  <novel1.txt> [novel2.txt ...] <database.mrag>\n"
              << "  " << program << " --append <novel1.txt> [novel2.txt ...] <database.mrag>\n"
              << "  " << program << " --chat   <database.mrag>\n";
}

int runIngest(const AppConfig &config,
              const std::vector<std::string> &txt_paths,
              const std::string &db_path) {
    MragApp app(config);
    return app.buildKnowledgeBase(txt_paths, db_path) ? 0 : 1;
}

int runAppend(const AppConfig &config,
               const std::vector<std::string> &txt_paths,
               const std::string &db_path) {
    MragApp app(config);
    return app.appendKnowledgeBase(txt_paths, db_path) ? 0 : 1;
}

int runChat(const AppConfig &config, const std::string &db_path) {
    MragApp app(config);
    return app.chatLoop(db_path) ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    llama_log_set(silentLlamaLog, nullptr);

    try {
        const AppConfig config = loadConfig();

        const auto is_ingest = std::string(argv[1]) == "--ingest";
        const auto is_append = std::string(argv[1]) == "--append";

        if ((is_ingest || is_append) && argc >= 4) {
            std::vector<std::string> txt_paths;
            txt_paths.reserve(static_cast<std::size_t>(argc - 3));
            for (int i = 2; i < argc - 1; ++i) {
                txt_paths.emplace_back(argv[i]);
            }
            if (is_append) {
                return runAppend(config, txt_paths, argv[argc - 1]);
            }
            return runIngest(config, txt_paths, argv[argc - 1]);
        }

        if (argc == 3 && std::string(argv[1]) == "--chat") {
            return runChat(config, argv[2]);
        }

        printUsage(argv[0]);
        return 1;
    } catch (const std::exception &ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 1;
    }
}
