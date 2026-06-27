# MRAG 设计说明

## 1. 项目目标

本项目实现一个基于 `llama.cpp` C API 的本地 RAG 问答系统。系统在本地完成小说文本切分、向量生成、向量检索和大模型回答，不依赖云端 API，适合对一个或多个指定小说文本进行离线知识库构建、增量追加和命令行问答。

## 2. 总体架构

系统分为离线建库、增量追加和在线问答三个流程。

```text
离线建库（--ingest）：
一个或多个 TXT 小说
  -> DocumentProcessor 读取并切分文本
  -> EmbeddingEngine 生成文本向量
  -> VectorDatabase 保存 Chunk 和 embedding
  -> 写入 .mrag 二进制数据库

增量追加（--append）：
一个或多个新 TXT 小说
  -> 加载已有 .mrag 数据库
  -> 读取最大 Chunk ID（maxId）
  -> DocumentProcessor 读取并切分新文本
  -> EmbeddingEngine 生成新文本向量
  -> 从 max_id + 1 开始分配 ID，追加到 VectorDatabase
  -> 合并写入原 .mrag 数据库

在线问答（--chat）：
用户问题
  -> EmbeddingEngine 生成问题向量
  -> VectorDatabase 检索 top-k 相关 Chunk
  -> GenerationEngine 组装参考上下文和 Prompt
  -> llama.cpp 本地生成流式回答
```

`MragApp` 作为应用主体协调上述模块，`main.cc` 负责解析命令行参数并选择运行模式。

项目通过 CMake `FetchContent` 集成 `llama.cpp` 和 `nlohmann/json`。其中 `llama.cpp` 固定到 `b9703` 标签，避免上游 `master` 分支变化导致课程验收时无法复现编译环境。

## 3. 模块划分

| 模块 | 对应文件 | 主要职责 |
| --- | --- | --- |
| 配置管理 | `config.h` / `config.cc` | 定义 `AppConfig`，从 `config.json` 读取模型、文档、检索和生成参数 |
| 数据结构 | `chunk.h` / `chunk.cc` | 定义 `Chunk`，实现二进制序列化和反序列化 |
| 文档处理 | `document.h` / `document.cc` | 读取 UTF-8 TXT 小说，识别章节标题，按字节阈值和 UTF-8 边界切分文本 |
| 向量数据库 | `vectordatabase.h` / `vectordatabase.cc` | 存储文本块向量，执行余弦相似度检索，提供 `maxId()` 查询，保存和加载 `.mrag` 文件 |
| 模型基类 | `llamamodel.h` / `llamamodel.cc` | 封装 `llama_model*` 和 `llama_context*` 的 RAII 生命周期 |
| 向量引擎 | `embeddingengine.h` / `embeddingengine.cc` | 调用 embedding 模型生成文本向量 |
| 生成引擎 | `generationengine.h` / `generationengine.cc` | 组装 RAG Prompt，调用生成模型进行流式推理 |
| 应用主体 | `mragapp.h` / `mragapp.cc` | 协调建库、追加和问答流程，管理 llama 后端初始化与释放 |
| 程序入口 | `main.cc` | 解析 `--ingest`、`--append` 和 `--chat` 命令，加载配置并启动应用 |

## 4. 核心数据结构

`Chunk` 表示一个可检索文本片段，包含：

- `id`：全局唯一编号。
- `text`：文本块正文。
- `metadata`：章节标题或来源信息。
- `embedding`：对应的浮点向量。

`Chunk::serialize()` 和 `Chunk::deserialize()` 负责二进制读写。为了提高跨平台稳定性，长度和编号字段写入时统一使用 `uint64_t`。

## 5. 配置管理

`AppConfig` 保存所有运行参数，包括：

- embedding 模型路径和 generation 模型路径。
- GPU 层数 `n_gpu_layers`。
- 文档切分参数 `chunk_size`（默认 1500）、`overlap_size`（默认 150）。
- 检索参数 `top_k`（默认 5）。
- `show_recalled_chunks`：控制问答时是否打印召回文本块的来源与内容。
- 生成模型上下文（默认 8192）、batch、最大输出 token（默认 1024）和采样参数（temperature 默认 0.3）。
- embedding 模型上下文和 batch 参数（默认 1024）。

`loadConfig()` 使用 `nlohmann/json` 解析 `config.json`。配置文件不存在时不报错，而是使用 `AppConfig` 的默认值；字段缺失时只覆盖已出现字段，其他字段继续保留默认值。

## 6. 文档处理

`DocumentProcessor::processNovel()` 完成 TXT 小说处理：

1. 逐行读取文件，并去除 Windows 换行符 `\r`。
2. 使用正则识别章节标题，将当前章节名写入 `Chunk::metadata`，并保留标题行进入正文缓冲区，使章节名本身也可被检索。
3. 按 `chunk_size` 字节阈值切分文本。
4. 切割时调整到 UTF-8 合法边界，避免截断中文多字节字符。
5. 优先在中文句末标点或换行处切分，使文本块更自然。
6. 相邻文本块之间保留 `overlap_size` 字节的重叠内容。
7. 文件末尾未达到阈值的残余文本也会保存为最后一个 `Chunk`。

该模块保证输出文本块尽量保持 UTF-8 合法性，并保留章节元数据，便于生成回答时引用上下文来源。

单个文件内部的初始块编号由 `DocumentProcessor` 生成；多文档建库时，`MragApp` 会重新分配跨文件全局递增 ID，并把来源文件名合并进 `metadata`。

## 7. 向量数据库

`VectorDatabase` 负责保存所有带 embedding 的 `Chunk`。检索时先计算问题向量和文本块向量的余弦相似度，再使用最小堆维护 top-k 结果，整体复杂度为 `O(n log k)`。

数据库持久化采用自定义二进制格式，文件头包含：

- 魔数 `0x4D524147`。
- 数据库版本号。
- 文本块数量。

加载数据库时会校验魔数和版本号。如果格式不匹配，程序会拒绝加载并输出错误信息。

`VectorDatabase::maxId()` 遍历内存中的所有 Chunk，返回最大 ID 值。该方法用于增量追加时确定新 Chunk 的起始编号，空数据库返回 0。

## 8. llama.cpp 封装

`LlamaModelBase` 是 embedding 模型和 generation 模型的共同基类。它负责：

- 根据 `ModelMode` 选择模型路径。
- 设置模型加载参数和上下文参数。
- 创建并持有 `llama_model*` 与 `llama_context*`。
- 在析构函数中按正确顺序释放 `ctx_` 和 `model_`。
- 禁用拷贝和移动，避免重复释放底层 C API 资源。
- 提供 `tokenize()` 工具函数，通过两次 `llama_tokenize` 获取精确 token 数量。

Embedding 模式会设置 `ctx_params.embeddings = true` 和平均池化类型；Generation 模式关闭 embeddings，并使用独立的生成上下文参数。

## 9. 向量生成

`EmbeddingEngine::generateEmbedding()` 的流程如下：

1. 使用 `sanitizeUtf8()` 清理输入中的非法 UTF-8 字节。
2. 调用 `tokenize()` 将文本转为 token。
3. 根据 embedding 上下文和 batch 上限截断过长输入。
4. 调用 `llama_memory_clear()` 清空 KV cache。
5. 构造 `llama_batch`，最后一个 token 开启 logits。
6. 调用 `llama_decode()` 执行前向计算。
7. 优先使用 `llama_get_embeddings_seq()` 读取序列级 embedding。
8. 如果序列级 embedding 不可用，则回退到 `llama_get_embeddings()`。

函数最终返回长度为 `llama_model_n_embd(model_)` 的 `std::vector<float>`。

## 10. 文本生成

`GenerationEngine::generateStream()` 实现完整的 RAG 生成流程：

1. 清空生成模型上下文中的 KV cache。
2. 将检索到的 `Chunk` 格式化为带出处的参考上下文。
3. 按 ChatML 风格组装 system、user 和 assistant prompt。
4. 对 prompt 进行 tokenize，并检查是否超过上下文窗口。
5. 使用 batch 完成 prompt prefill。
6. 构造采样链，顺序为 `top_k -> top_p -> temperature -> dist`。
7. 在自回归循环中采样下一个 token。
8. 遇到 EOG token 或达到 `max_output_tokens` 后停止。
9. 将 token piece 流式输出到终端。

Prompt 采用结构化系统指令，要求模型：(1) 只能依据参考上下文作答，(2) 先结论后引用原文证据，(3) 标注片段编号以便溯源，(4) 信息不足时明确告知无法回答，(5) 答案尽量完整详尽。该模板位于 `buildPrompt()` 函数中，可按需调整。

## 11. 应用流程

`MragApp` 负责连接各模块，提供三种操作模式：新建建库 (`buildKnowledgeBase`)、增量追加 (`appendKnowledgeBase`) 和交互问答 (`chatLoop`)。内部 `BackendGuard` 在 `MragApp` 构造时调用 `llama_backend_init()`，在析构时调用 `llama_backend_free()`，确保 llama 后端生命周期覆盖所有模型对象。

`buildKnowledgeBase(txt_paths, db)` 流程：

```text
逐个读取 TXT -> 切分 Chunk -> 标注来源文件 -> 分配全局 ID（从 0 开始） -> 逐块生成 embedding -> 插入 VectorDatabase -> 保存 .mrag
```

该流程支持一次传入多个 TXT 文件。所有文档共用同一个 `EmbeddingEngine` 和 `VectorDatabase`，最终写入同一个 `.mrag` 数据库。

`appendKnowledgeBase(txt_paths, db)` 流程：

```text
加载已有 .mrag -> 读取 maxId -> 逐个读取新 TXT -> 切分 Chunk -> 标注来源文件 -> 从 max_id + 1 开始分配 ID -> 逐块生成 embedding -> 追加到 VectorDatabase -> 合并保存 .mrag
```

增量追加时，程序先校验数据库文件是否存在且可读取，再通过 `VectorDatabase::maxId()` 获取最大 ID 作为新 Chunk 的起始编号。追加完成后输出新增数量和数据库总量。

`chatLoop(db)` 流程：

```text
加载 .mrag -> 循环读取问题 -> 生成问题向量 -> 检索 top-k Chunk
   -> 若 show_recalled_chunks=true，打印每个块的来源、ID 和内容预览
   -> 生成并输出回答
```

召回展示 (`printRecalledChunks`) 对每条检索结果依次输出排名、metadata（含来源文件与章节）、Chunk ID 和文本内容前 300 字节。该输出以 `━━━━━ 召回文本块 ━━━━━` 分隔，便于在终端中快速定位和验证检索质量。

为了降低资源占用，系统按需加载模型：建库和追加模式只加载 embedding 模型，问答模式同时加载 embedding 模型和 generation 模型。

## 12. 命令行接口

程序支持三种模式：

```bash
./build/mrag --ingest  <小说1.txt> [小说2.txt ...] <数据库路径>
./build/mrag --append <小说1.txt> [小说2.txt ...] <已有数据库路径>
./build/mrag --chat   <数据库路径>
```

- `--ingest`：新建数据库，处理一个或多个 TXT 文件后写入 `.mrag` 文件。若目标文件已存在则覆盖。
- `--append`：增量追加，先加载已有数据库，再处理新 TXT 文件并将新 Chunk 追加到数据库，最后合并写回。
- `--chat`：加载数据库进入交互式问答。

`main.cc` 在启动时调用 `llama_log_set()` 注册空日志回调，屏蔽 llama.cpp 底层日志，只保留本项目输出的进度和错误信息。

## 13. 错误处理与资源安全

- 模型文件不存在或加载失败时抛出包含路径的 `std::runtime_error`。
- JSON 解析失败时抛出带配置文件路径的异常。
- 数据库魔数或版本号不匹配时拒绝加载。
- prompt 超出上下文窗口时跳过本次生成并输出提示。
- `llama_decode()` 失败时安全停止当前推理流程。
- `llama_batch` 使用局部 RAII guard 自动释放。
- `llama_sampler` 使用局部 RAII guard 自动释放。
- 项目中没有手动混用裸 `new/delete` 管理 llama C API 资源。
- `config.json` 会验证 `overlap_size < chunk_size`，避免 overlap 过大造成切块逻辑异常。

## 14. 已知局限性

- 文本切分仍以字节长度为主要依据，没有实现复杂语义段落分析。
- 检索只使用余弦相似度，没有接入重排序模型。
- 生成质量受本地模型能力、切块策略、检索参数和 prompt 长度影响。
- 暂未实现运行时热切换模型或配置。
- `.mrag` 二进制格式按本机字节序写入，主要面向常见 x86-64 Linux 环境使用。
- Prompt 模板采用 ChatML 风格，默认更适配 Qwen 系列指令模型。
