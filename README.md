# MRAG 本地小说问答系统

本项目是一个基于 `llama.cpp` C API 的本地 RAG 问答系统。程序支持将一个或多个 UTF-8 小说 TXT 文件切分为文本块并生成向量数据库，同时支持对已有数据库做增量追加，再根据用户问题检索相关片段，最后调用本地大语言模型进行流式回答。

## 目录结构

```text
my_lab/
├── CMakeLists.txt
├── README.md
├── DESIGN.md
├── config.json
└── src/
    ├── main.cc
    ├── config.h
    ├── config.cc
    ├── chunk.h
    ├── chunk.cc
    ├── document.h
    ├── document.cc
    ├── vectordatabase.h
    ├── vectordatabase.cc
    ├── llamamodel.h
    ├── llamamodel.cc
    ├── embeddingengine.h
    ├── embeddingengine.cc
    ├── generationengine.h
    ├── generationengine.cc
    ├── mragapp.h
    └── mragapp.cc
```

## 依赖安装

推荐在 Ubuntu 22.04/24.04 x86-64 环境下编译运行：

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

项目使用 CMake `FetchContent` 自动拉取以下第三方依赖：

- `llama.cpp`
- `nlohmann/json`

第一次执行 CMake 配置时需要联网下载依赖。`CMakeLists.txt` 将 `llama.cpp` 固定到 `b9703` 标签，避免直接跟随 `master` 分支造成 API 漂移。

## 模型准备

程序需要两个 GGUF 模型：

- Embedding 模型：用于将小说片段和用户问题转换为向量。
- Generation 模型：用于根据检索结果生成自然语言回答。

推荐模型：

- `bge-small-zh-v1.5-f16.gguf`
- `qwen2.5-1.5b-instruct-q4_k_m.gguf`

参考下载地址：

- BGE: https://huggingface.co/BAAI/bge-small-zh-v1.5
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- llama.cpp: https://github.com/ggml-org/llama.cpp

建议将模型放在项目根目录的 `models/` 目录中：

```text
models/bge-small-zh-v1.5-f16.gguf
models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

然后在 `config.json` 中配置模型路径：

```json
{
  "models": {
    "embedding": "./models/bge-small-zh-v1.5-f16.gguf",
    "generation": "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "n_gpu_layers": 99
  }
}
```

如果使用其他 GGUF 模型，只需要把 `config.json` 中的路径改为实际文件路径。

## 编译方法

CPU 版本：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

编译完成后可执行文件位于：

```text
build/mrag
```

如果本机已经安装 NVIDIA 驱动和 CUDA Toolkit，也可以编译 CUDA 版本：

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda -j
```

不同显卡环境可能需要额外指定 `CMAKE_CUDA_ARCHITECTURES`。

## 运行方法

**离线建库**，支持单个或多个 TXT 文件：

```bash
./build/mrag --ingest <小说1.txt> [小说2.txt ...] <输出数据库路径>
```

单文档示例：

```bash
./build/mrag --ingest 三国演义.txt sanguo.mrag
```

多文档示例：

```bash
./build/mrag --ingest 三国演义.txt 水浒传.txt 西游记.txt classics.mrag
```

多文档建库时，最后一个参数固定为输出数据库路径，前面的 TXT 文件都会被写入同一个 `.mrag` 数据库。程序会为所有文本块重新分配全局递增 ID，并在 metadata 中加入来源文件名，便于问答时区分出处。

**增量追加**，向已有 `.mrag` 数据库追加新文档：

```bash
./build/mrag --append <小说1.txt> [小说2.txt ...] <已有数据库路径>
```

单文档追加示例：

```bash
./build/mrag --append 水浒传.txt sanguo.mrag
```

多文档追加示例：

```bash
./build/mrag --append 水浒传.txt 西游记.txt sanguo.mrag
```

增量追加模式下，程序先加载已有数据库，读取其中的最大文本块 ID，新文档的文本块从 `max_id + 1` 开始编号，避免 ID 冲突。追加成功后，数据库中原有的文本块保持不变，新增部分和原有部分合并写入同一个 `.mrag` 文件。

**在线问答**：

```bash
./build/mrag --chat <数据库路径>
```

示例：

```bash
./build/mrag --chat sanguo.mrag
```

进入问答模式后，输入问题并回车即可生成回答；输入空行退出程序。

问答时，程序默认会在回答前打印**召回的文本块详情**，包括每条结果在数据库中的排名、来源文件与章节、文本块 ID、以及内容预览（前 300 字节）。示例输出：

```text
> 关羽是怎么死的？

━━━━━ 召回文本块 ━━━━━

[块 1] 来源: 三国演义.txt / 第七十七回 玉泉山关公显圣 洛阳城曹操感神  |  ID: 542
内容: 却说关公在麦城，计点马步军兵，止剩三百余人；粮草又尽。是夜，城外吴兵
招唤各军姓名，越城而去者甚多。……（后文省略）

[块 2] 来源: 三国演义.txt / 第七十七回 玉泉山关公显圣 洛阳城曹操感神  |  ID: 543
内容: 公大怒，急披挂上马，引随行数骑出北门。一声喊起，伏兵尽出，长钩套索，
一齐并举，先把关公坐下马绊倒。关公翻身落马，被潘璋部将马忠所获。…

━━━━━━━━━━━━━━━━━━━━

回答：根据参考上下文，关羽是在麦城突围时……
```

如果希望关闭此输出只显示回答，可在 `config.json` 中将 `retrieval.show_recalled_chunks` 设为 `false`。

## 配置说明

`config.json` 支持以下字段（括号内为默认值）：

- `models.embedding`：embedding 模型路径。
- `models.generation`：generation 模型路径。
- `models.n_gpu_layers`：加载到 GPU 的模型层数（99，即全部）。
- `document.chunk_size`：文本块大小，默认 1500 字节。
- `document.overlap_size`：相邻文本块重叠大小，默认 150 字节，必须小于 `chunk_size`。
- `retrieval.top_k`：检索返回的文本块数量，默认 5。
- `retrieval.show_recalled_chunks`：是否打印召回文本块的来源和内容，默认 `true`。
- `generation.n_ctx`：生成模型上下文大小，默认 8192。
- `generation.n_batch`：生成模型 batch 大小，默认 8192。
- `generation.n_ubatch`：生成模型 micro-batch 大小，默认 512。
- `generation.max_output_tokens`：最大输出 token 数，默认 1024。
- `generation.temperature`：采样温度，默认 0.3（越低越确定性）。
- `generation.top_k`：采样 top-k，默认 40。
- `generation.top_p`：采样 top-p，默认 0.9。
- `embedding.n_ctx`：embedding 模型上下文大小，默认 1024。
- `embedding.n_batch`：embedding 模型 batch 大小，默认 1024。
- `embedding.n_ubatch`：embedding 模型 micro-batch 大小，默认 1024。

如果 `config.json` 不存在，程序会使用 `AppConfig` 中的默认值；如果部分字段缺失，缺失字段也会保留默认值。

## 常见问题

- 如果提示模型加载失败，请检查 `config.json` 中的模型路径是否存在，并确认模型是 GGUF 格式。
- 如果 CMake 下载依赖失败，请检查网络连接，或提前准备好可访问 GitHub 的环境。
- 如果回答质量不稳定，可以适当增大 `retrieval.top_k` 或调整 `document.chunk_size`。
- 如果提示 prompt 超出上下文窗口，可以减小 `top_k`、减小文本块大小，或使用更大上下文的生成模型。

## 输出质量调优

以下参数对回答质量影响最大，按优先级排列：

| 参数 | 默认值 | 作用 | 调优建议 |
|------|--------|------|---------|
| `document.chunk_size` | 1500 | 每个文本块的字节数 | 太小则上下文不足，太大则检索精度下降。中文小说建议 1000~2000 字节 |
| `retrieval.top_k` | 5 | 检索返回的文本块数量 | 增大可提供更多上下文，但也会增加 prompt 长度。建议 3~8 |
| `generation.temperature` | 0.3 | 采样温度，越低越确定性 | 事实问答建议 0.1~0.5，创意写作可设 0.7~1.0 |
| `generation.max_output_tokens` | 1024 | 回答最大 token 数 | 增大可让回答更详细，但也会增加生成时间 |
| `generation.n_ctx` | 8192 | 模型上下文窗口大小 | 需要 ≥ prompt token 数 + max_output_tokens |
| `embedding.n_ctx` | 1024 | embedding 模型的上下文窗口 | 需要 ≥ 单个 chunk 的 token 数，避免截断 |

**如果更换了更大的生成模型**（如 Qwen2.5-7B），建议同步增大 `n_ctx` 和 `max_output_tokens`，并适当降低 `temperature`。Prompt 模板在 `generationengine.cc` 的 `buildPrompt()` 函数中，可根据需要调整系统提示词。

## 已知局限性

- 文本切分主要依据字节长度和标点位置，未实现复杂语义分段。
- 检索阶段只使用向量余弦相似度，没有使用重排序模型。
- 问答效果依赖本地模型能力、embedding 质量、切块大小和检索参数。
- 模型文件体积较大，按课程提交规范不放入源码压缩包。
- `.mrag` 二进制数据库主要面向常见 x86-64 Linux 环境，未额外做跨大小端转换。
- Prompt 模板采用 ChatML 风格，默认更适配 Qwen 系列指令模型。
