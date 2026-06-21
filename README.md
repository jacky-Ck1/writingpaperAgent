# ?? Paper Writing Agent — Pluggable LLM Agent Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://github.com/langchain-ai/langchain)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal.svg)](https://fastapi.tiangolo.com/)

> 可插拔架构的 AI 学术论文写作 Agent 框架 — 支持多 LLM / 多 Embedding / 多向量库一键切换，混合检索 + 智能分块 + API 服务化。

简体中文

## ? 核心能力

| 能力 | 说明 |
|---|---|
| ?? **可插拔 Provider 层** | LLM（DeepSeek / OpenAI / 本地模型）、Embedding（BGE / OpenAI / 本地）、Vector Store（FAISS / Chroma）均可通过环境变量一键切换 |
| ?? **混合检索** | BM25 稀疏检索 + 向量密集检索 + Reciprocal Rank Fusion + Cross-encoder 重排序 |
| ?? **论文感知智能分块** | 自动检测论文章节（Abstract / Method / Experiment 等），按章节边界切分，不同章节自适应 chunk size |
| ?? **多 Agent 协作写作** | Planner → Writer(s) → Reviewer 迭代循环（LangGraph），支持并行 Writer 和人工介入 |
| ?? **FastAPI 服务化** | 生产就绪的 REST API，自动生成 Swagger 文档，CORS 支持 |
| ?? **完整评估体系** | RAG / Agent / Workflow 三维度评估，支持批量和自动评估 |

## ?? 架构设计

```
src/
├── providers/          # 可插拔抽象层 ★ NEW
│   ├── llm.py          #   LLM Provider (DeepSeek / OpenAI / Local)
│   ├── embedding.py    #   Embedding Provider (BGE / OpenAI / Local)
│   └── vectorstore.py  #   Vector Store (FAISS / Chroma)
├── rag/                # RAG 模块
│   ├── smart_splitter.py   # 论文感知智能分块 ★ NEW
│   ├── hybrid_retriever.py # 混合检索 BM25+Dense+Reranker ★ NEW
│   ├── loader.py       # 文档加载
│   ├── embedding.py    # 嵌入创建
│   └── vectorstore.py  # 向量存储
├── agents/             # Agent 实现
├── workflows/          # LangGraph 工作流
├── tools/              # 工具系统
├── core/               # 核心模块（路由、统一Agent、事件流）
├── evaluation/         # 评估体系
├── api/                # FastAPI 服务层 ★ NEW
│   ├── server.py       #   应用入口
│   └── routes.py       #   API 路由
└── config.py           # 统一配置管理
```

### Provider 切换示例

```bash
# 切换到 OpenAI
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o
export EMBEDDING_PROVIDER=openai

# 切换到本地 Ollama
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=http://localhost:11434/v1
export LOCAL_LLM_MODEL=qwen2.5:7b

# 切换到 Chroma 向量库
export VECTORSTORE_PROVIDER=chroma
```

### 混合检索流程

```
Query → [BM25 稀疏搜索] ──┐
                          ├── RRF 融合 ──→ Cross-Encoder 重排序 → Top-K 结果
       → [向量密集搜索] ──┘
```

## ?? 快速开始

### 一键安装

**Windows:**
```bash
install.bat
```

**Mac/Linux:**
```bash
chmod +x install.sh && ./install.sh
```

### 启动 API 服务

```bash
# 安装依赖（含 FastAPI）
pip install -r requirements.txt

# 启动服务
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000/docs 查看交互式 API 文档。

### API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查 + 当前 provider 信息 |
| POST | `/api/v1/qa` | RAG 问答（支持混合检索） |
| POST | `/api/v1/write` | 多 Agent 写作工作流 |
| POST | `/api/v1/rag-write` | RAG + Agent 综述写作 |

### 命令行使用

```bash
# 构建索引（自动使用配置的 provider）
python scripts/build_index.py --pdf data/pdf/your_paper.pdf

# 问答
python scripts/run_qa.py --query "What is the main contribution?"

# 写作
python scripts/run_writing.py --topic "Deep Learning in NLP"

# RAG 写作（检索论文 + 生成综述）
python scripts/run_rag_writing_simple.py
```

## ??? 技术栈

- **LangChain / LangGraph**: Agent 编排与工作流
- **FAISS / Chroma**: 向量数据库
- **HuggingFace BGE / OpenAI Embeddings**: 文本嵌入
- **BM25 + RRF + Cross-Encoder**: 混合检索
- **DeepSeek / OpenAI / Ollama**: 大语言模型
- **FastAPI + Uvicorn**: API 服务
- **Pydantic**: 数据校验

## ?? 待办

- [ ] 添加更多 Embedding 模型（Jina、Cohere）
- [ ] 支持更多文档格式（Word、LaTeX、Markdown）
- [ ] 添加 Web UI 界面
- [ ] 流式 API 响应（SSE）
- [ ] 性能基准测试报告

## ?? 许可证

MIT License
