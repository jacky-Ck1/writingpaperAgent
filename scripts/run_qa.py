"""
RAG 问答系统 - 混合检索版 (BM25 + Dense + Reranker)

检索策略: 两路并行 → RRF 融合 → Cross-Encoder 重排序
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from src.config import Config
from src.rag import load_index, load_pdf
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.embedding import create_embeddings
import glob


def main(query: str = None, interactive: bool = False):
    """初始化所有组件，然后进入问答循环"""
    print("正在初始化 RAG 问答系统...")

    # ---- 组件 1: LLM ----
    llm = ChatOpenAI(
        model=Config.DEEPSEEK_MODEL,
        api_key=Config.DEEPSEEK_API_KEY,
        base_url=Config.DEEPSEEK_BASE_URL
    )

    # ---- 组件 2: 向量索引 ----
    print(f"正在加载向量索引: {Config.INDEX_DIR}")
    vectorstore = load_index()
    print("✓ 索引加载完成")

    # ---- 组件 3: 原始文档（BM25 需要原始文本，不能只有向量）----
    pdfs = glob.glob(os.path.join(Config.PDF_DIR, "*.pdf"))
    documents = []
    if pdfs:
        documents = load_pdf(pdfs[0])
        print(f"✓ 加载 {len(documents)} 页文档 (供 BM25 使用)")
    else:
        print("⚠ 未找到 PDF 文件，BM25 将不可用，仅使用向量检索")

    # ---- 组件 4: Embedding 模型（HybridRetriever 需要显式传入）----
    embeddings = create_embeddings()

    # ---- 组件 5: 混合检索器（替代了 SimpleAgent + ToolRegistry）----
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        embedding=embeddings,
        documents=documents,
        enable_reranker=Config.ENABLE_HYBRID_RETRIEVAL,
    )
    print("✓ 混合检索器就绪 (BM25 + Dense + Reranker)\n")

    # ---- 问答循环 ----
    if interactive:
        print("交互模式（输入 quit / exit 退出）\n")
        while True:
            user_query = input("请输入问题: ").strip()
            if user_query.lower() in ('quit', 'exit', 'q'):
                print("再见！")
                break
            if not user_query:
                continue
            print()
            _answer(llm, retriever, user_query)
    else:
        if query is None:
            query = "What is CLIP?"
        print(f"问题: {query}\n")
        _answer(llm, retriever, query)


def _answer(llm, retriever, query: str):
    """
    核心管道: 检索 → 构建上下文 → LLM 生成答案

    检索管道: retriever.retrieve() → llm.invoke()
    """

    # ---- 第 1 步: 混合检索 ----
    results = retriever.retrieve(
        query,
        top_k=Config.RETRIEVAL_K,
        use_reranker=Config.ENABLE_HYBRID_RETRIEVAL,
    )
    docs = [doc for doc, _score in results]

    print(f"检索到 {len(docs)} 个相关段落:")
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "?")
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i+1}] {section} | {preview}...")
    print()

    # ---- 第 2 步: 构建上下文 ----
    context_parts = []
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "")
        header = f"[段落 {i+1}]" + (f" (章节: {section})" if section else "")
        context_parts.append(f"{header}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # ---- 第 3 步: LLM 生成答案 ----
    prompt = f"""基于以下论文内容回答问题。

{context}

问题: {query}

要求: 基于上述内容回答，如果内容中没有相关信息请明确说明。"""

    print("正在生成回答...")
    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    print(f"回答:\n{answer}")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 问答系统（混合检索版）")
    parser.add_argument("--query", type=str, default=None, help="查询问题")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    args = parser.parse_args()
    main(args.query, args.interactive)

