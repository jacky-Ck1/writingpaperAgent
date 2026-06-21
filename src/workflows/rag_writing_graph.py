"""
RAG + Agent 综述写作工作流
完整流程：query → RAG检索 → summary总结 → reviewer审核 → refine优化

核心特点：
1. 基于检索到的论文内容写综述
2. 自动审核和迭代优化
3. 支持多轮修改直到满意
"""

import glob
import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add
import json
import re
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

from ..config import Config
from ..agents.rag_agent import RAGAgent


# =============================
# 数据模型定义
# =============================

class ReviewResult(BaseModel):
    """审核结果模型"""
    approved: bool  # 是否通过审核
    feedback: str  # 审核反馈意见
    score: int  # 评分 (1-10)


class RAGWritingState(TypedDict):
    """RAG 写作工作流的状态"""
    query: str  # 用户查询主题
    retrieved_docs: Annotated[list[str], add]  # 检索到的文档内容
    summary: str  # 初步总结
    review_feedback: str  # 审核反馈
    refined_summary: str  # 优化后的综述
    approved: bool  # 是否通过审核
    iteration: int  # 当前迭代次数
    max_iterations: int  # 最大迭代次数


def _auto_build_hybrid_retriever(vectorstore, retrieval_k):
    """Try to auto-build a HybridRetriever from available PDFs."""
    try:
        from ..rag.loader import load_pdf
        from ..rag.embedding import create_embeddings
        from ..rag.hybrid_retriever import HybridRetriever

        pdf_dir = Config.PDF_DIR
        pdfs = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        if not pdfs:
            print("[HybridRetriever] 未找到 PDF 文件，回退到纯向量检索")
            return None

        print(f"[HybridRetriever] 加载 PDF: {pdfs[0]}")
        documents = load_pdf(pdfs[0])
        embeddings = create_embeddings()

        retriever = HybridRetriever(
            vectorstore=vectorstore,
            embedding=embeddings,
            documents=documents,
            enable_reranker=Config.ENABLE_HYBRID_RETRIEVAL,
        )
        print(f"[HybridRetriever] ✓ 已构建 (BM25 + Dense + Reranker)")
        return retriever
    except Exception as e:
        print(f"[HybridRetriever] 构建失败 ({e})，回退到纯向量检索")
        return None


def create_rag_writing_graph(llm=None, rag_agent=None, max_iterations=None, hybrid_retriever=None):
    """
    创建 RAG 综述写作工作流图

    Args:
        llm: 大语言模型实例（可选）
        rag_agent: RAG Agent 实例（可选）
        max_iterations: 最大迭代次数（可选）
        hybrid_retriever: HybridRetriever 实例（可选，传入后启用混合检索）

    Returns:
        编译后的工作流图
    """
    # 初始化 LLM
    if llm is None:
        llm = ChatOpenAI(
            model=Config.DEEPSEEK_MODEL,
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=Config.DEEPSEEK_BASE_URL
        )

    # 初始化 RAG Agent（优先使用混合检索）
    if rag_agent is None:
        # 先加载向量存储
        from ..rag import load_index
        vectorstore = load_index()

        # 如果未显式传入 hybrid_retriever，尝试自动构建
        if hybrid_retriever is None:
            hybrid_retriever = _auto_build_hybrid_retriever(
                vectorstore, Config.RETRIEVAL_K
            )

        rag_agent = RAGAgent(
            llm=llm, vectorstore=vectorstore,
            hybrid_retriever=hybrid_retriever,
        )

    max_iterations = max_iterations or Config.MAX_ITERATIONS

    # =============================
    # 节点函数定义
    # =============================

    def retriever(state: RAGWritingState):
        """
        检索节点：使用 RAG 检索相关论文内容

        这是整个流程的第一步，从向量数据库中找到与查询相关的文档
        """
        print(f"\n{'='*50}")
        print(f"[Retriever] 第 {state['iteration']} 轮迭代")
        print(f"[Retriever] 查询主题: {state['query']}")
        print(f"{'='*50}\n")

        # 使用 RAG Agent 检索文档
        docs = rag_agent.retrieve(state['query'])

        # 提取文档内容
        doc_contents = [doc.page_content for doc in docs]

        print(f"[Retriever] ✓ 成功检索到 {len(doc_contents)} 个相关文档片段\n")

        return {
            "retrieved_docs": doc_contents
        }

    def summarizer(state: RAGWritingState):
        """
        总结节点：基于检索到的文档生成综述

        这一步将检索到的多个文档片段整合成一篇连贯的综述
        """
        print(f"[Summarizer] 正在生成综述...")

        # 构建文档上下文
        context = "\n\n".join([
            f"【文档 {i+1}】\n{doc}"
            for i, doc in enumerate(state['retrieved_docs'])
        ])

        # 如果有审核反馈，加入到 prompt 中
        feedback_section = ""
        if state.get('review_feedback'):
            feedback_section = f"""

审核反馈意见：
{state['review_feedback']}

请根据以上反馈意见进行改进。"""

        prompt = f"""
你是一位专业的学术综述写作专家。请基于以下检索到的文档内容，撰写一篇关于"{state['query']}"的学术综述。

检索到的相关文档：
{context}
{feedback_section}

要求：
1. 综合分析所有文档的核心观点
2. 逻辑清晰，结构合理
3. 突出重点和创新点
4. 使用学术化的语言
5. 字数控制在 500-800 字

请直接输出综述内容，不要添加额外的说明。"""

        response = llm.invoke(prompt)
        summary = response.content.strip()

        print(f"[Summarizer] ✓ 综述生成完成 (长度: {len(summary)} 字符)\n")

        return {
            "summary": summary
        }

    def reviewer(state: RAGWritingState):
        """
        审核节点：评估综述质量，决定是否通过

        通过审核条件：
        - approved=True 且 score >= 8
        - 达到最大迭代次数（自动通过）
        """
        print(f"[Reviewer] 正在审核综述质量...")

        feedback = state.get('review_feedback', '')

        prompt = f"""
你是一位严格的学术审核专家。请审核以下综述的质量。

查询主题：{state['query']}

综述内容：
{state['summary']}

{f'之前审核反馈：{feedback}' if feedback else ''}

请以 JSON 格式返回审核结果：
{{
    "approved": true/false,
    "score": 1-10,
    "feedback": "具体的改进建议"
}}

评估标准：
- 内容准确性 (3分)
- 逻辑连贯性 (3分)
- 学术规范性 (2分)
- 创新性 (2分)
- 总分 >= 8 为通过

请直接返回 JSON，不要添加其他内容。"""

        response = llm.invoke(prompt)
        response_text = response.content.strip()

        # 解析 JSON
        try:
            result_dict = json.loads(response_text)
        except json.JSONDecodeError:
            # 解析失败，尝试提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result_dict = json.loads(json_match.group())
            else:
                # 解析失败，使用默认值
                result_dict = {
                    "approved": False,
                    "score": 5,
                    "feedback": "评审结果解析失败，建议重新生成"
                }

        # 验证并转换为 ReviewResult 对象
        try:
            result = ReviewResult(**result_dict)
        except Exception as e:
            print(f"[Reviewer] 警告：评审结果验证失败 - {e}")
            result = ReviewResult(
                approved=False,
                score=5,
                feedback="评审结果格式错误，建议重新生成"
            )

        print(f"[Reviewer] 评分: {result.score}/10")
        print(f"[Reviewer] 是否通过: {'✓ 是' if result.approved else '✗ 否'}")
        print(f"[Reviewer] 反馈: {result.feedback[:100]}...\n")

        new_iteration = state['iteration'] + 1

        # 判断是否达到最大迭代次数
        if new_iteration >= state['max_iterations']:
            print(f"[Reviewer] ⚠ 已达到最大迭代次数 ({state['max_iterations']})，结束流程\n")
            return Command(
                update={
                    "approved": result.approved,
                    "review_feedback": result.feedback,
                    "iteration": new_iteration,
                    "refined_summary": state.get('summary', '')
                },
                goto=END
            )

        # 如果审核通过（评分 >= 8），结束流程
        if result.approved or result.score >= 8:
            print(f"[Reviewer] ✓ 综述质量优秀，通过审核！\n")
            return Command(
                update={
                    "approved": True,
                    "review_feedback": result.feedback,
                    "iteration": new_iteration,
                    "refined_summary": state.get('summary', '')
                },
                goto=END
            )

        # 需要优化，跳转到 refiner 节点
        print(f"[Reviewer] → 需要优化，进入 refiner 节点\n")
        return Command(
            update={
                "approved": False,
                "review_feedback": result.feedback,
                "iteration": new_iteration
            },
            goto="refiner"
        )

    def refiner(state: RAGWritingState):
        """
        优化节点：根据审核反馈优化综述

        这一步会根据审核意见对综述进行针对性改进
        """
        print(f"[Refiner] 正在根据反馈优化综述...")

        prompt = f"""
你是一位专业的学术写作编辑。请根据审核反馈，优化以下综述。

原始查询主题：{state['query']}

当前综述：
{state['summary']}

审核反馈：
{state['review_feedback']}

要求：
1. 针对反馈意见进行改进
2. 保持综述的整体结构和风格
3. 提升内容质量和学术性
4. 字数控制在 500-800 字

请直接输出优化后的综述，不要添加额外说明。"""

        response = llm.invoke(prompt)
        refined = response.content.strip()

        print(f"[Refiner] ✓ 优化完成 (长度: {len(refined)} 字符)")
        print(f"[Refiner] → 返回 reviewer 节点重新审核\n")

        return Command(
            update={
                "summary": refined,
                "refined_summary": refined
            },
            goto="reviewer"
        )

    # =============================
    # 构建工作流图
    # =============================

    graph = StateGraph(RAGWritingState)

    # 添加节点
    graph.add_node("retriever", retriever)
    graph.add_node("summarizer", summarizer)
    graph.add_node("reviewer", reviewer)
    graph.add_node("refiner", refiner)

    # 设置入口点
    graph.set_entry_point("retriever")

    # 添加边：定义节点之间的流转关系
    graph.add_edge("retriever", "summarizer")
    graph.add_edge("summarizer", "reviewer")
    # reviewer 和 refiner 的流转由 Command 控制，不需要显式添加边

    return graph
