"""
Paper Writing Agent - Unified CLI entry point.

Usage:
    python main.py build --pdf data/pdf/paper.pdf
    python main.py qa --query "What is CLIP?"
    python main.py qa --interactive
    python main.py write --topic "Deep Learning in NLP"
"""

import sys
import argparse
from scripts.build_index import main as build_index
from scripts.run_qa import main as run_qa
from scripts.run_writing import run_basic_workflow, run_interactive_workflow


def main():
    parser = argparse.ArgumentParser(
        description="Paper Writing Agent - Pluggable LLM Agent Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build FAISS vector index from PDF")
    build_parser.add_argument("--pdf", type=str, help="Path to PDF file")

    qa_parser = subparsers.add_parser("qa", help="RAG Q&A with hybrid retrieval (BM25 + Dense + Reranker)")
    qa_parser.add_argument("--query", type=str, help="Question to ask")
    qa_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    write_parser = subparsers.add_parser("write", help="Multi-agent writing workflow")
    write_parser.add_argument("--topic", type=str, default="Artificial Intelligence", help="Writing topic")
    write_parser.add_argument("--max-iterations", type=int, default=3, help="Max review iterations")
    write_parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode with human review")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "build":
            build_index(args.pdf)
        elif args.command == "qa":
            run_qa(args.query, args.interactive)
        elif args.command == "write":
            if args.interactive:
                run_interactive_workflow(args.topic, args.max_iterations)
            else:
                run_basic_workflow(args.topic, args.max_iterations)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
