"""Run JVM RAG retrieval evaluation for multiple document versions.

用途：

1. 固定同一个 RAG 检索评测数据集；
2. 依次评测多个文档索引版本；
3. 横向打印 Hit@K、Recall@K、Precision@K、MRR@K、无答案误放行率和耗时；
4. 帮助判断候选切块策略是否可以切换为 active。

注意：非 dry-run 模式会真实调用 Embedding / Reranker，并向数据库写入 eval run 和 case result。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.database import SessionLocal
from day04_app.models import KnowledgeDocumentVersion, KnowledgeRetrievalEvalDataset
from day04_app.services.knowledge_retrieval_eval_service import run_retrieval_evaluation


DEFAULT_DATASET_ID = "8e13fa807fe44ecaa735e9c581d1e97f"
DEFAULT_VERSION_IDS = [
    # v2：当前线上 active，普通细粒度固定切块。
    "fb0259350b9d49c6af0a967637f027f9",
    # v4：父子切块，child_max_characters=350。
    "c8e00ce7055d4c90a1e226480c4d37ba",
    # v5：优化版父子切块，child_max_characters=260 + 短标题边界识别。
    "ee9f3d4904f941d3a38391146bc4887d",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare JVM retrieval eval results.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="检索评测数据集 ID。")
    parser.add_argument(
        "--version-id",
        action="append",
        dest="version_ids",
        help="要评测的文档版本 ID。可重复传入；不传则使用课程默认 v2/v4/v5。",
    )
    parser.add_argument("--top-k", type=int, default=5, help="最终参与指标计算的 Top-K。")
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        default=True,
        help="启用 Reranker 精排。默认启用。",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_false",
        dest="use_reranker",
        help="关闭 Reranker，只看 Milvus 粗排效果。",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=50,
        help="启用 Reranker 前，从 Milvus 粗排保留的候选数量。",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="可选拒答阈值；传入后会计算 no_answer 误放行率。",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="可选：只评测前 N 条 active 样本，用于学习阶段低成本试跑；正式准入不要传。",
    )
    parser.add_argument("--created-by", default="liu_jiahao", help="评测发起人。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要评测的版本，不调用模型，不写数据库。",
    )
    return parser.parse_args()


def _fmt_number(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    version_ids = args.version_ids or DEFAULT_VERSION_IDS
    db = SessionLocal()
    try:
        dataset = db.scalar(
            select(KnowledgeRetrievalEvalDataset).where(
                KnowledgeRetrievalEvalDataset.dataset_id == args.dataset_id
            )
        )
        if dataset is None:
            raise SystemExit(f"数据集不存在：{args.dataset_id}")

        versions = list(
            db.scalars(
                select(KnowledgeDocumentVersion)
                .where(KnowledgeDocumentVersion.version_id.in_(version_ids))
                .order_by(KnowledgeDocumentVersion.version_number.asc())
            )
        )
        version_by_id = {version.version_id: version for version in versions}
        missing_version_ids = [version_id for version_id in version_ids if version_id not in version_by_id]
        if missing_version_ids:
            raise SystemExit(f"文档版本不存在：{missing_version_ids}")

        print(
            f"数据集：{dataset.dataset_name}/{dataset.dataset_version} "
            f"dataset_id={dataset.dataset_id} sample_count={dataset.sample_count}"
        )
        print(
            f"评测参数：top_k={args.top_k}, use_reranker={args.use_reranker}, "
            f"rerank_top_n={args.rerank_top_n}, score_threshold={args.score_threshold}, "
            f"sample_limit={args.sample_limit}"
        )
        print()

        if args.dry_run:
            for version_id in version_ids:
                version = version_by_id[version_id]
                print(
                    f"DRY v{version.version_number:<2} status={version.status:<8} "
                    f"chunks={version.chunk_count:<4} vectors={version.vector_count:<4} "
                    f"version_id={version.version_id}"
                )
            return

        results = []
        for version_id in version_ids:
            version = version_by_id[version_id]
            print(f"RUN  v{version.version_number} {version.version_id}")
            run = run_retrieval_evaluation(
                db,
                dataset_id=dataset.dataset_id,
                document_version_id=version.version_id,
                retrieval_top_k=args.top_k,
                score_threshold=args.score_threshold,
                created_by=args.created_by,
                use_reranker=args.use_reranker,
                rerank_top_n=args.rerank_top_n,
                sample_limit=args.sample_limit,
            )
            results.append((version, run))

        print()
        print(
            "version | status  | chunks | run_status      | hit@k  | recall@k | "
            "precision@k | mrr@k  | no_answer_fp | elapsed_ms | run_id"
        )
        print("-" * 132)
        for version, run in results:
            no_answer_fp = (
                _fmt_number(run.no_answer_false_positive_rate)
                if run.no_answer_false_positive_rate is not None
                else "-"
            )
            print(
                f"v{version.version_number:<7}| "
                f"{version.status:<7} | "
                f"{version.chunk_count:<6} | "
                f"{run.status:<15} | "
                f"{_fmt_number(run.hit_at_k):<6} | "
                f"{_fmt_number(run.recall_at_k):<8} | "
                f"{_fmt_number(run.precision_at_k):<11} | "
                f"{_fmt_number(run.mrr_at_k):<6} | "
                f"{no_answer_fp:<12} | "
                f"{_fmt_number(run.elapsed_ms):<10} | "
                f"{run.run_id}"
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
