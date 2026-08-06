"""Seed JVM RAG retrieval evaluation samples.

这批样本用于 Day22 的 RAG 检索策略对比：

- 固定同一份人工标注样本集；
- 分别评测不同文档索引版本；
- 用 Hit@K、Recall@K、Precision@K、MRR@K 判断候选切块策略是否值得上线。

脚本是幂等的：如果同一个 dataset 下已经存在相同 question，会自动跳过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import func, select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from day04_app.database import SessionLocal
from day04_app.models import KnowledgeRetrievalEvalDataset, KnowledgeRetrievalEvalSample
from day04_app.services.knowledge_retrieval_eval_service import create_retrieval_eval_sample


DEFAULT_DATASET_ID = "8e13fa807fe44ecaa735e9c581d1e97f"


JVM_RETRIEVAL_SAMPLES: list[dict[str, object]] = [
    {
        "question": "JVM 虚拟机栈中的局部变量表和操作数栈分别做什么？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [6, 7],
        "expected_note": "期望召回 Paragraph:7-8：局部变量表保存局部变量和参数，操作数栈用于执行字节码指令。",
    },
    {
        "question": "线上 Java 进程 CPU 占用过高时，如何用 jstack 定位具体线程？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [22, 24, 25, 26, 27, 28],
        "expected_note": "期望召回 Paragraph:29、31-35：先定位进程和线程，再导出 jstack 并用十六进制 nid 匹配。",
    },
    {
        "question": "StringBuilder 和 StringBuffer 有什么区别，分别适合什么场景？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [63, 64, 65, 66, 67, 68],
        "expected_note": "期望召回 Paragraph:71-76：二者底层相似，区别在线程安全、性能和使用场景。",
    },
    {
        "question": "直接内存为什么比传统 IO 更适合大文件读写？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [69, 70, 71, 72, 73],
        "expected_note": "期望召回 Paragraph:77-81：直接内存属于本地内存，可减少一次堆内复制，适合大文件。",
    },
    {
        "question": "Java 为什么不用引用计数法判断对象是否可回收？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [77, 78],
        "expected_note": "期望召回 Paragraph:85-86：引用计数法无法解决循环引用，因此 Java 使用可达性分析。",
    },
    {
        "question": "可达性分析算法从哪些 GC Root 出发判断对象存活？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [79, 80],
        "expected_note": "期望召回 Paragraph:87-88：从虚拟机栈、方法区静态变量、常量、JNI、锁对象等 GC Root 出发。",
    },
    {
        "question": "Minor GC 的触发条件和完整执行步骤是什么？",
        "sample_type": "boundary",
        "expected_answerable": True,
        "expected_segment_indexes": [97, 98, 99, 100, 101, 102, 103, 104, 105, 106],
        "expected_note": "期望召回 Paragraph:109-118：Eden 不足触发，标记、复制到 To、年龄增加、晋升、清空、交换。",
    },
    {
        "question": "G1 回收器的 Region、RSet 和 Mixed GC 分别解决什么问题？",
        "sample_type": "boundary",
        "expected_answerable": True,
        "expected_segment_indexes": [126, 127, 128, 129, 130, 131, 132, 133],
        "expected_note": "期望召回 Paragraph:145-152：G1 以 Region/RSet/Mixed GC 支撑可控低延迟和分批回收。",
    },
    {
        "question": "双亲委派机制的加载流程是什么，它为什么能防止核心类被篡改？",
        "sample_type": "boundary",
        "expected_answerable": True,
        "expected_segment_indexes": [194, 195],
        "expected_note": "期望召回 Paragraph:234-235：loadClass 先向上委托，再逐级向下，防止自定义核心类替代 JDK 类。",
    },
    {
        "question": "字符串常量拼接和变量拼接在 JVM 中有什么区别？",
        "sample_type": "boundary",
        "expected_answerable": True,
        "expected_segment_indexes": [57, 58],
        "expected_note": "期望召回 Paragraph:64-65：变量拼接运行时用 StringBuilder，常量拼接编译期常量折叠。",
    },
    {
        "question": "类加载的链接阶段包括哪三步，准备阶段会如何处理 static final 常量？",
        "sample_type": "boundary",
        "expected_answerable": True,
        "expected_segment_indexes": [173, 174, 175, 176, 177],
        "expected_note": "期望召回 Paragraph:208-212：链接包含验证、准备、解析，static final 基本类型或字符串常量直接赋最终值。",
    },
    {
        "question": "反射调用超过 15 次后，JVM 会如何优化反射性能？",
        "sample_type": "normal",
        "expected_answerable": True,
        "expected_segment_indexes": [214, 215],
        "expected_note": "期望召回 Paragraph:259-260：超过阈值后动态生成包装类，后续调用接近直接调用。",
    },
    {
        "question": "ZGC 的染色指针和读屏障机制是什么？",
        "sample_type": "no_answer",
        "expected_answerable": False,
        "expected_segment_indexes": [],
        "expected_note": "当前 JVM 文档只覆盖串行、Parallel、CMS、G1，没有 ZGC 染色指针和读屏障内容，应拒答。",
    },
    {
        "question": "JVM 参数 -XX:MaxRAMPercentage 应该如何配置？",
        "sample_type": "no_answer",
        "expected_answerable": False,
        "expected_segment_indexes": [],
        "expected_note": "当前文档未说明 MaxRAMPercentage 参数配置，应避免把堆/新生代调优经验误当作该参数答案。",
    },
    {
        "question": "Java 21 虚拟线程在 JVM 中是如何调度的？",
        "sample_type": "no_answer",
        "expected_answerable": False,
        "expected_segment_indexes": [],
        "expected_note": "当前文档没有 Java 21 虚拟线程、Continuation 或调度器内容，应拒答。",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed JVM retrieval eval samples.")
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="目标 RAG 检索评测数据集业务 ID。",
    )
    parser.add_argument(
        "--created-by",
        default="liu_jiahao",
        help="样本标注人。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要新增/跳过的样本，不写入数据库。",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = _parse_args()
    db = SessionLocal()
    try:
        dataset = db.scalar(
            select(KnowledgeRetrievalEvalDataset).where(
                KnowledgeRetrievalEvalDataset.dataset_id == args.dataset_id
            )
        )
        if dataset is None:
            raise SystemExit(f"数据集不存在：{args.dataset_id}")
        if dataset.status != "draft":
            raise SystemExit(
                f"当前数据集状态为 {dataset.status}，只有 draft 数据集允许追加样本。"
            )

        existing_questions = set(
            db.scalars(
                select(KnowledgeRetrievalEvalSample.question).where(
                    KnowledgeRetrievalEvalSample.dataset_id == args.dataset_id
                )
            )
        )

        created_count = 0
        skipped_count = 0
        for sample in JVM_RETRIEVAL_SAMPLES:
            question = str(sample["question"])
            if question in existing_questions:
                skipped_count += 1
                print(f"SKIP   {question}")
                continue
            if args.dry_run:
                created_count += 1
                print(f"DRY    {question}")
                continue
            created = create_retrieval_eval_sample(
                db,
                dataset_id=args.dataset_id,
                question=question,
                sample_type=str(sample["sample_type"]),
                expected_answerable=bool(sample["expected_answerable"]),
                expected_segment_indexes=list(sample["expected_segment_indexes"]),
                expected_note=str(sample["expected_note"]),
                created_by=args.created_by,
            )
            existing_questions.add(question)
            created_count += 1
            print(f"CREATE {created.sample_id} {question}")

        if not args.dry_run:
            active_count = db.scalar(
                select(func.count())
                .select_from(KnowledgeRetrievalEvalSample)
                .where(
                    KnowledgeRetrievalEvalSample.dataset_id == args.dataset_id,
                    KnowledgeRetrievalEvalSample.status == "active",
                )
            ) or 0
            dataset = db.scalar(
                select(KnowledgeRetrievalEvalDataset).where(
                    KnowledgeRetrievalEvalDataset.dataset_id == args.dataset_id
                )
            )
            if dataset is not None:
                dataset.sample_count = active_count
                db.commit()

        print(
            f"完成：created={created_count}, skipped={skipped_count}, "
            f"dry_run={args.dry_run}, dataset_id={args.dataset_id}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
