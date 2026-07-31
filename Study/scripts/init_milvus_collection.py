"""创建或校验知识库 Milvus Collection。"""

import json

from day04_app.services.milvus_vector_store_service import ensure_knowledge_chunk_collection


def main() -> None:
    result = ensure_knowledge_chunk_collection()
    print("Milvus Collection 初始化完成")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
