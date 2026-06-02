"""
Demo 5: 完整 Agent 应用 — 公司客服助手 "小 A"
目标：把前 4 章学的全部串起来，做一个真实可用的 Agent

演示内容：
1. 业务工具：查订单、计算优惠、当前时间
2. RAG 工具：把检索器包成 Tool（用 create_retriever_tool）
3. 记忆：用 RunnableWithMessageHistory
4. 完整的多轮对话演示

技术栈：
- LLM: DeepSeek
- Embedding: Ollama (nomic-embed-text)
- Vector Store: Chroma
- Memory: 内存存储（生产用 Redis）

需要 API Key + 本地 Ollama

前置条件:
- pip install langchain langchain-openai langchain-ollama langchain-chroma langchain-community pypdf
- ollama pull nomic-embed-text
- 把下面的 API_KEY 改成你自己的
"""

import os
import shutil
from datetime import datetime

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ============================================================
# 配置
# ============================================================
API_KEY = "sk-2e56e7e02258424eaf6afe699b054031"
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"

DOCS_DIR = "./sample_docs"
PERSIST_DIR = "./company_kb"


# ============================================================
# 第 1 步:构建知识库 (RAG 部分)
# ============================================================
def build_knowledge_base():
    """构建公司知识库（首次运行）"""
    print("=" * 60)
    print("第 1 步：构建公司知识库")
    print("=" * 60)

    # 加载所有 txt 文档
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    print(f"  加载了 {len(docs)} 个文档")

    # 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    print(f"  分成了 {len(chunks)} 个块")

    # 向量化 + 存储
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 清理旧数据
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"  知识库构建完成 → {PERSIST_DIR}/")
    return vectorstore


# ============================================================
# 第 2 步:定义所有工具
# ============================================================
def build_tools(vectorstore):
    """组装所有工具：RAG + 业务工具"""
    print("\n" + "=" * 60)
    print("第 2 步：定义工具")
    print("=" * 60)

    # ----- 工具 1: RAG 工具 (把检索器包成 Tool) -----
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    knowledge_tool = create_retriever_tool(
        retriever,
        name="company_knowledge_base",
        description=(
            "搜索公司知识库，包括产品介绍、退款政策、配送规则、客服流程等内部信息。"
            "当用户询问公司业务、产品功能、政策规则相关问题时使用此工具。"
        ),
    )

    # ----- 工具 2: 查订单 -----
    FAKE_ORDERS = {
        "ORD-12345": {"status": "已发货，预计明天送达", "amount": 199.0, "item": "蓝牙耳机"},
        "ORD-67890": {"status": "已签收", "amount": 599.0, "item": "无线键盘"},
        "ORD-99999": {"status": "正在打包", "amount": 89.0, "item": "数据线"},
    }

    @tool
    def get_order_status(order_id: str) -> str:
        """查询订单状态、金额、商品名称。

        Args:
            order_id: 订单号，例如 "ORD-12345"，订单号一般以 "ORD-" 开头
        """
        order = FAKE_ORDERS.get(order_id)
        if not order:
            return f"未找到订单 {order_id}，请确认订单号是否正确"
        return f"订单 {order_id}: {order['item']}, 金额 {order['amount']} 元, 状态：{order['status']}"

    # ----- 工具 3: 计算器 -----
    @tool
    def calculator(expression: str) -> str:
        """执行数学计算，支持加减乘除。

        Args:
            expression: 数学表达式，例如 "200 * 0.8" 表示 200 元打 8 折
        """
        try:
            return str(eval(expression))
        except Exception as e:
            return f"计算错误: {e}"

    # ----- 工具 4: 当前时间 -----
    @tool
    def get_current_time() -> str:
        """获取当前的日期和时间"""
        return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    tools = [knowledge_tool, get_order_status, calculator, get_current_time]
    print(f"  共定义 {len(tools)} 个工具：")
    for t in tools:
        print(f"    - {t.name}")
    return tools


# ============================================================
# 第 3 步:创建带记忆的 Agent
# ============================================================
def build_agent(tools):
    """创建 Agent + AgentExecutor + Memory"""
    print("\n" + "=" * 60)
    print("第 3 步：创建带记忆的 Agent")
    print("=" * 60)

    # System Prompt
    SYSTEM_PROMPT = """你是公司客服助手，名字叫小 A。

【你的能力】
1. 用 company_knowledge_base 查询公司产品、政策、规则
2. 用 get_order_status 查询订单状态（订单号格式 ORD-XXXXX）
3. 用 calculator 进行数学计算（折扣、优惠后价格）
4. 用 get_current_time 获取当前时间

【行为准则】
- 优先使用工具获取真实信息，不要编造
- 用户问"我的订单"时，引导用户提供订单号
- 涉及公司业务的问题先查知识库
- 保持友好、简洁、专业的语气
- 用中文回答
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # LLM
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0,
    )

    # Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    # 记忆容器
    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 包装记忆
    agent_with_history = RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    print("  Agent 创建完成（带 RAG + 业务工具 + 记忆）")
    return agent_with_history


# ============================================================
# 第 4 步:运行多轮对话
# ============================================================
def run_conversation(agent, session_id="user_001"):
    print("\n" + "=" * 60)
    print(f"第 4 步：多轮对话演示（session={session_id}）")
    print("=" * 60)

    config = {"configurable": {"session_id": session_id}}

    questions = [
        "你好，我叫小明",                              # 测试自我介绍
        "你们的退款政策是什么？",                      # 测试 RAG
        "我的订单 ORD-12345 现在是什么状态？",         # 测试业务工具
        "现在几点了？",                                # 测试时间工具
        "一件 199 块的商品打 8 折后是多少？",          # 测试计算
        "我刚才订的什么？",                            # 测试记忆（要记得 ORD-12345 是耳机）
        "我叫什么名字？",                              # 测试记忆（最早说的"小明"）
    ]

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"👤 用户: {q}")
        print('─' * 60)

        result = agent.invoke({"input": q}, config=config)

        print(f"\n🤖 小A: {result['output']}")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":

    # 第 1 步：构建知识库
    vectorstore = build_knowledge_base()

    # 第 2 步：定义工具
    tools = build_tools(vectorstore)

    # 第 3 步：创建 Agent
    agent = build_agent(tools)

    # 第 4 步：开聊
    run_conversation(agent, session_id="user_xiaoming")

    # ============================================================
    # 总结
    # ============================================================
    print("\n\n" + "=" * 60)
    print("完成！这就是一个完整的 Agent 应用")
    print("=" * 60)
    print("""
组件回顾：
  ┌──────────────────────────────────────────┐
  │   RAG 工具 (knowledge_base)              │
  │   业务工具 (订单/计算/时间)              │
  │           ↓                              │
  │   create_tool_calling_agent              │
  │           ↓                              │
  │   AgentExecutor (循环、防死循环)         │
  │           ↓                              │
  │   RunnableWithMessageHistory (记忆)      │
  └──────────────────────────────────────────┘

四种问题的处理：
  - "退款政策" → 走 RAG 工具
  - "订单状态" → 走业务工具
  - "现在几点" → 走时间工具
  - "我叫什么"  → 不调工具，从历史里找

升级到生产环境：
  1. 把内存 store 换成 Redis
     from langchain_community.chat_message_histories import RedisChatMessageHistory
  2. 长对话用 trim_messages 截断
  3. 接入 LangSmith 做调用追踪
  4. 流式输出用 executor.astream_events
  5. 加上人工介入（human-in-the-loop）

恭喜！你已经能独立做一个 Agent 项目了。
""")
