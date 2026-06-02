"""
Demo 2: CrewAI 多 Agent 协作
使用 CrewAI 框架构建一个"内容创作团队"：研究员 + 写作者。

需要 API Key（配置下方变量）
安装依赖：pip install crewai crewai-tools
"""

from crewai import Agent, Task, Crew, Process

# ============================================================
# API 配置（替换为你自己的配置）
# ============================================================
import os

API_KEY = "your-api-key-here"
BASE_URL = "https://api.openai.com/v1"  # 或其他兼容 API 地址
MODEL = "gpt-4o-mini"  # 或其他模型

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_API_BASE"] = BASE_URL
os.environ["OPENAI_MODEL_NAME"] = MODEL

# ============================================================
# 定义 Agent（团队成员）
# ============================================================

researcher = Agent(
    role="高级研究分析师",
    goal="收集并分析关于 {topic} 的最新信息和趋势",
    backstory=(
        "你是一位资深的技术研究分析师，擅长从海量信息中"
        "提取关键洞察，善于发现趋势和模式。"
    ),
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="技术内容写作专家",
    goal="基于研究结果，撰写一篇通俗易懂的技术文章",
    backstory=(
        "你是一位优秀的技术写作者，擅长将复杂的技术概念"
        "转化为易于理解的内容，文风简洁有力。"
    ),
    verbose=True,
    allow_delegation=False,
)

# ============================================================
# 定义 Task（任务）
# ============================================================

research_task = Task(
    description=(
        "对 {topic} 进行深入研究。"
        "收集最新的发展动态、关键技术、主要玩家和未来趋势。"
        "提供至少 5 个关键发现，每个发现需要有具体的数据或案例支撑。"
    ),
    expected_output="一份结构化的研究报告，包含关键发现、数据支撑和趋势分析。",
    agent=researcher,
)

writing_task = Task(
    description=(
        "基于研究报告，撰写一篇 500 字左右的技术博客文章。"
        "要求：通俗易懂、结构清晰、有具体案例。"
        "目标读者是有一定技术背景但不熟悉该领域的开发者。"
    ),
    expected_output="一篇结构完整的技术博客文章，包含标题、引言、正文和总结。",
    agent=writer,
)

# ============================================================
# 创建 Crew（团队）并执行
# ============================================================

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

if __name__ == "__main__":
    print("CrewAI 多 Agent 协作 Demo")
    print("=" * 50)
    print("团队成员：研究分析师 + 内容写作专家")
    print("执行模式：串行（先研究，再写作）")
    print("=" * 50)

    result = crew.kickoff(inputs={"topic": "AI Agent 多智能体协作"})

    print("\n" + "=" * 50)
    print("最终输出：")
    print("=" * 50)
    print(result)
