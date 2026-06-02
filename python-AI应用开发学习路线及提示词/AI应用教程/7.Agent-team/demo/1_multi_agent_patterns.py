"""
Demo 1: 多 Agent 协作基础概念演示
用纯 Python 模拟多 Agent 协作的三种模式（串行、并行、层级），
不依赖任何框架，帮助理解核心原理。

不需要 API Key
"""

# ============================================================
# 模拟 Agent 基类
# ============================================================

class SimpleAgent:
    """简单的 Agent 模拟类"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, input_data: str) -> str:
        """模拟 Agent 处理任务"""
        print(f"  [{self.name}]({self.role}) 正在处理...")
        result = f"[{self.name}的输出] 基于输入「{input_data[:30]}...」完成了{self.role}工作"
        print(f"  [{self.name}] 完成")
        return result


# ============================================================
# 模式 1：串行模式（Pipeline）
# ============================================================

def demo_sequential():
    """串行模式：Agent 按顺序依次执行"""
    print("\n" + "=" * 60)
    print("模式 1：串行模式（Sequential / Pipeline）")
    print("=" * 60)
    print("流程：研究员 → 写作者 → 审核者\n")

    researcher = SimpleAgent("研究员", "信息收集与分析")
    writer = SimpleAgent("写作者", "内容撰写")
    reviewer = SimpleAgent("审核者", "质量审查")

    task = "分析 2024 年 AI Agent 发展趋势"
    print(f"任务：{task}\n")

    # 串行执行：每个 Agent 的输出是下一个的输入
    research_result = researcher.process(task)
    draft = writer.process(research_result)
    final = reviewer.process(draft)

    print(f"\n最终输出：{final}")


# ============================================================
# 模式 2：并行模式（Parallel）
# ============================================================

def demo_parallel():
    """并行模式：多个 Agent 同时处理不同子任务"""
    print("\n" + "=" * 60)
    print("模式 2：并行模式（Parallel）")
    print("=" * 60)
    print("流程：任务拆分 → 多个 Agent 并行 → 结果汇总\n")

    from concurrent.futures import ThreadPoolExecutor

    market_analyst = SimpleAgent("市场分析师", "市场趋势分析")
    tech_analyst = SimpleAgent("技术分析师", "技术可行性分析")
    finance_analyst = SimpleAgent("财务分析师", "成本收益分析")

    task = "评估是否应该投资 AI Agent 产品"
    print(f"任务：{task}\n")

    agents = [market_analyst, tech_analyst, finance_analyst]

    # 并行执行：使用线程池同时运行多个 Agent
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(agent.process, task) for agent in agents]
        results = [f.result() for f in futures]

    # 汇总结果
    print(f"\n汇总所有分析结果：")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result}")


# ============================================================
# 模式 3：层级模式（Hierarchical）
# ============================================================

def demo_hierarchical():
    """层级模式：Manager 分配任务，Worker 执行"""
    print("\n" + "=" * 60)
    print("模式 3：层级模式（Hierarchical / Manager）")
    print("=" * 60)
    print("流程：Manager 分析任务 → 分配给合适的 Worker → 收集结果\n")

    class ManagerAgent:
        """管理者 Agent：负责任务分配和结果汇总"""

        def __init__(self, workers: list):
            self.workers = {w.name: w for w in workers}

        def delegate(self, task: str) -> str:
            print(f"  [Manager] 分析任务：{task}")
            print(f"  [Manager] 决定分配给：研究员 和 写作者\n")

            # Manager 决定由谁来执行
            r1 = self.workers["研究员"].process(task)
            r2 = self.workers["写作者"].process(r1)

            print(f"\n  [Manager] 汇总结果，任务完成")
            return r2

    researcher = SimpleAgent("研究员", "信息收集")
    writer = SimpleAgent("写作者", "内容生成")
    coder = SimpleAgent("程序员", "代码实现")

    manager = ManagerAgent([researcher, writer, coder])

    task = "写一篇关于 LangGraph 的技术博客"
    print(f"任务：{task}\n")
    result = manager.delegate(task)
    print(f"\n最终输出：{result}")


# ============================================================
# 运行所有演示
# ============================================================

if __name__ == "__main__":
    print("多 Agent 协作模式演示")
    print("本 Demo 用纯 Python 模拟三种协作模式的核心逻辑")

    demo_sequential()
    demo_parallel()
    demo_hierarchical()

    print("\n" + "=" * 60)
    print("总结：")
    print("  串行：简单流水线，适合有先后顺序的任务")
    print("  并行：同时执行，适合独立子任务")
    print("  层级：Manager 协调，适合需要动态决策的复杂任务")
    print("=" * 60)
