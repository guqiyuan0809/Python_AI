"""
Demo 2: 余弦相似度计算
目标：理解相似度计算是 RAG 检索的核心原理
"""

import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    公式：cos(θ) = (A·B) / (||A|| * ||B||)
    - A·B: 向量点积
    - ||A||: 向量 A 的模（长度）
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


print("=" * 50)
print("Demo 1: 简单向量相似度")
print("=" * 50)

# 简单的 3 维向量示例
vec1 = [0.2, 0.8, 0.1]
vec2 = [0.3, 0.7, 0.2]  # 和 vec1 方向相似
vec3 = [0.9, 0.1, 0.5]  # 和 vec1 方向不同

sim_1_2 = cosine_similarity(vec1, vec2)
sim_1_3 = cosine_similarity(vec1, vec3)

print(f"vec1: {vec1}")
print(f"vec2: {vec2}")
print(f"vec3: {vec3}")
print(f"\nvec1 vs vec2 相似度: {sim_1_2:.3f} (相似)")
print(f"vec1 vs vec3 相似度: {sim_1_3:.3f} (不相似)")


print("\n" + "=" * 50)
print("Demo 2: 方向相同但长度不同")
print("=" * 50)

# 验证余弦相似度只看方向，不看长度
vec_a = [1, 2, 3]
vec_b = [2, 4, 6]  # vec_b 是 vec_a 的 2 倍
vec_c = [10, 20, 30]  # vec_c 是 vec_a 的 10 倍

sim_a_b = cosine_similarity(vec_a, vec_b)
sim_a_c = cosine_similarity(vec_a, vec_c)

print(f"vec_a: {vec_a}")
print(f"vec_b: {vec_b} (是 vec_a 的 2 倍)")
print(f"vec_c: {vec_c} (是 vec_a 的 10 倍)")
print(f"\nvec_a vs vec_b 相似度: {sim_a_b:.3f}")
print(f"vec_a vs vec_c 相似度: {sim_a_c:.3f}")
print("结论：方向相同，相似度都是 1.0，长度不影响相似度")


print("\n" + "=" * 50)
print("Demo 3: 方向相反")
print("=" * 50)

vec_x = [1, 2, 3]
vec_y = [-1, -2, -3]  # 方向完全相反

sim_x_y = cosine_similarity(vec_x, vec_y)

print(f"vec_x: {vec_x}")
print(f"vec_y: {vec_y} (方向完全相反)")
print(f"\nvec_x vs vec_y 相似度: {sim_x_y:.3f}")
print("结论：方向相反，相似度是 -1.0")


print("\n" + "=" * 50)
print("Demo 4: 相似度范围总结")
print("=" * 50)

print("""
余弦相似度范围：-1.0 到 1.0

 1.0  : 完全相同方向（完全相似）
 0.8-1.0 : 非常相似
 0.5-0.8 : 有些相似
 0.0-0.5 : 不太相似
 0.0  : 完全垂直（无关）
-1.0  : 完全相反方向（完全相反）

在文本相似度中，通常只会出现 0.0 到 1.0 的值
""")

print("=" * 50)
print("Demo 5: 不能简单靠正负号判断方向")
print("=" * 50)

# 两个向量全是正数，但方向不同
vec_p = [0.2, 0.8, 0.1]  # 第二个维度占主导
vec_q = [0.9, 0.1, 0.5]  # 第一个维度占主导

sim_p_q = cosine_similarity(vec_p, vec_q)

print(f"vec_p: {vec_p}  （第二个维度 0.8 最大）")
print(f"vec_q: {vec_q}  （第一个维度 0.9 最大）")
print(f"\n两个向量全是正数，但相似度: {sim_p_q:.3f}（不太相似）")
print("结论：方向取决于各维度的比例关系，不是正负号")
print("正负号完全相反（如 [1,2,3] vs [-1,-2,-3]）确实方向相反，但这只是极端情况")


print("\n" + "=" * 50)
print("✅ 完成！余弦相似度核心总结")
print("=" * 50)

print("""
核心要点：
1. 余弦相似度只看方向，不看长度
   - [1,2,3] 和 [2,4,6] 方向一样 → 相似度 1.0（长度不影响）

2. 方向 = 各维度的比例关系，不是正负号
   - [0.2, 0.8, 0.1] 和 [0.9, 0.1, 0.5] 全是正数但方向不同 → 相似度 0.66

3. 实际开发中看数值就行：
   - > 0.8：非常相似
   - 0.5-0.8：有些相似
   - < 0.5：不太相关

4. 你不需要手写这个函数，向量数据库（Chroma）内部自动帮你算
   理解原理就够了
""")
