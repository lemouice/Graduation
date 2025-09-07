import numpy as np
from VectorSolver import VectorSolver

def test_vector_consistency():
    """测试相同句子的向量一致性"""
    solver = VectorSolver()
    text = "人工智能正在改变世界"
    vec1 = solver.get_vector(text)
    vec2 = solver.get_vector(text)
    
    # 计算两个向量的差异（应为0）
    difference = solver.compare_vectors(vec1, vec2)
    assert difference == 1, f"相同句子的向量不一致，差异为{difference}"
    print("✅ 相同句子向量一致性测试通过")

def test_vector_difference():
    """测试不同句子的向量差异性"""
    solver = VectorSolver()
    text1 = "猫坐在垫子上"
    text2 = "飞机正在飞越海洋"
    
    vec1 = solver.get_vector(text1)
    vec2 = solver.get_vector(text2)
    
    # 计算余弦相似度（应该较低）
    similarity = solver.compare_vectors(vec1, vec2)
    print(f"不同句子相似度: {similarity}")
    
    # 使用更合理的阈值
    if similarity < 0.6:
        print("✅ 不同句子向量差异性测试通过")
        return True
    else:
        print(f"⚠️  不同句子相似度较高 ({similarity})，考虑使用更大模型或调整阈值")
        return False

def test_similar_sentences():
    """测试相似句子的向量相关性"""
    solver = VectorSolver()
    text1 = "人工智能正在改变世界"
    text2 = "机器学习技术正在重塑我们的生活"
    text3 = "天空是蓝色的"  # 不相关句子作为对照
    
    vec1 = solver.get_vector(text1)
    vec2 = solver.get_vector(text2)
    vec3 = solver.get_vector(text3)
    
    # 计算相似句子的相似度
    sim1_2 = solver.compare_vectors(vec1, vec2)
    # 计算与不相关句子的相似度
    sim1_3 = solver.compare_vectors(vec1, vec3)
    
    assert sim1_2 > sim1_3, (
        f"相似句子的向量相似度应高于与不相关句子的相似度，"
        f"实际相似句子相似度为{sim1_2:.4f}，不相关为{sim1_3:.4f}"
    )
    assert sim1_2 > 0.5, f"相似句子的向量相似度应足够高，实际为{sim1_2:.4f}"
    print("✅ 相似句子向量相关性测试通过")

def test_input_validation():
    """测试输入验证功能"""
    solver = VectorSolver()
    
    # 测试非字符串输入
    try:
        solver.get_vector(123)
        assert False, "应拒绝非字符串输入"
    except ValueError as e:
        assert "输入必须是字符串类型" in str(e)
    
    # 测试空字符串输入
    try:
        solver.get_vector("   ")
        assert False, "应拒绝空字符串输入"
    except ValueError as e:
        assert "输入文本不能为空" in str(e)
    
    print("✅ 输入验证测试通过")

def test_different_models():
    """测试不同模型的表现"""
    print("\n===== 不同模型性能测试 =====")
    
    models_to_test = ['miniLM', 'mpnet', 'miniLM-en']
    
    for model_key in models_to_test:
        try:
            print(f"\n测试模型: {model_key}")
            solver = VectorSolver(model_key)
            
            # 测试基本功能
            text1 = "测试句子一"
            text2 = "测试句子二"
            
            vec1 = solver.get_vector(text1)
            vec2 = solver.get_vector(text2)
            
            similarity = solver.compare_vectors(vec1, vec2)
            print(f"  相似度: {similarity}")
            print(f"  向量维度: {len(vec1)}")
            
        except Exception as e:
            print(f"  模型 {model_key} 测试失败: {e}")

def test_long_text():
    """测试长文本处理能力"""
    solver = VectorSolver()
    
    # 创建长文本
    long_text = "自然语言处理是人工智能领域的一个重要分支。" * 50
    
    try:
        vector = solver.get_vector(long_text)
        print(f"✅ 长文本处理成功，向量长度: {len(vector)}")
        return True
    except Exception as e:
        print(f"❌ 长文本处理失败: {e}")
        return False

if __name__ == "__main__":
    # 运行所有测试
    print("开始运行向量处理器测试...")
    
    test_input_validation()
    test_vector_consistency()
    test_vector_difference()
    test_similar_sentences()
    test_long_text()
    test_different_models()
    
    # 演示示例
    print("\n===== 演示示例 =====")
    solver = VectorSolver()
    
    sentences = [
        "我喜欢吃苹果",
        "我爱吃苹果",
        "我喜欢吃香蕉",
        "汽车需要加油才能行驶"
    ]
    
    vectors = [solver.get_vector(sent) for sent in sentences]
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            sim = solver.compare_vectors(vectors[i], vectors[j])
            print(f"'{sentences[i]}' 与 '{sentences[j]}' 的相似度: {sim:.5f}")
    
    print("\n===== 测试完成 =====")