#!/usr/bin/env python3
"""
sentence_encoder.py 测试模块

测试句子编码器的各种功能，包括单文本编码、批量编码、相似度计算等。
"""

import asyncio
import sys
import os
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page.sentence_encoder import SentenceEncoder, encode, similarity


async def test_different_length_texts():
    """测试不同长度文本编码"""
    print("\n🔍 测试不同长度文本编码:")
    
    encoder = SentenceEncoder()
    await encoder.load_model()
    
    test_cases = [
        ("短文本", "Hello"),
        ("中等文本", "这是一个中等长度的测试文本，包含一些中文和英文内容。"),
        ("长文本", "这是一个很长的测试文本。" * 100),  # 约500词
        ("超长文本", "这是一个超长的测试文本。" * 200),  # 约1000词
    ]
    
    for desc, text in test_cases:
        try:
            start_time = time.time()
            embedding = await encoder.encode(text)
            duration = time.time() - start_time
            word_count = len(text.split())
            print(f"   ✅ {desc}: {word_count}词, {embedding.shape[0]}维, {duration:.3f}秒")
        except Exception as e:
            print(f"   ❌ {desc}: 失败 - {e}")


async def test_multilingual_similarity():
    """测试多语言相似度计算"""
    print("\n🌍 测试多语言相似度计算:")
    
    encoder = SentenceEncoder()
    await encoder.load_model()
    
    test_pairs = [
        ("今天天气很好", "今天天气不错", "中文相似"),
        ("Hello world", "Hi there", "英文相似"),
        ("今天天气很好", "Hello world", "中英混合"),
        ("今天天气很好", "明天要下雨", "中文不相似"),
        ("The weather is nice today", "Il fait beau aujourd'hui", "英法混合"),
    ]
    
    for text1, text2, desc in test_pairs:
        try:
            similarity_score = await encoder.similarity(text1, text2)
            print(f"   ✅ {desc}: {similarity_score:.4f}")
        except Exception as e:
            print(f"   ❌ {desc}: 失败 - {e}")


async def test_batch_encoding_performance():
    """测试批量编码性能"""
    print("\n⚡ 测试批量编码性能:")
    
    encoder = SentenceEncoder()
    await encoder.load_model()
    
    # 生成测试数据
    test_texts = [f"这是第{i}个测试文本，包含一些中文内容。" for i in range(100)]
    
    try:
        start_time = time.time()
        embeddings = await encoder.encode_batch(test_texts, show_progress=True)
        duration = time.time() - start_time
        
        print(f"   ✅ 批量编码: {len(test_texts)}个文本, {embeddings.shape}, {duration:.3f}秒")
        print(f"   📊 平均速度: {len(test_texts)/duration:.1f} 文本/秒")
        
        # 显示性能统计
        stats = encoder.get_stats()
        print(f"   📈 性能统计: {stats}")
        
    except Exception as e:
        print(f"   ❌ 批量编码失败: {e}")


async def test_singleton_pattern():
    """测试单例模式"""
    print("\n🔧 测试单例模式:")
    
    encoder1 = SentenceEncoder()
    encoder2 = SentenceEncoder()
    
    print(f"   ✅ 单例模式: {encoder1 is encoder2}")
    print(f"   ✅ 模型实例: {encoder1.model is encoder2.model}")


async def test_convenience_functions():
    """测试便捷函数"""
    print("\n🚀 测试便捷函数:")
    
    try:
        # 测试便捷编码函数
        embedding = await encode("测试文本")
        print(f"   ✅ 便捷编码函数: {embedding.shape}")
        
        # 测试便捷相似度函数
        sim = await similarity("今天天气很好", "今天天气不错")
        print(f"   ✅ 便捷相似度函数: {sim:.4f}")
        
    except Exception as e:
        print(f"   ❌ 便捷函数测试失败: {e}")


async def test_error_handling():
    """测试错误处理"""
    print("\n🛡️ 测试错误处理:")
    
    encoder = SentenceEncoder()
    
    try:
        # 测试空文本
        empty_embedding = await encoder.encode("")
        print(f"   ✅ 空文本处理: {empty_embedding.shape}")
        
        # 测试空文本列表
        empty_batch = await encoder.encode_batch([])
        print(f"   ✅ 空文本列表处理: {empty_batch.shape}")
        
        # 测试空文本相似度
        empty_sim = await encoder.similarity("", "")
        print(f"   ✅ 空文本相似度: {empty_sim}")
        
    except Exception as e:
        print(f"   ❌ 错误处理测试失败: {e}")


async def test_environment_variables():
    """测试环境变量配置"""
    print("\n⚙️ 测试环境变量配置:")
    
    # 测试默认配置
    encoder = SentenceEncoder()
    print(f"   ✅ 默认模型: {encoder.model_name}")
    print(f"   ✅ 默认设备: {encoder.device}")
    print(f"   ✅ 离线模式: {encoder.offline_mode}")


async def test_performance_monitoring():
    """测试性能监控"""
    print("\n📊 测试性能监控:")
    
    encoder = SentenceEncoder()
    await encoder.load_model()
    
    # 重置统计
    encoder.reset_stats()
    
    # 执行一些操作
    await encoder.encode("测试文本1")
    await encoder.encode("测试文本2")
    await encoder.similarity("文本1", "文本2")
    
    # 获取统计信息
    stats = encoder.get_stats()
    print(f"   ✅ 总编码次数: {stats['total_encodings']}")
    print(f"   ✅ 总相似度计算: {stats['total_similarities']}")
    print(f"   ✅ 平均编码时间: {stats['avg_encoding_time']:.3f}秒")
    print(f"   ✅ 处理速度: {stats['throughput']:.1f} 文本/秒")
    print(f"   ✅ 峰值内存: {stats['peak_memory']:.1f}MB")


async def main():
    """主函数 - 运行所有测试"""
    print("🚀 开始测试SentenceEncoder v2.0功能")
    print("=" * 60)
    
    try:
        # 测试单例模式
        await test_singleton_pattern()
        
        # 测试环境变量配置
        await test_environment_variables()
        
        # 测试便捷函数
        await test_convenience_functions()
        
        # 测试错误处理
        await test_error_handling()
        
        # 测试不同长度文本编码
        await test_different_length_texts()
        
        # 测试多语言相似度计算
        await test_multilingual_similarity()
        
        # 测试批量编码性能
        await test_batch_encoding_performance()
        
        # 测试性能监控
        await test_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("🎉 SentenceEncoder v2.0功能测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
