#!/usr/bin/env python3
"""
简单的数据库测试脚本
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page.database import (
    init_db, get_db, ArticleCreate, ReportCreate, ReportType,
    create_article, get_article, update_article, delete_article,
    create_report, get_report, update_report, delete_report,
    check_database_health, logger
)

async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 开始测试数据库基本功能...")
    
    try:
        # 1. 初始化数据库
        print("1️⃣ 初始化数据库...")
        await init_db()
        print("✅ 数据库初始化成功")
        
        # 2. 健康检查
        print("2️⃣ 健康检查...")
        health = await check_database_health()
        print(f"✅ 健康状态: {health['status']}")
        
        # 3. 测试文章CRUD
        print("3️⃣ 测试文章CRUD...")
        async for session in get_db():
            # 创建文章
            article_data = ArticleCreate(
                title="测试文章",
                author="测试作者",
                source_url="https://example.com/test",
                content="这是一篇测试文章的内容"
            )
            article = await create_article(session, article_data)
            print(f"✅ 创建文章成功: {article.id}")
            
            # 获取文章
            retrieved = await get_article(session, article.id)
            if retrieved:
                print(f"✅ 获取文章成功: {retrieved.title}")
            else:
                print("❌ 获取文章失败")
                return False
            
            # 更新文章
            from page.database import ArticleUpdate
            update_data = ArticleUpdate(title="更新后的标题")
            updated = await update_article(session, article.id, update_data)
            if updated:
                print(f"✅ 更新文章成功: {updated.title}")
            else:
                print("❌ 更新文章失败")
                return False
            
            # 删除文章
            deleted = await delete_article(session, article.id)
            if deleted:
                print("✅ 删除文章成功")
            else:
                print("❌ 删除文章失败")
                return False
        
        # 4. 测试报告CRUD
        print("4️⃣ 测试报告CRUD...")
        async for session in get_db():
            # 创建报告
            report_data = ReportCreate(
                user_id=1,
                report_type=ReportType.SINGLE,
                results={"similarity_score": 0.85, "matched_articles": []}
            )
            report = await create_report(session, report_data)
            print(f"✅ 创建报告成功: {report.id}")
            
            # 获取报告
            retrieved_report = await get_report(session, report.id)
            if retrieved_report:
                print(f"✅ 获取报告成功: {retrieved_report.report_type}")
            else:
                print("❌ 获取报告失败")
                return False
            
            # 更新报告
            from page.database import ReportUpdate
            update_data = ReportUpdate(results={"similarity_score": 0.90, "matched_articles": []})
            updated_report = await update_report(session, report.id, update_data)
            if updated_report:
                print("✅ 更新报告成功")
            else:
                print("❌ 更新报告失败")
                return False
            
            # 删除报告
            deleted_report = await delete_report(session, report.id)
            if deleted_report:
                print("✅ 删除报告成功")
            else:
                print("❌ 删除报告失败")
                return False
        
        print("\n🎉 所有测试通过！数据库模块工作正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.error(f"测试失败: {e}")
        return False

async def test_constraints():
    """测试约束"""
    print("\n🔒 测试数据库约束...")
    
    try:
        async for session in get_db():
            # 测试唯一约束
            article1_data = ArticleCreate(
                title="第一篇文章",
                source_url="https://example.com/unique-test",
                content="内容1"
            )
            article1 = await create_article(session, article1_data)
            print("✅ 创建第一篇文章成功")
            
            # 尝试创建相同URL的文章
            article2_data = ArticleCreate(
                title="第二篇文章",
                source_url="https://example.com/unique-test",  # 相同URL
                content="内容2"
            )
            
            try:
                await create_article(session, article2_data)
                print("❌ 唯一约束测试失败: 应该抛出异常")
                return False
            except Exception as e:
                if "UNIQUE constraint failed" in str(e) or "ConstraintError" in str(e):
                    print("✅ 唯一约束测试成功")
                else:
                    print(f"❌ 唯一约束测试失败: {e}")
                    return False
            
            # 清理
            try:
                await delete_article(session, article1.id)
                print("✅ 清理测试数据成功")
            except Exception as e:
                print(f"⚠️ 清理数据时出现警告: {e}")
            return True
            
    except Exception as e:
        print(f"❌ 约束测试失败: {e}")
        return False

async def main():
    """主函数"""
    print("🚀 数据库模块测试开始")
    print("=" * 50)
    
    # 测试基本功能
    basic_ok = await test_basic_functionality()
    
    # 测试约束
    constraint_ok = await test_constraints()
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 测试结果总结:")
    print(f"基本功能测试: {'✅ 通过' if basic_ok else '❌ 失败'}")
    print(f"约束测试: {'✅ 通过' if constraint_ok else '❌ 失败'}")
    
    all_passed = basic_ok and constraint_ok
    print("=" * 50)
    print(f"总体结果: {'🎉 所有测试通过' if all_passed else '⚠️ 部分测试失败'}")
    print("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
