"""
文件处理器模块的pytest测试

测试file_processor.py模块的各种功能
"""

import asyncio
import os
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import UploadFile

# 添加项目路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from page.file_processor import (
    process_upload,
    extract_web_content,
    detect_file_type,
    UnsupportedFileTypeError,
    FileSizeExceededError,
    FileProcessingError,
    InvalidURLError,
    WebScrapingError
)


class TestFileProcessing:
    """文件处理测试类"""
    
    @pytest.fixture
    def sample_txt_content(self):
        """创建测试用的TXT内容"""
        return """
        这是一个测试文档。
        
        包含多行文本内容，用于测试文件处理功能。
        
        测试内容包括：
        1. 中文字符
        2. 英文字符 English text
        3. 数字 123456
        4. 特殊符号 !@#$%^&*()
        
        这个文档应该能够被正确处理和提取。
        """.encode('utf-8')
    
    @pytest.fixture
    def sample_gbk_content(self):
        """创建测试用的GBK编码内容"""
        return "这是一个GBK编码的测试文档。包含中文字符。".encode('gbk')
    
    @pytest.fixture
    def large_file_content(self):
        """创建超大文件内容（>10MB）"""
        # 创建11MB的内容
        content = "这是一个测试文档。" * 1000000  # 约11MB
        return content.encode('utf-8')
    
    @pytest.fixture
    def corrupted_pdf_content(self):
        """创建损坏的PDF内容"""
        return "这不是一个有效的PDF文件内容".encode('utf-8')
    
    def create_upload_file(self, content: bytes, filename: str, content_type: str = None):
        """创建UploadFile对象"""
        file_obj = BytesIO(content)
        return UploadFile(
            filename=filename,
            file=file_obj,
            size=len(content),
            headers={"content-type": content_type or "application/octet-stream"}
        )
    
    @pytest.mark.asyncio
    async def test_process_txt_file(self, sample_txt_content):
        """测试TXT文件处理"""
        upload_file = self.create_upload_file(
            sample_txt_content, 
            "test.txt", 
            "text/plain"
        )
        
        result = await process_upload(upload_file)
        
        assert result is not None
        assert len(result) > 10
        assert "测试文档" in result
        assert "中文字符" in result
    
    @pytest.mark.asyncio
    async def test_process_gbk_encoded_file(self, sample_gbk_content):
        """测试GBK编码文件处理"""
        upload_file = self.create_upload_file(
            sample_gbk_content,
            "test_gbk.txt",
            "text/plain"
        )
        
        result = await process_upload(upload_file)
        
        assert result is not None
        assert len(result) > 5
        assert "GBK编码" in result
    
    @pytest.mark.asyncio
    async def test_file_type_detection(self, sample_txt_content):
        """测试文件类型检测"""
        mime_type = detect_file_type(sample_txt_content)
        assert mime_type == "text/plain"
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self):
        """测试不支持的文件类型"""
        # 创建一个假的EXE文件内容
        exe_content = b"MZ\x90\x00"  # PE文件头
        upload_file = self.create_upload_file(
            exe_content,
            "test.exe",
            "application/octet-stream"
        )
        
        with pytest.raises(UnsupportedFileTypeError):
            await process_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_file_size_exceeded(self, large_file_content):
        """测试文件大小超限"""
        upload_file = self.create_upload_file(
            large_file_content,
            "large_file.txt",
            "text/plain"
        )
        
        with pytest.raises(FileSizeExceededError):
            await process_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_corrupted_file(self):
        """测试损坏文件处理"""
        # 创建一个真正的损坏PDF文件内容（PDF文件头但内容损坏）
        corrupted_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n297\n%%EOF"
        
        upload_file = self.create_upload_file(
            corrupted_pdf,
            "corrupted.pdf",
            "application/pdf"
        )
        
        # 这个测试应该抛出FileProcessingError，因为PDF没有可提取的文本
        with pytest.raises(FileProcessingError):
            await process_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_empty_file(self):
        """测试空文件处理"""
        upload_file = self.create_upload_file(
            b"",
            "empty.txt",
            "text/plain"
        )
        
        with pytest.raises(UnsupportedFileTypeError):
            await process_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_file_size_validation(self):
        """测试文件大小验证"""
        # 测试正常大小文件
        normal_content = b"normal content"
        upload_file = self.create_upload_file(
            normal_content,
            "normal.txt",
            "text/plain"
        )
        
        # 应该不抛出异常
        result = await process_upload(upload_file)
        assert result is not None
        
        # 测试超大文件
        large_content = b"x" * (10 * 1024 * 1024 + 1)  # 超过10MB
        large_upload_file = self.create_upload_file(
            large_content,
            "large.txt",
            "text/plain"
        )
        
        with pytest.raises(FileSizeExceededError):
            await process_upload(large_upload_file)


class TestWebContentExtraction:
    """网页内容提取测试类"""
    
    @pytest.mark.asyncio
    async def test_extract_static_webpage(self):
        """测试静态网页抓取"""
        # 使用一个简单的静态网页进行测试
        url = "https://httpbin.org/html"
        
        result = await extract_web_content(url)
        
        assert result is not None
        assert "title" in result
        assert "content" in result
        assert "source_url" in result
        assert result["source_url"] == url
        assert len(result["content"]) > 10
    
    @pytest.mark.asyncio
    async def test_extract_wikipedia_page(self):
        """测试Wikipedia页面抓取"""
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        result = await extract_web_content(url)
        
        assert result is not None
        assert "Python" in result["title"]
        assert len(result["content"]) > 100
        assert "programming" in result["content"].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_url(self):
        """测试无效URL"""
        invalid_urls = [
            "invalid-url",
            "ftp://example.com",
            "not-a-url",
            ""
        ]
        
        for url in invalid_urls:
            with pytest.raises(InvalidURLError):
                await extract_web_content(url)
    
    @pytest.mark.asyncio
    async def test_nonexistent_url(self):
        """测试不存在的URL"""
        url = "https://this-domain-does-not-exist-12345.com"
        
        with pytest.raises(WebScrapingError):
            await extract_web_content(url)
    
    @pytest.mark.asyncio
    async def test_web_content_structure(self):
        """测试网页内容结构"""
        url = "https://httpbin.org/html"
        
        result = await extract_web_content(url)
        
        # 验证返回结构
        required_fields = ["title", "content", "source_url"]
        for field in required_fields:
            assert field in result
        
        # 验证数据类型
        assert isinstance(result["title"], str)
        assert isinstance(result["content"], str)
        assert isinstance(result["source_url"], str)
        
        # 可选字段
        if "authors" in result:
            assert isinstance(result["authors"], list)
        if "publish_date" in result:
            assert result["publish_date"] is None or isinstance(result["publish_date"], str)


class TestErrorHandling:
    """错误处理测试类"""
    
    @pytest.mark.asyncio
    async def test_file_processing_error_handling(self):
        """测试文件处理错误处理"""
        # 测试各种错误情况
        test_cases = [
            # (content, filename, expected_exception)
            (b"", "empty.txt", UnsupportedFileTypeError),
            (b"x" * (11 * 1024 * 1024), "large.txt", FileSizeExceededError),
        ]
        
        for content, filename, expected_exception in test_cases:
            upload_file = UploadFile(
                filename=filename,
                file=BytesIO(content),
                size=len(content),
                headers={"content-type": "application/octet-stream"}
            )
            
            with pytest.raises(expected_exception):
                await process_upload(upload_file)
    
    @pytest.mark.asyncio
    async def test_web_scraping_error_handling(self):
        """测试网页抓取错误处理"""
        # 测试各种错误情况
        test_cases = [
            ("invalid-url", InvalidURLError),
            ("https://this-domain-does-not-exist-12345.com", WebScrapingError),
            ("", InvalidURLError),
        ]
        
        for url, expected_exception in test_cases:
            with pytest.raises(expected_exception):
                await extract_web_content(url)


class TestPerformance:
    """性能测试类"""
    
    @pytest.mark.asyncio
    async def test_processing_time(self):
        """测试处理时间"""
        import time
        
        # 创建测试内容
        sample_content = "这是一个测试文档。包含中文字符和英文字符。Test content with English.".encode('utf-8')
        upload_file = UploadFile(
            filename="test.txt",
            file=BytesIO(sample_content),
            size=len(sample_content),
            headers={"content-type": "text/plain"}
        )
        
        start_time = time.time()
        result = await process_upload(upload_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result is not None
        assert processing_time < 5.0  # 应该在5秒内完成
    
    @pytest.mark.asyncio
    async def test_web_scraping_time(self):
        """测试网页抓取时间"""
        import time
        
        url = "https://httpbin.org/html"
        
        start_time = time.time()
        result = await extract_web_content(url)
        end_time = time.time()
        
        scraping_time = end_time - start_time
        
        assert result is not None
        assert scraping_time < 30.0  # 应该在30秒内完成


# 测试资源文件创建
def create_test_resources():
    """创建测试资源文件"""
    test_dir = Path(__file__).parent / "test_resources"
    test_dir.mkdir(exist_ok=True)
    
    # 创建测试TXT文件
    txt_file = test_dir / "test.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("这是一个测试TXT文件。\n包含中文字符和英文字符。\nTest content with English.")
    
    # 创建GBK编码的TXT文件
    gbk_file = test_dir / "test_gbk.txt"
    with open(gbk_file, 'w', encoding='gbk') as f:
        f.write("这是一个GBK编码的测试文件。")
    
    # 创建损坏的PDF文件
    corrupted_pdf = test_dir / "corrupted.pdf"
    with open(corrupted_pdf, 'wb') as f:
        f.write(b"This is not a valid PDF file content")
    
    print(f"测试资源文件已创建在: {test_dir}")


if __name__ == "__main__":
    # 创建测试资源文件
    create_test_resources()
    
    # 运行pytest
    pytest.main([__file__, "-v", "--tb=short"])
