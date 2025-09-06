"""
文件处理器模块 - 精简版

支持处理多种文件格式和网页内容抓取
包括PDF、DOCX、DOC、TXT文件处理以及网页内容提取
"""

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urlparse

import aiofiles
import docx2txt
import magic
import pdfplumber
import requests
from bs4 import BeautifulSoup
from fastapi import UploadFile
from newspaper import Article
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, validator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('file_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 配置常量
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
WEB_SCRAPING_TIMEOUT = 10
SUPPORTED_MIME_TYPES = {
    'application/pdf': 'pdf',
    'application/msword': 'doc',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'text/plain': 'txt'
}
TEXT_ENCODINGS = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin-1', 'cp1252']


class WebContentResult(BaseModel):
    """网页内容提取结果"""
    title: str = Field(..., description="网页标题")
    content: str = Field(..., description="网页正文内容")
    authors: List[str] = Field(default_factory=list, description="作者列表")
    publish_date: Optional[datetime] = Field(None, description="发布日期")
    source_url: str = Field(..., description="原始URL")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("网页内容过短或为空")
        return v.strip()


# 自定义异常类
class UnsupportedFileTypeError(Exception):
    """不支持的文件类型异常"""
    pass


class FileSizeExceededError(Exception):
    """文件大小超限异常"""
    pass


class FileProcessingError(Exception):
    """文件处理异常"""
    pass


class InvalidURLError(Exception):
    """无效URL异常"""
    pass


class WebScrapingError(Exception):
    """网页抓取异常"""
    pass


def detect_file_type(file_content: bytes) -> str:
    """使用python-magic检测真实文件类型"""
    try:
        return magic.from_buffer(file_content, mime=True)
    except Exception:
        return "application/octet-stream"


def validate_url(url: str) -> None:
    """验证URL格式"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc or parsed.scheme not in ['http', 'https']:
            raise InvalidURLError("URL格式无效")
    except Exception as e:
        raise InvalidURLError(f"URL验证失败: {e}")


async def extract_text_from_file(file_path: str, file_type: str) -> str:
    """统一的文件文本提取函数"""
    try:
        if file_type == 'application/pdf':
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                if not text.strip():
                    raise FileProcessingError("PDF文件中未找到可提取的文本内容")
                return text
                
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = docx2txt.process(file_path)
            if not text.strip():
                raise FileProcessingError("DOCX文件中未找到可提取的文本内容")
            return text
            
        elif file_type == 'application/msword':
            # 简化的DOC处理 - 实际项目中需要安装antiword
            raise FileProcessingError("DOC文件处理需要安装antiword，请使用DOCX格式")
            
        elif file_type == 'text/plain':
            # 尝试多种编码
            for encoding in TEXT_ENCODINGS:
                try:
                    async with aiofiles.open(file_path, mode='r', encoding=encoding) as f:
                        text = await f.read()
                    if text.strip():
                        return text
                except UnicodeDecodeError:
                    continue
            raise FileProcessingError("无法使用任何编码读取TXT文件")
            
        else:
            raise FileProcessingError(f"不支持的文件类型: {file_type}")
            
    except Exception as e:
        logger.error(f"文件文本提取失败: {e}")
        raise FileProcessingError(f"文件文本提取失败: {e}")


async def extract_web_content_with_method(url: str, method: str) -> Dict:
    """统一的网页内容提取函数"""
    try:
        if method == 'playwright':
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                page.set_default_timeout(WEB_SCRAPING_TIMEOUT * 1000)
                
                await page.goto(url, wait_until='domcontentloaded', timeout=WEB_SCRAPING_TIMEOUT * 1000)
                await page.wait_for_timeout(2000)
                
                title = await page.title()
                content_selectors = ['article', '.content', '.post-content', 'main', '.main-content']
                
                content = ""
                for selector in content_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            content = await element.inner_text()
                            if content and len(content.strip()) > 100:
                                break
                    except Exception:
                        continue
                
                if not content or len(content.strip()) < 100:
                    body = await page.query_selector('body')
                    if body:
                        content = await body.inner_text()
                
                await browser.close()
                
                if not content or len(content.strip()) < 10:
                    raise WebScrapingError("无法提取有效的网页内容")
                
                return {
                    'title': title or "无标题",
                    'content': content.strip(),
                    'authors': [],
                    'publish_date': None,
                    'source_url': url
                }
                
        elif method == 'newspaper':
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text or len(article.text.strip()) < 10:
                raise WebScrapingError("无法提取有效的网页内容")
            
            return {
                'title': article.title or "无标题",
                'content': article.text.strip(),
                'authors': list(article.authors) if article.authors else [],
                'publish_date': article.publish_date,
                'source_url': url
            }
            
        elif method == 'beautifulsoup':
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=WEB_SCRAPING_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title = title.get_text().strip() if title else "无标题"
            
            content_selectors = ['article', '.content', '.post-content', 'main', '.main-content']
            content = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content = element.get_text().strip()
                    if content and len(content) > 100:
                        break
            
            if not content or len(content) < 100:
                body = soup.find('body')
                if body:
                    content = body.get_text().strip()
            
            if not content or len(content) < 10:
                raise WebScrapingError("无法提取有效的网页内容")
            
            return {
                'title': title,
                'content': content,
                'authors': [],
                'publish_date': None,
                'source_url': url
            }
            
    except Exception as e:
        logger.error(f"{method}提取失败: {e}")
        raise WebScrapingError(f"{method}内容提取失败: {e}")


async def process_upload(file: UploadFile) -> str:
    """
    处理上传文件并返回纯文本内容
    
    Args:
        file: 上传的文件对象
        
    Returns:
        提取的纯文本内容
        
    Raises:
        FileSizeExceededError: 文件大小超限
        UnsupportedFileTypeError: 不支持的文件类型
        FileProcessingError: 文件处理失败
    """
    start_time = time.time()
    logger.info(f"开始处理文件: {file.filename}, 大小: {file.size} bytes")
    
    temp_path = None
    try:
        # 1. 验证文件大小
        if file.size and file.size > MAX_FILE_SIZE:
            raise FileSizeExceededError(f"文件大小超过限制 ({MAX_FILE_SIZE / 1024 / 1024:.1f}MB)")
        
        # 2. 读取文件内容
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise FileSizeExceededError(f"文件大小超过限制 ({MAX_FILE_SIZE / 1024 / 1024:.1f}MB)")
        
        # 3. 检测文件类型
        real_mime_type = detect_file_type(file_content)
        logger.info(f"检测到文件类型: {real_mime_type}")
        
        if real_mime_type not in SUPPORTED_MIME_TYPES:
            raise UnsupportedFileTypeError(f"不支持的文件类型: {real_mime_type}")
        
        # 4. 创建临时文件
        file_ext = SUPPORTED_MIME_TYPES[real_mime_type]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_path = temp_file.name
            temp_file.write(file_content)
            temp_file.flush()
        
        # 5. 提取文本
        text = await extract_text_from_file(temp_path, real_mime_type)
        
        # 6. 记录结果
        processing_time = time.time() - start_time
        word_count = len(text.split())
        char_count = len(text)
        logger.info(f"文件处理完成 - 类型: {real_mime_type}, 字符数: {char_count}, 词数: {word_count}, 耗时: {processing_time:.2f}秒")
        
        return text
        
    except (FileSizeExceededError, UnsupportedFileTypeError, FileProcessingError):
        raise
    except Exception as e:
        logger.error(f"文件处理失败: {e}", exc_info=True)
        raise FileProcessingError(f"文件处理失败: {str(e)}")
    
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")


async def extract_web_content(url: str) -> Dict:
    """
    抓取网页内容并返回结构化数据
    
    Args:
        url: 网页URL
        
    Returns:
        包含网页信息的字典
        
    Raises:
        InvalidURLError: 无效URL
        WebScrapingError: 抓取失败
    """
    start_time = time.time()
    logger.info(f"开始抓取网页内容: {url}")
    
    # 1. 验证URL
    validate_url(url)
    
    # 2. 尝试多种方法抓取
    methods = ['playwright', 'newspaper', 'beautifulsoup']
    last_error = None
    
    for attempt in range(3):  # 最多3次尝试
        for method in methods:
            try:
                logger.info(f"尝试使用{method}抓取 (第{attempt + 1}次)")
                result = await extract_web_content_with_method(url, method)
                
                # 记录成功结果
                processing_time = time.time() - start_time
                word_count = len(result['content'].split())
                char_count = len(result['content'])
                logger.info(f"{method}抓取成功: {result['title']}, 字符数: {char_count}, 词数: {word_count}, 耗时: {processing_time:.2f}秒")
                
                return result
                
            except WebScrapingError as e:
                logger.warning(f"{method}抓取失败: {e}")
                last_error = e
                continue
        
        # 如果所有方法都失败，等待后重试
        if attempt < 2:
            wait_time = (attempt + 1) * 2
            logger.info(f"所有方法失败，等待{wait_time}秒后重试...")
            await asyncio.sleep(wait_time)
    
    # 3. 所有尝试都失败
    total_time = time.time() - start_time
    logger.error(f"经过3次尝试后仍无法抓取网页内容，总耗时: {total_time:.2f}秒")
    raise WebScrapingError(f"经过3次尝试后仍无法抓取网页内容: {last_error}")


# 导出主要函数
__all__ = [
    'process_upload',
    'extract_web_content',
    'WebContentResult',
    'UnsupportedFileTypeError',
    'FileSizeExceededError',
    'FileProcessingError',
    'InvalidURLError',
    'WebScrapingError'
]
