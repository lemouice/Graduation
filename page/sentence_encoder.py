#!/usr/bin/env python3
"""
句子编码器模块 v2.0 (精简版)

基于sentence-transformers库实现多语言文本编码和相似度计算功能。
使用单例模式确保模型只加载一次，支持离线模式、自动设备选择、性能监控等高级功能。
"""

import asyncio
import logging
import os
import re
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """模型加载失败异常"""
    pass


class EncodingError(Exception):
    """编码处理异常"""
    pass


class SentenceEncoder:
    """
    句子编码器类 (单例模式)
    
    支持单例模式、自动设备选择、离线模式、性能监控等高级功能。
    """
    
    _instance: Optional['SentenceEncoder'] = None
    _initialized: bool = False
    
    # 模型配置
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    FALLBACK_MODEL = "all-MiniLM-L6-v2"
    MAX_WORDS_PER_CHUNK = 256
    DEFAULT_BATCH_SIZE = 32
    
    def __new__(cls, *args, **kwargs):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(SentenceEncoder, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        offline_mode: bool = False
    ):
        """初始化句子编码器 (单例模式)"""
        if self._initialized:
            return
        
        # 从环境变量获取配置
        self.model_name = model_name or os.getenv('MODEL_NAME', self.DEFAULT_MODEL)
        self.offline_mode = offline_mode or os.getenv('OFFLINE_MODE', 'false').lower() == 'true'
        self.batch_size = batch_size
        
        # 设备选择 (MPS > CUDA > CPU)
        self.device = self._select_device(device)
        
        # 模型和线程池
        self.model: Optional[SentenceTransformer] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 性能统计
        self.stats = {
            'total_encodings': 0,
            'total_batches': 0,
            'total_similarities': 0,
            'total_time': 0.0,
            'peak_memory': 0.0
        }
        self.start_time = time.time()
        
        self._initialized = True
        logger.info(f"初始化SentenceEncoder - 模型: {self.model_name}, 设备: {self.device}")
    
    def _select_device(self, device: Optional[str]) -> str:
        """自动选择计算设备 (MPS > CUDA > CPU)"""
        if device and device in ['mps', 'cuda', 'cpu']:
            return device
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    
    async def load_model(self) -> None:
        """异步加载模型 (支持降级机制)"""
        if self.model is not None:
            return
        
        try:
            logger.info(f"开始加载主模型: {self.model_name}")
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                self.model_name
            )
            
            load_time = time.time() - start_time
            logger.info(f"主模型加载完成 - 耗时: {load_time:.2f}秒")
            
        except Exception as e:
            logger.warning(f"主模型加载失败: {e}")
            logger.info(f"尝试加载备用模型: {self.FALLBACK_MODEL}")
            
            try:
                self.model = await loop.run_in_executor(
                    self.executor,
                    self._load_model_sync,
                    self.FALLBACK_MODEL
                )
                logger.info("备用模型加载完成")
            except Exception as fallback_error:
                raise ModelLoadError(f"无法加载任何模型: {str(fallback_error)}")
    
    def _load_model_sync(self, model_name: str) -> SentenceTransformer:
        """同步加载模型"""
        if self.offline_mode:
            logger.info(f"离线模式: 从本地缓存加载模型 {model_name}")
        
        model = SentenceTransformer(model_name, device=self.device)
        model.max_seq_length = 512
        return model
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量 (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def _adjust_batch_size(self, current_batch_size: int) -> int:
        """根据内存使用情况调整批处理大小"""
        memory_usage = self._get_memory_usage()
        if memory_usage > 2000:  # 超过2GB
            new_batch_size = max(1, current_batch_size // 2)
            logger.warning(f"内存使用过高 ({memory_usage:.1f}MB)，调整批处理大小: {current_batch_size} -> {new_batch_size}")
            return new_batch_size
        return current_batch_size
    
    @lru_cache(maxsize=1000)
    def _preprocess_text(self, text: str) -> str:
        """预处理文本 (带缓存)"""
        if not text or not text.strip():
            return ""
        
        # 去除多余空白字符和特殊字符
        text = " ".join(text.split())
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()（）【】""''""''，。！？；：]', ' ', text)
        return " ".join(text.split())
    
    def _split_text_by_words(self, text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
        """按词数分割文本"""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        chunks = []
        for i in range(0, len(words), max_words):
            chunks.append(" ".join(words[i:i + max_words]))
        
        logger.info(f"文本分块: {len(words)} 词 -> {len(chunks)} 块")
        return chunks
    
    async def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """单文本编码 (支持长文本分段)"""
        if self.model is None:
            raise ModelLoadError("模型未加载，请先调用 load_model()")
        
        if not text or not text.strip():
            logger.warning("输入文本为空，返回零向量")
            return np.zeros(384, dtype=np.float32)
        
        try:
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            # 预处理文本
            processed_text = self._preprocess_text(text)
            
            # 检查是否需要分块 (按词数)
            word_count = len(processed_text.split())
            if word_count > self.MAX_WORDS_PER_CHUNK:
                chunks = self._split_text_by_words(processed_text, self.MAX_WORDS_PER_CHUNK)
                logger.info(f"长文本分段处理: {word_count} 词 -> {len(chunks)} 段")
                
                # 编码所有段
                loop = asyncio.get_event_loop()
                chunk_embeddings = []
                
                for chunk in chunks:
                    embedding = await loop.run_in_executor(
                        self.executor,
                        self._encode_sync,
                        chunk,
                        normalize
                    )
                    chunk_embeddings.append(embedding)
                
                # 平均池化
                final_embedding = np.mean(chunk_embeddings, axis=0)
            else:
                # 直接编码
                loop = asyncio.get_event_loop()
                final_embedding = await loop.run_in_executor(
                    self.executor,
                    self._encode_sync,
                    processed_text,
                    normalize
                )
            
            # 记录性能
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            self.stats['total_encodings'] += 1
            self.stats['total_time'] += processing_time
            self.stats['peak_memory'] = max(self.stats['peak_memory'], memory_after)
            
            logger.info(f"单文本编码完成 - 词数: {word_count}, 耗时: {processing_time:.3f}秒, 内存: {memory_delta:+.1f}MB")
            return final_embedding
            
        except Exception as e:
            logger.error(f"单文本编码失败: {e}", exc_info=True)
            raise EncodingError(f"文本编码失败: {str(e)}")
    
    def _encode_sync(self, text: str, normalize: bool = True) -> np.ndarray:
        """同步编码文本"""
        embedding = self.model.encode(text, normalize_embeddings=normalize)
        return embedding.astype(np.float32)
    
    async def encode_batch(
        self, 
        texts: List[str], 
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """批量文本编码 (支持自动批处理大小调整)"""
        if self.model is None:
            raise ModelLoadError("模型未加载，请先调用 load_model()")
        
        if not texts:
            return np.array([])
        
        try:
            start_time = time.time()
            memory_before = self._get_memory_usage()
            
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            valid_texts = [text for text in processed_texts if text.strip()]
            
            if not valid_texts:
                return np.array([])
            
            # 自动调整批处理大小
            current_batch_size = batch_size or self.batch_size
            current_batch_size = self._adjust_batch_size(current_batch_size)
            
            # 批量编码
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor,
                self._encode_batch_sync,
                valid_texts,
                normalize,
                show_progress,
                current_batch_size
            )
            
            # 记录性能
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            self.stats['total_batches'] += 1
            self.stats['total_encodings'] += len(valid_texts)
            self.stats['total_time'] += processing_time
            
            logger.info(f"批量编码完成 - 文本数: {len(valid_texts)}, 批大小: {current_batch_size}, 耗时: {processing_time:.3f}秒, 内存: {memory_delta:+.1f}MB")
            return embeddings
            
        except Exception as e:
            logger.error(f"批量编码失败: {e}", exc_info=True)
            raise EncodingError(f"批量编码失败: {str(e)}")
    
    def _encode_batch_sync(
        self, 
        texts: List[str], 
        normalize: bool = True,
        show_progress: bool = False,
        batch_size: int = None
    ) -> np.ndarray:
        """同步批量编码"""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    async def similarity(
        self, 
        text1: str, 
        text2: str, 
        method: str = "cosine"
    ) -> float:
        """计算两个文本的相似度 (支持多语言混合文本)"""
        if self.model is None:
            raise ModelLoadError("模型未加载，请先调用 load_model()")
        
        if not text1 or not text2:
            return 0.0
        
        try:
            start_time = time.time()
            
            # 编码两个文本
            embedding1 = await self.encode(text1, normalize=True)
            embedding2 = await self.encode(text2, normalize=True)
            
            # 计算相似度
            if method == "cosine":
                similarity_score = float(cos_sim(embedding1, embedding2).item())
            elif method == "dot":
                similarity_score = float(np.dot(embedding1, embedding2))
            else:
                raise ValueError(f"不支持的相似度计算方法: {method}")
            
            # 记录性能
            processing_time = time.time() - start_time
            self.stats['total_similarities'] += 1
            self.stats['total_time'] += processing_time
            
            logger.info(f"相似度计算完成 - 方法: {method}, 分数: {similarity_score:.4f}, 耗时: {processing_time:.3f}秒")
            return similarity_score
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}", exc_info=True)
            raise EncodingError(f"相似度计算失败: {str(e)}")
    
    async def similarity_batch(
        self, 
        query_text: str, 
        candidate_texts: List[str],
        method: str = "cosine"
    ) -> List[float]:
        """计算查询文本与候选文本列表的相似度"""
        if not candidate_texts:
            return []
        
        try:
            # 编码查询文本
            query_embedding = await self.encode(query_text, normalize=True)
            
            # 批量编码候选文本
            candidate_embeddings = await self.encode_batch(candidate_texts, normalize=True)
            
            # 计算相似度
            similarities = []
            for candidate_embedding in candidate_embeddings:
                if method == "cosine":
                    sim = float(cos_sim(query_embedding, candidate_embedding).item())
                elif method == "dot":
                    sim = float(np.dot(query_embedding, candidate_embedding))
                else:
                    raise ValueError(f"不支持的相似度计算方法: {method}")
                similarities.append(sim)
            
            logger.info(f"批量相似度计算完成 - 候选数: {len(candidate_texts)}")
            return similarities
            
        except Exception as e:
            logger.error(f"批量相似度计算失败: {e}", exc_info=True)
            raise EncodingError(f"批量相似度计算失败: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = self.stats.copy()
        stats['uptime'] = time.time() - self.start_time
        if stats['total_encodings'] > 0:
            stats['avg_encoding_time'] = stats['total_time'] / stats['total_encodings']
            stats['throughput'] = stats['total_encodings'] / stats['uptime']
        else:
            stats['avg_encoding_time'] = 0.0
            stats['throughput'] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.stats = {
            'total_encodings': 0,
            'total_batches': 0,
            'total_similarities': 0,
            'total_time': 0.0,
            'peak_memory': 0.0
        }
        self.start_time = time.time()
        logger.info("性能统计已重置")
    
    async def close(self) -> None:
        """关闭编码器，释放资源"""
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("SentenceEncoder已关闭")


# 全局编码器实例
_global_encoder: Optional[SentenceEncoder] = None


async def get_encoder() -> SentenceEncoder:
    """获取全局编码器实例"""
    global _global_encoder
    
    if _global_encoder is None:
        _global_encoder = SentenceEncoder()
        await _global_encoder.load_model()
    
    return _global_encoder


async def encode(text: str, normalize: bool = True) -> np.ndarray:
    """便捷的单文本编码函数"""
    encoder = await get_encoder()
    return await encoder.encode(text, normalize)


async def encode_batch(
    texts: List[str], 
    normalize: bool = True,
    show_progress: bool = False
) -> np.ndarray:
    """便捷的批量编码函数"""
    encoder = await get_encoder()
    return await encoder.encode_batch(texts, normalize, show_progress)


async def similarity(text1: str, text2: str, method: str = "cosine") -> float:
    """便捷的相似度计算函数"""
    encoder = await get_encoder()
    return await encoder.similarity(text1, text2, method)


