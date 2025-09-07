import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Normalize

class VectorSolver:
    """
    句子向量处理工具类，提供文本向量化和相似度计算功能
    
    示例:
        >>> solver = VectorSolver()
        >>> vector = solver.get_vector("这是一个示例文本")
        >>> similarity = solver.compare_vectors(vector1, vector2)
    """
    
    # 预定义的模型列表
    PREDEFINED_MODELS = {
        'miniLM': 'paraphrase-multilingual-MiniLM-L12-v2',
        'mpnet': 'all-mpnet-base-v2',
        'miniLM-en': 'paraphrase-MiniLM-L6-v2',
        'default': 'paraphrase-multilingual-MiniLM-L12-v2'
    }
    
    def __init__(self, model_name='default'):
        """
        初始化向量处理器
        
        参数:
            model_name: 字符串，可以是预定义模型键名或HuggingFace模型路径
                       可选值: 'miniLM', 'mpnet', 'miniLM-en', 'default' 或自定义模型路径
        """
        # 获取实际模型名称
        actual_model_name = self.PREDEFINED_MODELS.get(
            model_name, 
            model_name  # 如果不是预定义键，则直接使用传入的值
        )
        
        try:
            # 创建带归一化层的模型
            base_model = SentenceTransformer(actual_model_name)
            normalized_model = Normalize()
            modules = [base_model, normalized_model]
            self.model = SentenceTransformer(modules=modules)
            
            print(f"✅ 成功加载模型: {actual_model_name}")
            self.model_name = actual_model_name
            
        except Exception as e:
            raise ValueError(f"无法加载模型 '{model_name}': {str(e)}")
    
    def get_vector(self, text):
        """
        将文本转换为向量表示，支持长文本处理
        
        参数:
            text: 字符串，要转换的文本（可以是长文本）
            
        返回:
            list: 文本的向量表示
            
        异常:
            ValueError: 当输入不是有效字符串时抛出
        """
        # 输入验证
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串类型")
        if not text.strip():
            raise ValueError("输入文本不能为空")
        
        try:
            # 处理文本并生成向量
            cleaned_text = text.strip()
            # 使用模型编码文本
            vector = self.model.encode(cleaned_text, convert_to_tensor=False)
            return vector.tolist()
            
        except Exception as e:
            raise RuntimeError(f"文本向量化过程中出错: {str(e)}")
    
    def compare_vectors(self, vec1, vec2):
            """
            计算两个向量的余弦相似度
            
            参数:
                vec1: 第一个向量（列表或数组）
                vec2: 第二个向量（列表或数组）
                
            返回:
                float: 相似度得分，范围[0, 1]，保留5位小数
            """
            try:
                # 转换为numpy数组
                vec1_np = np.array(vec1)
                vec2_np = np.array(vec2)
                
                # 计算余弦相似度
                dot_product = np.dot(vec1_np, vec2_np)
                norm_vec1 = np.linalg.norm(vec1_np)
                norm_vec2 = np.linalg.norm(vec2_np)
                
                # 避免除以零
                if norm_vec1 == 0 or norm_vec2 == 0:
                    return 0.0
                    
                similarity = dot_product / (norm_vec1 * norm_vec2)
                
                # 确保结果在[0,1]范围内并保留5位小数
                similarity = max(0.0, min(1.0, similarity))
                return round(similarity, 5)
                
            except Exception as e:
                raise RuntimeError(f"相似度计算过程中出错: {str(e)}")
    
    def get_model_info(self):
        """
        获取当前模型信息
        
        返回:
            dict: 包含模型名称和维度信息的字典
        """
        # 获取模型维度信息
        test_vector = self.get_vector("测试文本")
        return {
            'model_name': self.model_name,
            'vector_dimension': len(test_vector)
        }