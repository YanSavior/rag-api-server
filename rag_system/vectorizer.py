from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
import numpy as np
from typing import List, Dict, Tuple
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleVectorizer:
    """简单的TF-IDF向量化器，用于离线模式"""
    
    def __init__(self):
        """初始化TF-IDF向量化器"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        self.is_fitted = False
        print("使用TF-IDF向量化器（离线模式）")
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """拟合并转换文本"""
        embeddings = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return embeddings.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """转换新文本"""
        if not self.is_fitted:
            raise ValueError("向量化器尚未拟合，请先调用fit_transform")
        embeddings = self.vectorizer.transform(texts)
        return embeddings.toarray()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量编码文本为向量"""
        if not self.is_fitted:
            return self.fit_transform(texts)
        return self.transform(texts)
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.transform([text])[0]

class BGEVectorizer:
    """BGE模型向量化器，使用Sentence Transformers接口"""
    
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5", force_bge=False):
        """
        初始化BGE向量化器
        
        Args:
            model_name: BGE模型名称
            force_bge: 是否强制使用BGE模型（如果失败会抛出异常）
        """
        try:
            print(f"🔄 正在加载BGE模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            print(f"✅ BGE模型加载完成: {model_name}")
            self.use_simple = False
        except Exception as e:
            print(f"❌ 无法加载BGE模型: {e}")
            if force_bge:
                raise Exception(f"强制BGE模式失败: {e}")
            print("🔄 切换到TF-IDF向量化器（离线模式）")
            self.model = SimpleVectorizer()
            self.use_simple = True
        
        # 检查GPU可用性
        if not self.use_simple:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(self.device)
            print(f"🖥️ 使用设备: {self.device}")
    
    def get_model_info(self):
        """获取模型信息"""
        if self.use_simple:
            return {
                "type": "TF-IDF",
                "mode": "offline",
                "description": "使用TF-IDF向量化器，无需网络连接"
            }
        else:
            return {
                "type": "BGE",
                "mode": "online",
                "description": f"使用BGE模型: {self.model.model_name if hasattr(self.model, 'model_name') else 'Unknown'}"
            }
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组
        """
        if self.use_simple:
            return self.model.encode_texts(texts, batch_size)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embeddings.cpu().numpy()
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            向量
        """
        if self.use_simple:
            return self.model.encode_single(text)
        
        embedding = self.model.encode(
            [text],
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        return embedding[0].cpu().numpy()

class SimpleReranker:
    """简单的重排器，使用TF-IDF相似度"""
    
    def __init__(self):
        """初始化简单重排器"""
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
        print("使用TF-IDF重排器（离线模式）")
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        对文档进行重排
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前k个结果
            
        Returns:
            重排后的结果列表
        """
        # 准备所有文本
        all_texts = [query] + documents
        
        # 向量化
        embeddings = self.vectorizer.fit_transform(all_texts)
        
        # 计算相似度
        query_embedding = embeddings[0:1]
        doc_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # 创建结果列表
        results = []
        for i, (doc, score) in enumerate(zip(documents, similarities)):
            results.append({
                'document': doc,
                'score': float(score),
                'index': i
            })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

class CrossEncoderReranker:
    """Cross-Encoder重排器"""
    
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        """初始化Cross-Encoder重排器"""
        try:
            print(f"🔄 正在加载Cross-Encoder模型: {model_name}")
            self.model = CrossEncoder(model_name)
            print(f"✅ Cross-Encoder模型加载完成: {model_name}")
            self.use_simple = False
        except Exception as e:
            print(f"❌ 无法加载Cross-Encoder模型: {e}")
            print("🔄 切换到TF-IDF重排器（离线模式）")
            self.model = SimpleReranker()
            self.use_simple = True
    
    def get_model_info(self):
        """获取模型信息"""
        if self.use_simple:
            return {
                "type": "TF-IDF",
                "mode": "offline",
                "description": "使用TF-IDF重排器，无需网络连接"
            }
        else:
            return {
                "type": "Cross-Encoder",
                "mode": "online",
                "description": f"使用Cross-Encoder模型: {self.model.model_name if hasattr(self.model, 'model_name') else 'Unknown'}"
            }
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict]:
        """
        对文档进行重排
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前k个结果
            
        Returns:
            重排后的结果列表
        """
        if self.use_simple:
            return self.model.rerank(query, documents, top_k)
        
        # 准备查询-文档对
        query_doc_pairs = [(query, doc) for doc in documents]
        
        # 计算分数
        scores = self.model.predict(query_doc_pairs)
        
        # 创建结果列表
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                'document': doc,
                'score': float(score),
                'index': i
            })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

class AdvancedRAGVectorizer:
    """高级RAG向量化器，集成Bi-Encoder和Cross-Encoder"""
    
    def __init__(self, 
                 bi_encoder_model="BAAI/bge-small-zh-v1.5",
                 cross_encoder_model="BAAI/bge-reranker-v2-m3"):
        """
        初始化高级RAG向量化器
        
        Args:
            bi_encoder_model: Bi-Encoder模型名称
            cross_encoder_model: Cross-Encoder模型名称
        """
        self.bi_encoder = BGEVectorizer(bi_encoder_model)
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        print("高级RAG向量化器初始化完成")
    
    def retrieve_and_rerank(self, 
                           query: str, 
                           documents: List[str], 
                           top_k_retrieve: int = 20, 
                           top_k_final: int = 5) -> List[Dict]:
        """
        检索-重排流程
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k_retrieve: 粗检索候选数量
            top_k_final: 最终结果数量
            
        Returns:
            最终结果列表
        """
        print(f"开始检索-重排流程...")
        print(f"查询: {query}")
        print(f"文档总数: {len(documents)}")
        
        # 第一步：Bi-Encoder粗检索
        print("第一步：Bi-Encoder粗检索...")
        candidate_docs = self._bi_encoder_retrieve(query, documents, top_k_retrieve)
        
        # 第二步：Cross-Encoder精排
        print("第二步：Cross-Encoder精排...")
        final_results = self._cross_encoder_rerank(query, candidate_docs, top_k_final)
        
        print(f"检索-重排完成，返回 {len(final_results)} 个结果")
        return final_results
    
    def _bi_encoder_retrieve(self, query: str, documents: List[str], top_k: int) -> List[Dict]:
        """Bi-Encoder粗检索"""
        
        # 编码查询和文档
        query_embedding = self.bi_encoder.model.encode(query, convert_to_tensor=True)
        doc_embeddings = self.bi_encoder.model.encode(documents, convert_to_tensor=True)
        
        # 计算余弦相似度
        similarities = torch.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings)
        
        # 获取top_k候选
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        candidates = []
        for idx in top_indices:
            candidates.append({
                'document': documents[idx],
                'bi_encoder_score': similarities[idx].item(),
                'index': idx.item()
            })
        
        print(f"粗检索完成，获得 {len(candidates)} 个候选文档")
        return candidates
    
    def _cross_encoder_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Cross-Encoder精排"""
        
        # 提取文档内容
        documents = [candidate['document'] for candidate in candidates]
        
        # 使用Cross-Encoder重排
        reranked_results = self.cross_encoder.rerank(query, documents, top_k)
        
        # 合并结果
        final_results = []
        for i, reranked in enumerate(reranked_results):
            # 找到对应的原始候选
            original_candidate = candidates[reranked['index']]
            
            final_results.append({
                'document': reranked['document'],
                'bi_encoder_score': original_candidate['bi_encoder_score'],
                'cross_encoder_score': reranked['score'],
                'index': reranked['index']
            })
        
        print(f"精排完成，返回 {len(final_results)} 个最终结果")
        return final_results 