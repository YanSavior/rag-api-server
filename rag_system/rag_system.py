from .vectorizer import AdvancedRAGVectorizer
from .vector_db import ChromaVectorDB
from .data_processor import DataProcessor
import os
from typing import List, Dict, Optional

class RAGSystem:
    """完整的RAG系统"""
    
    def __init__(self, 
                 bi_encoder_model="BAAI/bge-small-zh-v1.5",
                 cross_encoder_model="BAAI/bge-reranker-v2-m3",
                 persist_directory="./vector_db"):
        """
        初始化RAG系统
        
        Args:
            bi_encoder_model: Bi-Encoder模型名称
            cross_encoder_model: Cross-Encoder模型名称
            persist_directory: 向量数据库存储目录
        """
        self.vectorizer = AdvancedRAGVectorizer(bi_encoder_model, cross_encoder_model)
        self.vector_db = ChromaVectorDB(persist_directory)
        self.data_processor = DataProcessor()
        print("RAG系统初始化完成")
    
    def build_knowledge_base(self, 
                           student_csv: str = None, 
                           plan_txt: str = None,
                           collection_name: str = "student_knowledge") -> None:
        """
        构建知识库
        
        Args:
            student_csv: 学生数据CSV文件路径
            plan_txt: 培养方案文本文件路径
            collection_name: 集合名称
        """
        print("开始构建知识库...")
        
        # 处理数据
        documents, metadatas = self.data_processor.process_mixed_data(student_csv, plan_txt)
        
        if not documents:
            print("没有找到有效数据")
            return
        
        # 向量化文档
        print("开始向量化文档...")
        embeddings = self.vectorizer.bi_encoder.encode_texts(documents)
        
        # 存储到向量数据库
        print("存储到向量数据库...")
        self.vector_db.get_or_create_collection(collection_name)
        self.vector_db.add_documents(documents, embeddings, metadatas)
        
        print(f"知识库构建完成，共 {len(documents)} 个文档")
    
    def query(self, 
              question: str, 
              top_k_retrieve: int = 20, 
              top_k_final: int = 5,
              collection_name: str = "student_knowledge") -> Dict:
        """
        执行RAG查询
        
        Args:
            question: 用户问题
            top_k_retrieve: 粗检索候选数量
            top_k_final: 最终结果数量
            collection_name: 集合名称
            
        Returns:
            查询结果字典
        """
        print(f"执行RAG查询: {question}")
        
        # 确保集合存在
        self.vector_db.get_or_create_collection(collection_name)
        
        # 第一步：从向量数据库检索候选文档
        print("第一步：向量数据库检索...")
        search_results = self.vector_db.search_by_text(
            question, 
            self.vectorizer.bi_encoder, 
            top_k_retrieve
        )
        
        if not search_results['documents']:
            return {
                "question": question,
                "answer": "抱歉，没有找到相关信息。",
                "relevant_docs": [],
                "scores": []
            }
        
        candidate_docs = search_results['documents'][0]
        
        # 第二步：Cross-Encoder重排
        print("第二步：Cross-Encoder重排...")
        final_results = self.vectorizer.cross_encoder.rerank(
            question, 
            candidate_docs, 
            top_k_final
        )
        
        # 构建上下文
        context = "\n".join([result['document'] for result in final_results])
        
        # 生成回答（这里可以集成LLM）
        answer = self.generate_answer(question, context)
        
        return {
            "question": question,
            "answer": answer,
            "relevant_docs": [result['document'] for result in final_results],
            "scores": [result['score'] for result in final_results],
            "context": context
        }
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        生成回答（简化版本，可以集成LLM）
        
        Args:
            question: 用户问题
            context: 相关上下文
            
        Returns:
            生成的回答
        """
        # 这里可以集成OpenAI、Claude等LLM
        # 目前返回简化版本
        return f"基于检索到的信息，我可以为您提供以下回答：\n\n{context}\n\n这些信息应该能够回答您的问题：{question}"
    
    def get_knowledge_base_info(self, collection_name: str = "student_knowledge") -> Dict:
        """获取知识库信息"""
        self.vector_db.get_or_create_collection(collection_name)
        return self.vector_db.get_collection_info()
    
    def search_similar_documents(self, 
                                query: str, 
                                n_results: int = 5,
                                collection_name: str = "student_knowledge") -> Dict:
        """搜索相似文档"""
        self.vector_db.get_or_create_collection(collection_name)
        return self.vector_db.search_by_text(
            query, 
            self.vectorizer.bi_encoder, 
            n_results
        )
    
    def export_knowledge_base(self, 
                             export_path: str, 
                             collection_name: str = "student_knowledge") -> None:
        """导出知识库"""
        self.vector_db.get_or_create_collection(collection_name)
        self.vector_db.export_collection(export_path)
    
    def import_knowledge_base(self, import_path: str) -> None:
        """导入知识库"""
        self.vector_db.import_collection(import_path)
    
    def delete_knowledge_base(self, collection_name: str) -> None:
        """删除知识库"""
        self.vector_db.delete_collection(collection_name)
    
    def list_knowledge_bases(self) -> List[str]:
        """列出所有知识库"""
        return self.vector_db.list_collections() 