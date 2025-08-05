import chromadb
from chromadb.config import Settings
import os
import numpy as np
from typing import List, Dict, Optional
import json

class ChromaVectorDB:
    """ChromaDB向量数据库操作类"""
    
    def __init__(self, persist_directory="./vector_db"):
        """
        初始化ChromaDB向量数据库
        
        Args:
            persist_directory: 数据持久化目录
        """
        self.persist_directory = persist_directory
        
        # 使用新的ChromaDB客户端配置
        try:
            # 尝试使用新的配置
            self.client = chromadb.PersistentClient(path=persist_directory)
            print(f"ChromaDB初始化完成（新配置），数据目录: {persist_directory}")
        except Exception as e:
            print(f"新配置失败，尝试旧配置: {e}")
            # 回退到旧配置
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
            print(f"ChromaDB初始化完成（旧配置），数据目录: {persist_directory}")
        
        self.collection = None
    
    def create_collection(self, name: str = "student_knowledge") -> None:
        """
        创建向量集合
        
        Args:
            name: 集合名称
        """
        try:
            self.collection = self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"向量集合创建成功: {name}")
        except Exception as e:
            print(f"集合已存在或创建失败: {e}")
            # 尝试获取已存在的集合
            self.collection = self.client.get_collection(name=name)
            print(f"获取已存在的集合: {name}")
    
    def get_or_create_collection(self, name: str = "student_knowledge"):
        """获取或创建集合"""
        try:
            self.collection = self.client.get_collection(name=name)
            print(f"获取已存在的集合: {name}")
        except:
            self.create_collection(name)
    
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: np.ndarray, 
                     metadatas: Optional[List[Dict]] = None,
                     ids: Optional[List[str]] = None) -> None:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档内容列表
            embeddings: 向量数组
            metadatas: 元数据列表
            ids: 文档ID列表
        """
        if not self.collection:
            self.create_collection()
        
        # 生成文档ID
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{"type": "document"} for _ in documents]
        
        # 确保向量格式正确
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings
        
        # 添加文档
        self.collection.add(
            documents=documents,
            embeddings=embeddings_list,
            ids=ids,
            metadatas=metadatas
        )
        print(f"成功添加 {len(documents)} 个文档到向量数据库")
    
    def search(self, 
              query_embedding: np.ndarray, 
              n_results: int = 5,
              where: Optional[Dict] = None) -> Dict:
        """
        搜索相似文档
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            where: 过滤条件
            
        Returns:
            搜索结果字典
        """
        if not self.collection:
            raise ValueError("向量集合未初始化")
        
        # 确保向量格式正确
        if isinstance(query_embedding, np.ndarray):
            query_embedding_list = query_embedding.tolist()
        else:
            query_embedding_list = query_embedding
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def search_by_text(self, 
                      query_text: str, 
                      vectorizer,
                      n_results: int = 5,
                      where: Optional[Dict] = None) -> Dict:
        """
        通过文本搜索
        
        Args:
            query_text: 查询文本
            vectorizer: 向量化器
            n_results: 返回结果数量
            where: 过滤条件
            
        Returns:
            搜索结果字典
        """
        # 编码查询文本
        query_embedding = vectorizer.encode_single(query_text)
        
        # 执行搜索
        return self.search(query_embedding, n_results, where)
    
    def get_collection_info(self) -> Dict:
        """获取集合信息"""
        if not self.collection:
            return {"error": "集合未初始化"}
        
        count = self.collection.count()
        return {
            "name": self.collection.name,
            "document_count": count,
            "metadata": self.collection.metadata
        }
    
    def delete_collection(self, name: str) -> None:
        """删除集合"""
        try:
            self.client.delete_collection(name=name)
            print(f"集合删除成功: {name}")
        except Exception as e:
            print(f"删除集合失败: {e}")
    
    def list_collections(self) -> List[str]:
        """列出所有集合"""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def update_document(self, 
                       id: str, 
                       document: str, 
                       embedding: np.ndarray, 
                       metadata: Optional[Dict] = None) -> None:
        """
        更新文档
        
        Args:
            id: 文档ID
            document: 新文档内容
            embedding: 新向量
            metadata: 新元数据
        """
        if not self.collection:
            raise ValueError("向量集合未初始化")
        
        self.collection.update(
            ids=[id],
            documents=[document],
            embeddings=[embedding.tolist()],
            metadatas=[metadata] if metadata else None
        )
        print(f"文档更新成功: {id}")
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        删除文档
        
        Args:
            ids: 要删除的文档ID列表
        """
        if not self.collection:
            raise ValueError("向量集合未初始化")
        
        self.collection.delete(ids=ids)
        print(f"成功删除 {len(ids)} 个文档")
    
    def export_collection(self, export_path: str) -> None:
        """
        导出集合数据
        
        Args:
            export_path: 导出文件路径
        """
        if not self.collection:
            raise ValueError("向量集合未初始化")
        
        # 获取所有数据
        all_data = self.collection.get()
        
        # 保存到文件
        export_data = {
            "documents": all_data['documents'],
            "embeddings": all_data['embeddings'],
            "metadatas": all_data['metadatas'],
            "ids": all_data['ids']
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"集合数据导出成功: {export_path}")
    
    def import_collection(self, import_path: str) -> None:
        """
        导入集合数据
        
        Args:
            import_path: 导入文件路径
        """
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        self.add_documents(
            documents=import_data['documents'],
            embeddings=import_data['embeddings'],
            metadatas=import_data['metadatas'],
            ids=import_data['ids']
        )
        
        print(f"集合数据导入成功: {import_path}") 