#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统API服务器 - Vercel版本
适用于无服务器环境
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import RAGSystem

app = FastAPI(title="RAG系统API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储RAG系统实例
rag_system = None

class QueryRequest(BaseModel):
    question: str
    top_k_retrieve: Optional[int] = 20
    top_k_final: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    relevant_docs: List[str]
    scores: List[float]

def get_rag_system():
    """获取或初始化RAG系统"""
    global rag_system
    if rag_system is None:
        try:
            print("初始化RAG系统...")
            rag_system = RAGSystem()
            print("RAG系统初始化完成")
        except Exception as e:
            print(f"RAG系统初始化失败: {e}")
            return None
    return rag_system

@app.get("/")
async def root():
    return {"message": "RAG系统API服务", "status": "running"}

@app.get("/api/health")
async def health_check():
    rag = get_rag_system()
    return {
        "status": "healthy",
        "rag_system_initialized": rag is not None
    }

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        rag = get_rag_system()
        if not rag:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        result = rag.query(
            question=request.question,
            top_k_retrieve=request.top_k_retrieve,
            top_k_final=request.top_k_final
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/api/build_knowledge_base")
async def build_knowledge_base():
    try:
        rag = get_rag_system()
        if not rag:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        rag.build_knowledge_base(
            student_csv="data/students.csv",
            plan_txt="data/cultivation_plan.txt"
        )
        
        return {"message": "知识库构建成功"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")

# Vercel需要这个函数
def handler(request, context):
    """Vercel函数处理器"""
    return app(request, context)

# 本地开发时使用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 