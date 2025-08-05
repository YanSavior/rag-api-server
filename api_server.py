#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统API服务器
提供RESTful API接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os

from rag_system import RAGSystem

app = FastAPI(title="RAG系统API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.on_event("startup")
async def startup_event():
    global rag_system
    print("初始化RAG系统...")
    rag_system = RAGSystem()
    print("RAG系统初始化完成")

@app.get("/")
async def root():
    return {"message": "RAG系统API服务", "status": "running"}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        result = rag_system.query(
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
        if not rag_system:
            raise HTTPException(status_code=500, detail="RAG系统未初始化")
        
        rag_system.build_knowledge_base(
            student_csv="data/students.csv",
            plan_txt="data/cultivation_plan.txt"
        )
        
        return {"message": "知识库构建成功"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_system_initialized": rag_system is not None
    }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 