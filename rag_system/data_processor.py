import pandas as pd
import os
from typing import List, Dict, Tuple

class DataProcessor:
    """数据处理类"""
    
    def __init__(self):
        print("数据处理器初始化完成")
    
    def process_student_data(self, csv_file: str) -> Tuple[List[str], List[Dict]]:
        """处理学生数据CSV文件"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"文件不存在: {csv_file}")
        
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"读取学生数据: {len(df)} 条记录")
        
        documents = []
        metadatas = []
        
        for _, row in df.iterrows():
            description = self.generate_student_description(row)
            documents.append(description)
            
            metadata = {
                "type": "student",
                "student_id": str(row.get('student_id', '')),
                "name": str(row.get('name', '')),
                "major": str(row.get('major', '')),
                "graduation_year": str(row.get('graduation_year', ''))
            }
            metadatas.append(metadata)
        
        print(f"处理完成，生成 {len(documents)} 个学生文档")
        return documents, metadatas
    
    def generate_student_description(self, student_data: pd.Series) -> str:
        """生成学生描述文本"""
        parts = []
        
        if pd.notna(student_data.get('name')):
            parts.append(f"学生{student_data['name']}")
        
        if pd.notna(student_data.get('major')):
            parts.append(f"{student_data['major']}专业")
        
        if pd.notna(student_data.get('gpa')):
            parts.append(f"GPA成绩{student_data['gpa']}")
        
        if pd.notna(student_data.get('graduation_year')):
            parts.append(f"{student_data['graduation_year']}年毕业")
        
        if pd.notna(student_data.get('employment_status')):
            if student_data['employment_status'] == '已就业':
                if pd.notna(student_data.get('company')):
                    parts.append(f"在{student_data['company']}工作")
                if pd.notna(student_data.get('position')):
                    parts.append(f"担任{student_data['position']}")
        
        return "，".join(parts) + "。"
    
    def process_cultivation_plan(self, text_file: str) -> Tuple[List[str], List[Dict]]:
        """处理培养方案文档"""
        if not os.path.exists(text_file):
            raise FileNotFoundError(f"文件不存在: {text_file}")
        
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"读取培养方案文档，长度: {len(content)} 字符")
        
        chunks = self.split_text(content)
        
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata = {
                "type": "cultivation_plan",
                "chunk_id": i,
                "source": text_file
            }
            metadatas.append(metadata)
        
        print(f"处理完成，生成 {len(documents)} 个文档块")
        return documents, metadatas
    
    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """分割长文本为小块"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            if end < text_length:
                # 寻找最近的句号作为分割点
                last_period = text.rfind('。', start, end)
                if last_period > start:
                    end = last_period + 1
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) > 20:
                chunks.append(chunk)
            
            # 移动到下一个位置
            start = max(start + 1, end - overlap)
            
            # 防止无限循环
            if start >= text_length:
                break
        
        return chunks
    
    def process_mixed_data(self, 
                          student_csv: str = None, 
                          plan_txt: str = None) -> Tuple[List[str], List[Dict]]:
        """处理混合数据"""
        all_documents = []
        all_metadatas = []
        
        # 处理学生数据
        if student_csv and os.path.exists(student_csv):
            print("处理学生数据...")
            student_docs, student_metas = self.process_student_data(student_csv)
            all_documents.extend(student_docs)
            all_metadatas.extend(student_metas)
        
        # 处理培养方案
        if plan_txt and os.path.exists(plan_txt):
            print("处理培养方案...")
            plan_docs, plan_metas = self.process_cultivation_plan(plan_txt)
            all_documents.extend(plan_docs)
            all_metadatas.extend(plan_metas)
        
        print(f"混合数据处理完成，总共 {len(all_documents)} 个文档")
        return all_documents, all_metadatas 