import os
import re
import csv
import math
import numpy as np
from collections import defaultdict, Counter

# 导入三个独立的算法模块
from algorithm1_lsi import LSIAlgorithm
from algorithm2_language_model import LanguageModelAlgorithm  
from algorithm3_bm25 import BM25Algorithm


# ====================== TEXT PREPROCESSOR ======================
class StandaloneTextPreprocessor:
    """
    完全独立的文本预处理器 - 无NLTK依赖
    """
    def __init__(self):
        # 综合英语停用词列表
        self.stop_words = {
            'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
            'any', 'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
            'below', 'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do',
            'does', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
            'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him',
            'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
            'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of',
            'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
            'over', 'own', 's', 'same', 'she', 'should', 'so', 'some', 'such', 't', 'than',
            'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these',
            'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very',
            'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
            'why', 'will', 'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves'
        }
    
    def simple_tokenize(self, text):
        """使用正则表达式简单分词"""
        # 替换标点符号为空格
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词并过滤空字符串
        tokens = [token.strip().lower() for token in text.split() if token.strip()]
        return tokens
    
    def simple_stem(self, word):
        """基础词干提取 - 移除常见英语后缀"""
        # 定义要移除的常见后缀
        suffixes = [
            'ings', 'ing', 'edly', 'ied', 'ies', 'ed', 'es', 's',
            'erly', 'er', 'est', 'ly', 'tion', 'ness', 'ment'
        ]
        
        word = word.lower()
        original_word = word
        
        # 尝试移除后缀
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stemmed = word[:-len(suffix)]
                # 基本规则避免过度词干化
                if len(stemmed) >= 3:
                    word = stemmed
                    break
        
        return word
    
    def preprocess_text(self, text):
        """完整预处理流程"""
        # 第1步：转换为小写并分词
        tokens = self.simple_tokenize(text)
        
        # 第2步：移除停用词和过短词汇
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # 第3步：应用简单词干提取
        stemmed_tokens = [self.simple_stem(token) for token in tokens]
        
        # 第4步：移除重复词汇但保持顺序
        seen = set()
        unique_tokens = []
        for token in stemmed_tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        return ' '.join(unique_tokens)


# UI
class InformationRetrievalSystem:
    def __init__(self):
        print("信息检索系统")
        print("DTS305TC Natural Language Processing")
        print("-" * 40)
        
        self.documents = []
        self.document_names = []
        self.processed_documents = []
        self.preprocessor = StandaloneTextPreprocessor()
        
        # 实例化三种检索算法
        self.lsi_algorithm = LSIAlgorithm(k_dimensions=10)
        self.language_model = LanguageModelAlgorithm(lambda_param=0.7)
        self.bm25_algorithm = BM25Algorithm(k1=1.2, b=0.75)
        
        self.system_ready = False
        
        print("系统就绪。加载算法...")
    
    def database_input_function(self, folder_path="dataset"):
        """(3) 数据库输入函数"""
        print(f"\n从 {folder_path} 加载文档...")
        
        try:
            self.documents = []
            self.document_names = []
            
            if not os.path.exists(folder_path):
                print(f"错误: '{folder_path}' 文件夹不存在")
                return False
            
            total_files = 0
            
            for root, dirs, files in os.walk(folder_path):
                txt_files = [f for f in files if f.endswith('.txt')]
                
                if txt_files:
                    rel_path = os.path.relpath(root, folder_path)
                    category = rel_path if rel_path != '.' else 'main'
                    
                    for filename in sorted(txt_files):
                        filepath = os.path.join(root, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                                content = file.read().strip()
                                if content:
                                    self.documents.append(content)
                                    doc_name = f"{category}_{filename}" if category != 'main' else filename
                                    self.document_names.append(doc_name)
                                    total_files += 1
                        except Exception as e:
                            print(f"加载失败 {filename}")
            
            if total_files == 0:
                print("未找到文档")
                return False
            
            print(f"加载了 {total_files} 个文档")
            return True
            
        except Exception as e:
            print(f"文档加载错误: {e}")
            return False
    
    def text_preprocessing_function(self):
        """(4) 文本预处理函数"""
        print("预处理文档...")
        
        try:
            self.processed_documents = []
            
            for i, doc in enumerate(self.documents):
                processed = self.preprocessor.preprocess_text(doc)
                self.processed_documents.append(processed)
            
            # 统计信息
            all_words = ' '.join(self.processed_documents).split()
            print(f"预处理完成: {len(set(all_words))} 个唯一词汇")
            
            return True
            
        except Exception as e:
            print(f"预处理错误: {e}")
            return False
    
    def information_retrieval_algorithm_1(self, query, top_k=5):
        """(5) 信息检索算法1：使用算法1检索用户输入并提供前五个候选结果"""
        try:
            processed_query = self.preprocessor.preprocess_text(query)
            if not processed_query.strip():
                return []
            
            results = self.lsi_algorithm.retrieve_documents(processed_query, top_k)
            return results
            
        except Exception as e:
            print(f"算法1错误: {e}")
            return []
    
    def information_retrieval_algorithm_2(self, query, top_k=5):
        """(6) 信息检索算法2：使用算法2检索用户输入并提供前五个候选结果"""
        try:
            processed_query = self.preprocessor.preprocess_text(query)
            if not processed_query.strip():
                return []
            
            results = self.language_model.retrieve_documents(processed_query, top_k)
            return results
            
        except Exception as e:
            print(f"算法2错误: {e}")
            return []
    
    def information_retrieval_algorithm_3(self, query, top_k=5):
        """信息检索算法3：使用算法3检索用户输入并提供前五个候选结果"""
        try:
            processed_query = self.preprocessor.preprocess_text(query)
            if not processed_query.strip():
                return []
            
            results = self.bm25_algorithm.retrieve_documents(processed_query, top_k)
            return results
            
        except Exception as e:
            print(f"算法3错误: {e}")
            return []
    
    def output_function(self, query, lsi_results, lm_results, bm25_results=None):
        """(7) 输出函数"""
        print(f"\n查询: {query}")
        print("-" * 40)
        
        print("检索算法1结果:")
        for i, (doc_idx, score) in enumerate(lsi_results[:5]):  # 确保前5个
            print(f"{i+1}. {self.document_names[doc_idx]} ({score:.3f})")
        
        print("\n检索算法2结果:")
        for i, (doc_idx, score) in enumerate(lm_results[:5]):  # 确保前5个
            print(f"{i+1}. {self.document_names[doc_idx]} ({score:.3f})")
        
        # 如果有第三个算法的结果，也显示
        if bm25_results is not None:
            print("\n检索算法3结果:")
            for i, (doc_idx, score) in enumerate(bm25_results[:5]):  # 确保前5个
                print(f"{i+1}. {self.document_names[doc_idx]} ({score:.3f})")
    
    def fusion_output_function(self, query, lsi_results, lm_results, bm25_results=None, alpha=0.6):
        """(8) 融合输出函数"""
        try:
            # 归一化分数
            def normalize(results):
                if not results:
                    return {}
                scores = [s for _, s in results]
                min_s, max_s = min(scores), max(scores)
                if max_s == min_s:
                    return {idx: 0.5 for idx, _ in results}
                return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}
            
            lsi_scores = normalize(lsi_results)
            lm_scores = normalize(lm_results)
            
            if bm25_results is not None:
                # 三算法融合
                bm25_scores = normalize(bm25_results)
                all_docs = set(lsi_scores.keys()) | set(lm_scores.keys()) | set(bm25_scores.keys())
                
                fusion_results = []
                weights = [0.25, 0.35, 0.4]  # LSI, LM, BM25权重
                
                for doc_idx in all_docs:
                    lsi_score = lsi_scores.get(doc_idx, 0)
                    lm_score = lm_scores.get(doc_idx, 0)
                    bm25_score = bm25_scores.get(doc_idx, 0)
                    fusion_score = (weights[0] * lsi_score + 
                                   weights[1] * lm_score + 
                                   weights[2] * bm25_score)
                    fusion_results.append((doc_idx, fusion_score))
                
                print(f"\n融合结果 (权重: 算法1={weights[0]}, 算法2={weights[1]}, 算法3={weights[2]}):")
            else:
                # 两算法融合（保持原有逻辑）
                all_docs = set(lsi_scores.keys()) | set(lm_scores.keys())
                
                fusion_results = []
                for doc_idx in all_docs:
                    lsi_score = lsi_scores.get(doc_idx, 0)
                    lm_score = lm_scores.get(doc_idx, 0)
                    fusion_score = alpha * lsi_score + (1 - alpha) * lm_score
                    fusion_results.append((doc_idx, fusion_score))
                
                print(f"\n融合结果 (α={alpha}):")
            
            fusion_results.sort(key=lambda x: x[1], reverse=True)
            
            for i, (doc_idx, score) in enumerate(fusion_results[:5]):
                print(f"{i+1}. {self.document_names[doc_idx]} ({score:.3f})")
            
            return fusion_results[:5]
            
        except Exception as e:
            print(f"融合错误: {e}")
            return []
    
    def user_input_function(self):
        """(2) 用户输入函数"""
        print("\n输入查询 (输入 'quit' 退出):")
        
        while True:
            try:
                query = input("> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                elif not query:
                    continue
                
                # 处理查询
                lsi_results = self.information_retrieval_algorithm_1(query)
                lm_results = self.information_retrieval_algorithm_2(query)
                bm25_results = self.information_retrieval_algorithm_3(query)
                
                self.output_function(query, lsi_results, lm_results, bm25_results)
                self.fusion_output_function(query, lsi_results, lm_results, bm25_results)
                
            except KeyboardInterrupt:
                print("\n再见!")
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def initialize_system(self, dataset_folder="dataset"):
        """初始化系统"""
        print("初始化系统...")
        
        if not self.database_input_function(dataset_folder):
            return False
        
        if not self.text_preprocessing_function():
            return False
        
        print("构建搜索索引...")
        
        if not self.lsi_algorithm.build_index(self.processed_documents):
            print("警告: LSI索引构建失败")
        
        if not self.language_model.build_index(self.processed_documents):
            print("警告: 语言模型索引构建失败")
        
        if not self.bm25_algorithm.build_index(self.processed_documents):
            print("警告: BM25索引构建失败")
        
        self.system_ready = True
        print("系统就绪!\n")
        
        return True


def main():
    """(1) 主函数"""
    irs = InformationRetrievalSystem()
    
    if irs.initialize_system():
        irs.user_input_function()
    else:
        print("系统初始化失败")


if __name__ == "__main__":
    main()