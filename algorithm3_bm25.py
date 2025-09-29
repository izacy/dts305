import math
import numpy as np
from collections import defaultdict, Counter


class BM25Algorithm:
    """
    BM25 (Best Matching 25) 算法实现
    与现有LSI和Language Model算法接口保持一致
    """
    
    def __init__(self, k1=1.2, b=0.75):
        """
        初始化BM25算法
        
        Args:
            k1 (float): 控制词频饱和度的参数，通常取1.2-2.0
            b (float): 控制文档长度归一化的参数，通常取0.75
        """
        self.k1 = k1  # 词频饱和参数
        self.b = b    # 文档长度归一化参数
        
        # 索引数据结构
        self.doc_freqs = []      # 每个文档的词频字典列表
        self.doc_lengths = []    # 每个文档的长度
        self.avg_doc_length = 0  # 平均文档长度
        self.vocabulary = set()  # 词汇表
        self.doc_count = 0       # 文档总数
        self.idf_cache = {}      # IDF值缓存
        self.is_trained = False
        
    def build_index(self, processed_documents):
        """
        构建BM25索引
        
        Args:
            processed_documents (list): 预处理后的文档列表
            
        Returns:
            bool: 构建成功返回True，失败返回False
        """
        try:
            if len(processed_documents) == 0:
                return False
            
            # 重置数据结构
            self.doc_freqs = []
            self.doc_lengths = []
            self.vocabulary = set()
            
            # 第一步：计算每个文档的词频和长度
            for doc_idx, doc in enumerate(processed_documents):
                words = doc.split()
                
                # 计算词频
                word_freq = Counter(words)
                self.doc_freqs.append(word_freq)
                
                # 记录文档长度
                doc_length = len(words)
                self.doc_lengths.append(doc_length)
                
                # 更新词汇表
                self.vocabulary.update(words)
            
            # 第二步：计算统计信息
            self.doc_count = len(processed_documents)
            self.avg_doc_length = sum(self.doc_lengths) / self.doc_count
            
            # 第三步：预计算IDF值
            self._compute_idf_values()
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"BM25索引构建失败: {e}")
            return False
    
    def _compute_idf_values(self):
        """
        计算所有词汇的IDF值
        
        IDF公式: IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
        其中 N 是文档总数，df(t) 是包含词汇t的文档数
        """
        self.idf_cache = {}
        
        for word in self.vocabulary:
            # 计算文档频率 (Document Frequency)
            df = sum(1 for doc_freq in self.doc_freqs if word in doc_freq)
            
            # 计算IDF值 (使用BM25的IDF公式)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))
            self.idf_cache[word] = idf
    
    def _calculate_bm25_score(self, query_words, doc_idx):
        """
        计算单个文档对查询的BM25分数
        
        Args:
            query_words (list): 查询词汇列表
            doc_idx (int): 文档索引
            
        Returns:
            float: BM25分数
        """
        if doc_idx >= len(self.doc_freqs):
            return 0.0
        
        doc_freq = self.doc_freqs[doc_idx]
        doc_length = self.doc_lengths[doc_idx]
        score = 0.0
        
        # 对查询中的每个词计算BM25贡献
        for word in query_words:
            if word not in self.vocabulary:
                continue  # 跳过不在词汇表中的词
            
            # 获取词频和IDF
            tf = doc_freq.get(word, 0)
            if tf == 0:
                continue  # 如果文档中没有这个词，跳过
            
            idf = self.idf_cache.get(word, 0)
            
            # BM25公式核心部分
            # score += IDF × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d| / avgdl))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            word_score = idf * (numerator / denominator)
            score += word_score
        
        return score
    
    def retrieve_documents(self, query_text, top_k=5):
        """
        使用BM25算法检索文档
        
        Args:
            query_text (str): 查询文本
            top_k (int): 返回前k个文档
            
        Returns:
            list: (文档索引, BM25分数) 的列表，按分数降序排列
        """
        try:
            if not self.is_trained:
                return []
            
            if not query_text.strip():
                return []
            
            # 预处理查询
            query_words = query_text.strip().split()
            if not query_words:
                return []
            
            # 计算每个文档的BM25分数
            doc_scores = []
            for doc_idx in range(self.doc_count):
                score = self._calculate_bm25_score(query_words, doc_idx)
                doc_scores.append((doc_idx, score))
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前top_k个结果
            results = doc_scores[:top_k]
            
            return results
            
        except Exception as e:
            print(f"BM25检索错误: {e}")
            return []
    
    def get_algorithm_info(self):
        """
        获取BM25算法信息
        
        Returns:
            dict: 算法信息字典
        """
        return {
            'name': 'BM25 (Best Matching 25)',
            'type': 'Probabilistic Retrieval',
            'k1_parameter': self.k1,
            'b_parameter': self.b,
            'is_trained': self.is_trained,
            'vocabulary_size': len(self.vocabulary),
            'document_count': self.doc_count,
            'description': 'BM25是一种概率排序函数，改进了TF-IDF，考虑词频饱和和文档长度归一化'
        }


# 测试函数
if __name__ == "__main__":
    # BM25算法测试
    test_docs = [
        "british hurdler sarah claxton confident medal european indoor championships madrid",
        "athlete smash british record hurdles season set new mark second",
        "claxton won national hurdles title past three year struggled international",
        "scotland born athlete equal fifth fastest time world year",
        "birmingham grand prix claxton left european medal favourite russian trailing"
    ]
    
    print("🚀 BM25算法测试")
    print("=" * 50)
    
    bm25 = BM25Algorithm(k1=1.2, b=0.75)
    
    # 构建索引
    success = bm25.build_index(test_docs)
    if success:
        print("\n✅ BM25索引构建成功!")
        
        # 显示算法信息
        info = bm25.get_algorithm_info()
        print(f"\n📊 算法信息:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # 测试查询
        test_queries = [
            "british athlete hurdles",
            "claxton medal",
            "european championships",
            "record time"
        ]
        
        for query in test_queries:
            print(f"\n🔍 查询: '{query}'")
            results = bm25.retrieve_documents(query, top_k=3)
            
            print("BM25结果:")
            for i, (doc_idx, score) in enumerate(results):
                print(f"   {i+1}. 文档{doc_idx}: {score:.4f}")
                print(f"      内容: {test_docs[doc_idx][:80]}...")
    else:
        print("❌ BM25索引构建失败!")