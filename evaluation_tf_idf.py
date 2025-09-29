import os
import re
import csv
import math
import random
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 尝试导入NLTK，如果失败则使用备用方案
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.util import ngrams
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: NLTK不可用，使用简化的文本处理方法")

from main_system import InformationRetrievalSystem


class ImprovedQueryGenerator:
    """
    改进的查询生成器 - 消除人为偏向，提供科学严谨的评估
    """
    
    def __init__(self):
        self.irs = InformationRetrievalSystem()
        
        # 停用词设置
        if NLTK_AVAILABLE:
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                nltk.download('punkt')
                nltk.download('stopwords')
                self.stop_words = set(stopwords.words('english'))
        else:
            # 备用停用词列表
            self.stop_words = {
                'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
                'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
                'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
                'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
                'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
                'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
                'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
                'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
                'come', 'made', 'may', 'part'
            }
        
        # 扩展停用词
        self.stop_words.update(['said', 'also', 'would', 'could', 'one', 'two', 
                               'first', 'last', 'new', 'old', 'good', 'well',
                               'way', 'much', 'many', 'take', 'make', 'get'])
    
    def simple_tokenize(self, text):
        """简单分词方法（NLTK不可用时的备用方案）"""
        if NLTK_AVAILABLE:
            return word_tokenize(text.lower())
        else:
            # 简单的正则表达式分词
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            return words
    
    def generate_document_based_queries(self):
        """
        生成改进的自动查询集合
        
        Returns:
            list: 多样化查询列表
        """
        print("🔍 生成改进的自动查询集合")
        print("=" * 60)
        
        # 初始化IR系统
        if not self.irs.initialize_system():
            print("Failed to load documents")
            return []
        
        all_queries = []
        
        # 1. 基于TF-IDF的重要词汇查询
        tfidf_queries = self._generate_tfidf_queries()
        all_queries.extend(tfidf_queries)
        
        # 2. 基于词汇共现的查询
        cooccurrence_queries = self._generate_cooccurrence_queries()
        all_queries.extend(cooccurrence_queries)
        
        # 3. 跨文档相似性查询
        similarity_queries = self._generate_cross_document_queries()
        all_queries.extend(similarity_queries)
        
        # 4. 负面查询（测试鲁棒性）
        negative_queries = self._generate_negative_queries()
        all_queries.extend(negative_queries)
        
        # 5. 长度分层查询
        length_queries = self._generate_length_varied_queries()
        all_queries.extend(length_queries)
        
        # 选择最终的多样化查询集合
        selected_queries = self._select_diverse_queries(all_queries, target_count=5)
        
        print(f"\n✅ 生成了 {len(selected_queries)} 个改进查询")
        return selected_queries
    
    def _generate_tfidf_queries(self):
        """基于TF-IDF权重生成客观重要词汇查询"""
        print("📊 1. 基于TF-IDF生成重要词汇查询...")
        
        # 预处理文档
        processed_docs = []
        for doc in self.irs.documents:
            processed = self.irs.preprocessor.preprocess_text(doc)
            processed_docs.append(processed)
        
        # 计算TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                min_df=1,
                max_df=0.8,
                ngram_range=(1, 1),
                stop_words='english' if not self.stop_words else list(self.stop_words)
            )
            tfidf_matrix = vectorizer.fit_transform(processed_docs)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"   TF-IDF计算失败: {e}")
            return []
        
        queries = []
        
        # 为每个文档生成基于重要词汇的查询
        for doc_idx, doc_name in enumerate(self.irs.document_names):
            if doc_idx < tfidf_matrix.shape[0]:
                doc_scores = tfidf_matrix[doc_idx].toarray()[0]
                
                # 获取前5个重要词汇
                top_indices = np.argsort(doc_scores)[::-1][:5]
                top_words = []
                for idx in top_indices:
                    if doc_scores[idx] > 0:
                        word = feature_names[idx]
                        if len(word) > 2 and word not in self.stop_words:
                            top_words.append(word)
                
                if len(top_words) >= 2:
                    # 单词查询
                    queries.append({
                        'query': top_words[0],
                        'type': 'tfidf_single',
                        'source_document': doc_name,
                        'source_doc_index': doc_idx,
                        'expected_category': doc_name.split('_')[0] if '_' in doc_name else 'main',
                        'tfidf_score': float(doc_scores[top_indices[0]])
                    })
                    
                    # 双词组合查询
                    if len(top_words) >= 2:
                        queries.append({
                            'query': f"{top_words[0]} {top_words[1]}",
                            'type': 'tfidf_dual',
                            'source_document': doc_name,
                            'source_doc_index': doc_idx,
                            'expected_category': doc_name.split('_')[0] if '_' in doc_name else 'main',
                            'tfidf_score': float((doc_scores[top_indices[0]] + doc_scores[top_indices[1]]) / 2)
                        })
        
        print(f"   ✓ 生成 {len(queries)} 个TF-IDF查询")
        return queries
    
    def _generate_cooccurrence_queries(self):
        """基于词汇共现模式生成查询"""
        print("🔗 2. 基于词汇共现生成查询...")
        
        # 构建共现统计
        cooccurrence = defaultdict(lambda: defaultdict(int))
        window_size = 5
        
        for doc in self.irs.documents:
            words = self.simple_tokenize(doc)
            # 过滤停用词和短词
            words = [w for w in words if len(w) > 2 and w not in self.stop_words]
            
            # 统计窗口内共现
            for i, word1 in enumerate(words):
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                for j in range(start, end):
                    if i != j:
                        word2 = words[j]
                        cooccurrence[word1][word2] += 1
        
        queries = []
        
        # 生成共现查询
        for word1, cooccur_dict in cooccurrence.items():
            if len(cooccur_dict) >= 2:
                # 按共现频率排序
                sorted_pairs = sorted(cooccur_dict.items(), key=lambda x: x[1], reverse=True)
                
                for word2, freq in sorted_pairs[:1]:  # 只取最强共现
                    if freq >= 2:  # 至少共现2次
                        queries.append({
                            'query': f"{word1} {word2}",
                            'type': 'cooccurrence',
                            'source_document': 'multiple',
                            'source_doc_index': -1,
                            'expected_category': 'cross_category',
                            'cooccurrence_freq': freq
                        })
        
        # 按共现强度排序，选择前几个
        queries.sort(key=lambda x: x['cooccurrence_freq'], reverse=True)
        result = queries[:3]  # 限制数量
        
        print(f"   ✓ 生成 {len(result)} 个共现查询")
        return result
    
    def _generate_cross_document_queries(self):
        """基于文档间相似性生成跨文档查询"""
        print("🔄 3. 生成跨文档相似性查询...")
        
        # 计算文档相似性
        processed_docs = [self.irs.preprocessor.preprocess_text(doc) for doc in self.irs.documents]
        
        try:
            vectorizer = TfidfVectorizer(max_features=200, min_df=1, max_df=0.9)
            tfidf_matrix = vectorizer.fit_transform(processed_docs)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            print(f"   跨文档查询生成失败: {e}")
            return []
        
        queries = []
        
        # 找到中等相似度的文档对
        for i in range(len(self.irs.documents)):
            for j in range(i + 1, len(self.irs.documents)):
                similarity = similarity_matrix[i][j]
                
                if 0.1 < similarity < 0.7:  # 中等相似度
                    # 找到共同重要词汇
                    doc1_scores = tfidf_matrix[i].toarray()[0]
                    doc2_scores = tfidf_matrix[j].toarray()[0]
                    
                    common_words = []
                    for idx, (score1, score2) in enumerate(zip(doc1_scores, doc2_scores)):
                        if score1 > 0.1 and score2 > 0.1:
                            word = feature_names[idx]
                            if len(word) > 2 and word not in self.stop_words:
                                common_words.append((word, min(score1, score2)))
                    
                    if len(common_words) >= 2:
                        # 按重要性排序
                        common_words.sort(key=lambda x: x[1], reverse=True)
                        top_words = [word for word, score in common_words[:2]]
                        
                        queries.append({
                            'query': ' '.join(top_words),
                            'type': 'cross_document',
                            'source_document': f"{self.irs.document_names[i]}+{self.irs.document_names[j]}",
                            'source_doc_index': [i, j],
                            'expected_category': 'cross_category',
                            'similarity': float(similarity)
                        })
        
        # 选择前几个
        queries.sort(key=lambda x: x['similarity'], reverse=True)
        result = queries[:2]
        
        print(f"   ✓ 生成 {len(result)} 个跨文档查询")
        return result
    
    def _generate_negative_queries(self):
        """生成负面查询测试系统鲁棒性"""
        print("❌ 4. 生成负面查询...")
        
        # 收集文档中的所有词汇
        all_words = set()
        for doc in self.irs.documents:
            words = self.simple_tokenize(doc)
            all_words.update(words)
        
        # 预定义的无关领域词汇
        irrelevant_domains = [
            ['cooking', 'recipe', 'ingredient', 'kitchen', 'chef'],
            ['weather', 'temperature', 'rain', 'sunny', 'cloud'],
            ['mathematics', 'equation', 'formula', 'calculate', 'algebra'],
            ['music', 'song', 'melody', 'instrument', 'concert'],
            ['travel', 'journey', 'destination', 'vacation', 'tourist']
        ]
        
        queries = []
        
        for domain_words in irrelevant_domains:
            # 检查这些词是否真的不在文档中
            words_in_docs = [word for word in domain_words if word in all_words]
            
            if len(words_in_docs) == 0:  # 确保完全无关
                # 单词查询
                queries.append({
                    'query': domain_words[0],
                    'type': 'negative_single',
                    'source_document': 'none',
                    'source_doc_index': -1,
                    'expected_category': 'irrelevant'
                })
                break  # 只需要一个负面查询
        
        print(f"   ✓ 生成 {len(queries)} 个负面查询")
        return queries
    
    def _generate_length_varied_queries(self):
        """生成不同长度的查询测试算法对复杂度的处理"""
        print("📏 5. 生成长度分层查询...")
        
        queries = []
        
        # 短查询 (1词) - 基于高频重要词
        processed_docs = [self.irs.preprocessor.preprocess_text(doc) for doc in self.irs.documents]
        all_words = []
        for doc in processed_docs:
            all_words.extend(doc.split())
        
        word_freq = Counter(all_words)
        # 过滤停用词和短词
        important_words = [word for word, freq in word_freq.most_common(20) 
                          if len(word) > 3 and word not in self.stop_words and freq >= 2]
        
        if important_words:
            queries.append({
                'query': important_words[0],
                'type': 'short_query',
                'source_document': 'multiple',
                'source_doc_index': -1,
                'expected_category': 'general',
                'length': 1
            })
        
        print(f"   ✓ 生成 {len(queries)} 个长度分层查询")
        return queries
    
    def _select_diverse_queries(self, all_queries, target_count=5):
        """选择多样化的查询子集"""
        print(f"\n🎯 从 {len(all_queries)} 个候选查询中选择 {target_count} 个...")
        
        if len(all_queries) <= target_count:
            # 如果查询数量不足，直接返回所有查询
            for i, query in enumerate(all_queries, 1):
                query['id'] = f'Q{i}'
            return all_queries
        
        # 按类型分组
        by_type = defaultdict(list)
        for query in all_queries:
            by_type[query['type']].append(query)
        
        selected = []
        
        # 确保每种类型至少有一个代表
        type_list = list(by_type.keys())
        queries_per_type = target_count // len(type_list)
        remainder = target_count % len(type_list)
        
        for i, query_type in enumerate(type_list):
            count = queries_per_type + (1 if i < remainder else 0)
            type_queries = by_type[query_type]
            
            # 根据类型选择最佳查询
            if query_type in ['tfidf_single', 'tfidf_dual']:
                # TF-IDF查询按分数排序
                type_queries.sort(key=lambda x: x.get('tfidf_score', 0), reverse=True)
            elif query_type == 'cooccurrence':
                # 共现查询按频率排序
                type_queries.sort(key=lambda x: x.get('cooccurrence_freq', 0), reverse=True)
            elif query_type == 'cross_document':
                # 跨文档查询按相似度排序
                type_queries.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            else:
                # 其他类型随机选择
                random.shuffle(type_queries)
            
            # 选择该类型的查询
            selected.extend(type_queries[:count])
        
        # 如果仍然不够，从剩余查询中随机选择
        while len(selected) < target_count:
            remaining = [q for q in all_queries if q not in selected]
            if not remaining:
                break
            selected.append(random.choice(remaining))
        
        # 截取到目标数量并重新编号
        selected = selected[:target_count]
        for i, query in enumerate(selected, 1):
            query['id'] = f'Q{i}'
        
        # 显示选择的查询分布
        print("\n📋 选择的查询类型分布:")
        type_counts = Counter(q['type'] for q in selected)
        for qtype, count in type_counts.items():
            print(f"   • {qtype}: {count} 个")
        
        return selected
    
    def create_automatic_relevance_judgments(self, queries):
        """
        为查询创建改进的自动相关性判断
        
        Args:
            queries (list): 查询列表
            
        Returns:
            dict: 相关性判断字典
        """
        print(f"\n🎯 创建改进的自动相关性判断")
        print("-" * 50)
        
        relevance_judgments = {}
        
        # 预处理文档用于语义相似度计算
        processed_docs = [self.irs.preprocessor.preprocess_text(doc) for doc in self.irs.documents]
        
        # 构建TF-IDF向量用于语义相似度
        try:
            vectorizer = TfidfVectorizer(max_features=300, min_df=1, max_df=0.9)
            doc_vectors = vectorizer.fit_transform(processed_docs)
        except Exception as e:
            print(f"   语义相似度计算失败，使用简单匹配: {e}")
            doc_vectors = None
            vectorizer = None
        
        for query_info in queries:
            query_id = query_info['id']
            query_text = query_info['query']
            query_type = query_info['type']
            
            relevance = {}
            
            # 计算查询向量（如果可能）
            query_vector = None
            if vectorizer is not None:
                try:
                    query_processed = self.irs.preprocessor.preprocess_text(query_text)
                    query_vector = vectorizer.transform([query_processed])
                except:
                    pass
            
            for doc_idx in range(len(self.irs.documents)):
                doc_content = processed_docs[doc_idx]
                query_words = query_text.lower().split()
                
                # 1. 基础词汇匹配度
                word_matches = sum(1 for word in query_words if word in doc_content)
                match_ratio = word_matches / len(query_words) if query_words else 0
                
                # 2. 语义相似度（如果可用）
                semantic_sim = 0.0
                if query_vector is not None and doc_vectors is not None:
                    try:
                        sim_matrix = cosine_similarity(query_vector, doc_vectors[doc_idx:doc_idx+1])
                        semantic_sim = sim_matrix[0][0]
                    except:
                        pass
                
                # 3. 综合相关性计算
                relevance_score = self._calculate_relevance_score(
                    query_info, doc_idx, match_ratio, semantic_sim
                )
                
                relevance[doc_idx] = relevance_score
            
            relevance_judgments[query_id] = relevance
            
            # 显示相关性判断摘要
            relevant_docs = sum(1 for score in relevance.values() if score > 0)
            highly_relevant = sum(1 for score in relevance.values() if score >= 2)
            
            print(f"   {query_id}: '{query_text}' - {relevant_docs} 相关文档 ({highly_relevant} 高度相关)")
        
        return relevance_judgments
    
    def _calculate_relevance_score(self, query_info, doc_idx, match_ratio, semantic_sim):
        """
        计算改进的相关性分数
        
        Args:
            query_info: 查询信息
            doc_idx: 文档索引
            match_ratio: 词汇匹配比例
            semantic_sim: 语义相似度
            
        Returns:
            int: 相关性分数 (0-3)
        """
        query_type = query_info['type']
        
        # 负面查询应该不相关
        if query_type in ['negative_single', 'negative_dual']:
            return 0
        
        # 组合匹配度和语义相似度
        combined_score = 0.7 * match_ratio + 0.3 * semantic_sim
        
        # 根据查询类型调整相关性判断
        if query_type in ['tfidf_single', 'tfidf_dual']:
            # TF-IDF查询：对源文档给予适度偏重
            if isinstance(query_info['source_doc_index'], int) and doc_idx == query_info['source_doc_index']:
                # 源文档，但不自动给最高分，基于实际匹配度
                if combined_score >= 0.6:
                    return 3
                elif combined_score >= 0.4:
                    return 2
                else:
                    return 1
            else:
                # 非源文档，纯粹基于匹配度
                if combined_score >= 0.8:
                    return 2
                elif combined_score >= 0.5:
                    return 1
                else:
                    return 0
        
        elif query_type == 'cross_document':
            # 跨文档查询：对目标文档公平对待
            if isinstance(query_info['source_doc_index'], list) and doc_idx in query_info['source_doc_index']:
                if combined_score >= 0.5:
                    return 2
                else:
                    return 1
            else:
                if combined_score >= 0.7:
                    return 1
                else:
                    return 0
        
        elif query_type == 'cooccurrence':
            # 共现查询：基于实际匹配情况
            if combined_score >= 0.7:
                return 2
            elif combined_score >= 0.4:
                return 1
            else:
                return 0
        
        else:
            # 其他查询类型：标准判断
            if combined_score >= 0.8:
                return 2
            elif combined_score >= 0.5:
                return 1
            else:
                return 0


class DocumentBasedEvaluator:
    """
    改进的文档评估器
    """
    
    def __init__(self):
        self.query_generator = ImprovedQueryGenerator()
        self.irs = self.query_generator.irs
        self.selected_queries = []
        self.relevance_judgments = {}
        
    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k=5):
        """计算Precision@K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        top_k_docs = [doc_id for doc_id, _ in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in top_k_docs 
                               if relevant_docs.get(doc_id, 0) > 0)
        
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k=5):
        """计算Recall@K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        if total_relevant == 0:
            return 0.0
        
        top_k_docs = [doc_id for doc_id, _ in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in top_k_docs 
                               if relevant_docs.get(doc_id, 0) > 0)
        
        return relevant_retrieved / total_relevant
    
    def calculate_dcg_at_k(self, retrieved_docs, relevant_docs, k=5):
        """计算DCG@K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        dcg = 0.0
        for i, (doc_id, _) in enumerate(retrieved_docs[:k]):
            relevance = relevant_docs.get(doc_id, 0)
            
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / math.log2(i + 2)
        
        return dcg
    
    def calculate_ideal_dcg_at_k(self, relevant_docs, k=5):
        """计算Ideal DCG@K"""
        sorted_relevances = sorted(relevant_docs.values(), reverse=True)
        
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances[:k]):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / math.log2(i + 2)
        
        return idcg
    
    def calculate_ndcg_at_k(self, retrieved_docs, relevant_docs, k=5):
        """计算nDCG@K"""
        dcg = self.calculate_dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg = self.calculate_ideal_dcg_at_k(relevant_docs, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_fusion_results(self, lsi_results, lm_results, alpha=0.6):
        """计算融合结果"""
        def normalize_scores(results):
            if not results:
                return {}
            scores = [score for _, score in results]
            min_score, max_score = min(scores), max(scores)
            if max_score == min_score:
                return {doc_idx: 0.5 for doc_idx, _ in results}
            return {doc_idx: (score - min_score) / (max_score - min_score) 
                   for doc_idx, score in results}
        
        lsi_scores = normalize_scores(lsi_results)
        lm_scores = normalize_scores(lm_results)
        all_doc_indices = set(lsi_scores.keys()) | set(lm_scores.keys())
        
        fusion_results = []
        for doc_idx in all_doc_indices:
            lsi_score = lsi_scores.get(doc_idx, 0)
            lm_score = lm_scores.get(doc_idx, 0)
            fusion_score = alpha * lsi_score + (1 - alpha) * lm_score
            fusion_results.append((doc_idx, fusion_score))
        
        fusion_results.sort(key=lambda x: x[1], reverse=True)
        return fusion_results
    
    def evaluate_single_query(self, query_info):
        """评估单个查询"""
        query_id = query_info['id']
        query_text = query_info['query']
        
        print(f"\n🔎 评估 {query_id}: '{query_text}' (类型: {query_info['type']})")
        
        # 获取结果
        lsi_results = self.irs.information_retrieval_algorithm_1(query_text, top_k=5)
        lm_results = self.irs.information_retrieval_algorithm_2(query_text, top_k=5)
        fusion_results = self.calculate_fusion_results(lsi_results, lm_results)
        
        # 获取相关性判断
        relevant_docs = self.relevance_judgments[query_id]
        
        # 计算指标
        result = {
            'query_id': query_id,
            'query_text': query_text,
            'query_type': query_info['type'],
            'source_document': query_info.get('source_document', 'unknown'),
            'expected_category': query_info.get('expected_category', 'unknown')
        }
        
        methods = [
            ('LSI', lsi_results),
            ('Language_Model', lm_results),
            ('Fusion', fusion_results)
        ]
        
        for method_name, retrieved_docs in methods:
            precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k=5)
            recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k=5)
            ndcg = self.calculate_ndcg_at_k(retrieved_docs, relevant_docs, k=5)
            
            result[f'{method_name}_Precision@5'] = precision
            result[f'{method_name}_Recall@5'] = recall
            result[f'{method_name}_nDCG@5'] = ndcg
            
            print(f"   {method_name:12}: P@5={precision:.3f}, R@5={recall:.3f}, nDCG@5={ndcg:.3f}")
            
            # 显示前3个检索结果
            if retrieved_docs:
                top_3 = [(self.irs.document_names[doc_id], score, relevant_docs.get(doc_id, 0)) 
                        for doc_id, score in retrieved_docs[:3]]
                print(f"      前3名: {[(name.split('.')[0], f'{score:.3f}', f'rel={rel}') for name, score, rel in top_3]}")
        
        return result
    
    def run_document_based_evaluation(self):
        """运行完整的改进评估"""
        print("🚀 改进的文档基础查询评估")
        print("DTS305TC Natural Language Processing - 改进版 Part 4")
        print("=" * 80)
        
        print("🎯 改进方案特点:")
        print("  • 基于TF-IDF的客观重要性")
        print("  • 词汇共现模式分析")
        print("  • 跨文档相似性测试")
        print("  • 负面查询鲁棒性测试")
        print("  • 语义相似度增强相关性判断")
        print("  • 消除人为偏向，提供科学评估")
        print()
        
        # 生成改进的查询
        all_queries = self.query_generator.generate_document_based_queries()
        
        if len(all_queries) < 5:
            print(f"⚠️  只生成了 {len(all_queries)} 个查询，继续评估...")
        
        self.selected_queries = all_queries
        
        # 创建改进的相关性判断
        self.relevance_judgments = self.query_generator.create_automatic_relevance_judgments(self.selected_queries)
        
        # 评估每个查询
        print(f"\n" + "="*80)
        print("Part 4.1-4.4: 改进评估结果")
        print("="*80)
        
        evaluation_results = []
        
        for query_info in self.selected_queries:
            result = self.evaluate_single_query(query_info)
            evaluation_results.append(result)
        
        # 计算汇总统计
        self.calculate_summary_statistics(evaluation_results)
        
        # 保存结果
        self.save_results(evaluation_results)
        
        print(f"\n✅ 改进的文档基础评估完成!")
        print(f"📊 使用科学方法生成多样化查询")
        print(f"🎯 消除人为偏向，提供客观评估")
        print(f"📄 课程作业文件已保存在 'results/' 目录")
        
        return True
    
    def calculate_summary_statistics(self, results):
        """计算和显示汇总统计"""
        print(f"\n📈 汇总统计")
        print("=" * 50)
        
        methods = ['LSI', 'Language_Model', 'Fusion']
        metrics = ['Precision@5', 'Recall@5', 'nDCG@5']
        
        print(f"{'方法':<15} {'指标':<12} {'均值':<8} {'标准差':<8} {'最小值':<8} {'最大值':<8}")
        print("-" * 70)
        
        for method in methods:
            for i, metric in enumerate(metrics):
                metric_key = f'{method}_{metric}'
                values = [result[metric_key] for result in results]
                
                avg_value = np.mean(values)
                std_value = np.std(values) if len(values) > 1 else 0.0
                max_value = np.max(values)
                min_value = np.min(values)
                
                method_display = method if i == 0 else ""
                print(f"{method_display:<15} {metric:<12} {avg_value:.3f}    {std_value:.3f}    {min_value:.3f}    {max_value:.3f}")
            if method != methods[-1]:
                print()
        
        # 找到最佳方法
        print(f"\n🏆 最佳表现方法:")
        for metric in metrics:
            best_method = None
            best_score = -1
            
            for method in methods:
                avg_score = np.mean([result[f'{method}_{metric}'] for result in results])
                if avg_score > best_score:
                    best_score = avg_score
                    best_method = method
            
            print(f"   {metric:<12}: {best_method:<15} ({best_score:.3f})")
    
    def save_results(self, results):
        """保存结果到CSV文件"""
        print(f"\n💾 保存评估结果")
        print("-" * 30)
        
        os.makedirs('results', exist_ok=True)
        
        # 保存precision结果
        precision_data = []
        for result in results:
            row = {
                'Query_ID': result['query_id'],
                'Query_Text': result['query_text'],
                'Query_Type': result['query_type'],
                'Source_Document': result['source_document'],
                'LSI_Precision@5': result['LSI_Precision@5'],
                'Language_Model_Precision@5': result['Language_Model_Precision@5'],
                'Fusion_Precision@5': result['Fusion_Precision@5']
            }
            precision_data.append(row)
        
        with open('results/precision.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=precision_data[0].keys())
            writer.writeheader()
            writer.writerows(precision_data)
        
        # 保存recall结果
        recall_data = []
        for result in results:
            row = {
                'Query_ID': result['query_id'],
                'Query_Text': result['query_text'],
                'Query_Type': result['query_type'],
                'Source_Document': result['source_document'],
                'LSI_Recall@5': result['LSI_Recall@5'],
                'Language_Model_Recall@5': result['Language_Model_Recall@5'],
                'Fusion_Recall@5': result['Fusion_Recall@5']
            }
            recall_data.append(row)
        
        with open('results/recall.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=recall_data[0].keys())
            writer.writeheader()
            writer.writerows(recall_data)
        
        # 保存nDCG结果
        ndcg_data = []
        for result in results:
            row = {
                'Query_ID': result['query_id'],
                'Query_Text': result['query_text'],
                'Query_Type': result['query_type'],
                'Source_Document': result['source_document'],
                'LSI_nDCG@5': result['LSI_nDCG@5'],
                'Language_Model_nDCG@5': result['Language_Model_nDCG@5'],
                'Fusion_nDCG@5': result['Fusion_nDCG@5']
            }
            ndcg_data.append(row)
        
        with open('results/nDCG.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ndcg_data[0].keys())
            writer.writeheader()
            writer.writerows(ndcg_data)
        
        # 保存相关性判断
        relevance_data = []
        for query_id, judgments in self.relevance_judgments.items():
            for doc_id, relevance in judgments.items():
                row = {
                    'Query_ID': query_id,
                    'Document_ID': doc_id,
                    'Document_Name': self.irs.document_names[doc_id] if doc_id < len(self.irs.document_names) else f'doc_{doc_id}',
                    'Relevance_Score': relevance
                }
                relevance_data.append(row)
        
        with open('results/relevance_judgments.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=relevance_data[0].keys())
            writer.writeheader()
            writer.writerows(relevance_data)
        
        print(f"✅ 结果已保存:")
        print(f"   • precision.csv")
        print(f"   • recall.csv") 
        print(f"   • nDCG.csv")
        print(f"   • relevance_judgments.csv")


def main():
    """运行改进的文档基础评估"""
    evaluator = DocumentBasedEvaluator()
    
    if evaluator.run_document_based_evaluation():
        print(f"\n🎉 改进评估完成!")
        print(f"🔬 科学严谨的算法评估")
        print(f"📈 消除人为偏向的客观结果")
        print(f"📂 课程作业文件已准备就绪")
    else:
        print(f"\n❌ 评估失败!")


if __name__ == "__main__":
    main()