import os
import re
import csv
import math
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

# 尝试下载所需的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("下载NLTK数据...")
    nltk.download('punkt')
    nltk.download('stopwords')

from main_3 import InformationRetrievalSystem


class DocumentBasedQueryGenerator:
    """
    基于文档内容的查询生成器
    """
    
    def __init__(self):
        self.irs = InformationRetrievalSystem()
        self.stop_words = set(stopwords.words('english'))
        
        # 为更好的查询质量添加额外的停用词
        self.stop_words.update(['said', 'also', 'would', 'could', 'one', 'two', 
                               'first', 'last', 'new', 'old', 'good', 'well',
                               'way', 'much', 'many', 'take', 'make', 'get'])
    
    def extract_key_phrases_from_document(self, doc_content, doc_name, max_phrases=10):
        """
        从文档中提取有意义的短语
        
        Args:
            doc_content (str): 文档文本
            doc_name (str): 文档文件名
            max_phrases (int): 最多提取的短语数
            
        Returns:
            list: 关键短语列表，包含元数据
        """
        # 清理和分词
        sentences = sent_tokenize(doc_content)
        words = word_tokenize(doc_content.lower())
        
        # 移除停用词和短词
        meaningful_words = [word for word in words 
                           if word.isalpha() and len(word) > 3 
                           and word not in self.stop_words]
        
        # 获取词频
        word_freq = Counter(meaningful_words)
        
        # 提取不同类型的短语
        phrases = []
        
        # 1. 高频有意义词汇 (单词查询)
        for word, freq in word_freq.most_common(5):
            if freq >= 2:  # 词汇至少出现两次
                phrases.append({
                    'phrase': word,
                    'type': 'single_word',
                    'frequency': freq,
                    'source_doc': doc_name,
                    'length': 1
                })
        
        # 2. 提取名词短语和有意义的双词组合
        pos_tagged = nltk.pos_tag(meaningful_words)
        
        # 简单名词短语提取 (名词 + 名词, 形容词 + 名词)
        for i in range(len(pos_tagged) - 1):
            word1, pos1 = pos_tagged[i]
            word2, pos2 = pos_tagged[i + 1]
            
            # 名词 + 名词 或 形容词 + 名词
            if ((pos1.startswith('NN') and pos2.startswith('NN')) or
                (pos1.startswith('JJ') and pos2.startswith('NN'))):
                
                bigram = f"{word1} {word2}"
                
                # 检查此双词组合是否在原文中出现
                if bigram in doc_content.lower():
                    phrases.append({
                        'phrase': bigram,
                        'type': 'noun_phrase',
                        'frequency': doc_content.lower().count(bigram),
                        'source_doc': doc_name,
                        'length': 2
                    })
        
        # 3. 从句子中提取有意义的三元组
        for sentence in sentences[:5]:  # 检查前5个句子
            sentence_words = [word.lower() for word in word_tokenize(sentence) 
                            if word.isalpha() and word.lower() not in self.stop_words]
            
            if len(sentence_words) >= 3:
                for trigram in ngrams(sentence_words, 3):
                    trigram_phrase = ' '.join(trigram)
                    
                    # 只包含有意义内容
                    if any(word in word_freq and word_freq[word] >= 2 for word in trigram):
                        phrases.append({
                            'phrase': trigram_phrase,
                            'type': 'sentence_trigram',
                            'frequency': 1,
                            'source_doc': doc_name,
                            'length': 3
                        })
        
        # 4. 提取关键句子 (简化版)
        for sentence in sentences:
            words_in_sentence = [word.lower() for word in word_tokenize(sentence) 
                               if word.isalpha()]
            
            # 通过重要词汇频率为句子评分
            sentence_score = sum(word_freq.get(word, 0) for word in words_in_sentence)
            
            if sentence_score >= 5 and len(words_in_sentence) <= 8:
                # 从句子中的重要词汇创建短语
                important_words = [word for word in words_in_sentence 
                                 if word in word_freq and word_freq[word] >= 2][:4]
                
                if len(important_words) >= 2:
                    phrase = ' '.join(important_words)
                    phrases.append({
                        'phrase': phrase,
                        'type': 'key_sentence_extract',
                        'frequency': sentence_score,
                        'source_doc': doc_name,
                        'length': len(important_words)
                    })
        
        # 移除重复并按相关性排序
        unique_phrases = {}
        for phrase_info in phrases:
            phrase = phrase_info['phrase']
            if phrase not in unique_phrases:
                unique_phrases[phrase] = phrase_info
            elif phrase_info['frequency'] > unique_phrases[phrase]['frequency']:
                unique_phrases[phrase] = phrase_info
        
        # 按频率和长度排序 (偏好更具体的短语)
        sorted_phrases = sorted(unique_phrases.values(), 
                              key=lambda x: (x['frequency'] * x['length']), 
                              reverse=True)
        
        return sorted_phrases[:max_phrases]
    
    def generate_document_based_queries(self):
        """
        从集合中的所有文档生成查询
        
        Returns:
            list: 生成的查询列表，包含元数据
        """
        print("📚 从文档内容生成查询")
        print("=" * 60)
        
        # 初始化IR系统以加载文档
        if not self.irs.initialize_system():
            print("文档加载失败")
            return []
        
        all_queries = []
        query_id = 1
        
        # 从每个文档提取短语
        for i, (doc_name, doc_content) in enumerate(zip(self.irs.document_names, self.irs.documents)):
            print(f"\n📄 分析 {doc_name}:")
            
            # 提取关键短语
            phrases = self.extract_key_phrases_from_document(doc_content, doc_name, max_phrases=5)
            
            print(f"   提取了 {len(phrases)} 个关键短语:")
            
            # 将短语转换为查询
            for phrase_info in phrases[:3]:  # 每个文档使用前3个短语
                query_info = {
                    'id': f'Q{query_id}',
                    'query': phrase_info['phrase'],
                    'type': phrase_info['type'],
                    'length': phrase_info['length'],
                    'source_document': doc_name,
                    'source_doc_index': i,
                    'frequency_in_source': phrase_info['frequency'],
                    'expected_category': doc_name.split('_')[0] if '_' in doc_name else 'main'
                }
                
                all_queries.append(query_info)
                print(f"   Q{query_id}: '{phrase_info['phrase']}' ({phrase_info['type']}, freq={phrase_info['frequency']})")
                
                query_id += 1
        
        print(f"\n✅ 从文档内容生成了 {len(all_queries)} 个查询")
        
        return all_queries
    
    def create_automatic_relevance_judgments(self, queries):
        """
        基于查询来源和内容分析自动创建相关性判断
        
        Args:
            queries (list): 查询字典列表
            
        Returns:
            dict: 每个查询的相关性判断
        """
        print(f"\n🎯 创建自动相关性判断")
        print("-" * 50)
        
        relevance_judgments = {}
        
        for query_info in queries:
            query_id = query_info['id']
            query_text = query_info['query']
            source_doc_index = query_info['source_doc_index']
            expected_category = query_info['expected_category']
            
            # 初始化相关性分数
            relevance = {}
            
            for doc_idx in range(len(self.irs.documents)):
                doc_name = self.irs.document_names[doc_idx]
                doc_content = self.irs.documents[doc_idx].lower()
                doc_category = doc_name.split('_')[0] if '_' in doc_name else 'main'
                
                # 基础相关性计算
                relevance_score = 0
                
                # 1. 来源文档获得最高相关性
                if doc_idx == source_doc_index:
                    relevance_score = 3  # 高度相关
                
                # 2. 同类别文档
                elif doc_category == expected_category:
                    # 检查查询词汇是否在文档中出现
                    query_words = query_text.lower().split()
                    word_matches = sum(1 for word in query_words if word in doc_content)
                    
                    if word_matches == len(query_words):
                        relevance_score = 2  # 相关
                    elif word_matches > 0:
                        relevance_score = 1  # 部分相关
                    else:
                        relevance_score = 0  # 不相关
                
                # 3. 不同类别文档
                else:
                    # 检查语义相关性 (查询词汇在内容中)
                    query_words = query_text.lower().split()
                    word_matches = sum(1 for word in query_words if word in doc_content)
                    
                    if word_matches == len(query_words):
                        relevance_score = 1  # 部分相关 (跨领域)
                    elif word_matches > len(query_words) / 2:
                        relevance_score = 1  # 某些相关性
                    else:
                        relevance_score = 0  # 不相关
                
                relevance[doc_idx] = relevance_score
            
            relevance_judgments[query_id] = relevance
            
            # 显示相关性判断摘要
            relevant_docs = sum(1 for score in relevance.values() if score > 0)
            highly_relevant = sum(1 for score in relevance.values() if score >= 2)
            
            print(f"   {query_id}: '{query_text}' - {relevant_docs} 相关文档 ({highly_relevant} 高度相关)")
        
        return relevance_judgments
    
    def select_diverse_queries(self, all_queries, target_count=5):
        """
        选择多样化的查询集合用于评估
        
        Args:
            all_queries (list): 所有生成的查询
            target_count (int): 要选择的查询数
            
        Returns:
            list: 选择的多样化查询
        """
        print(f"\n🎲 选择 {target_count} 个多样化查询")
        print("-" * 40)
        
        # 按特征分组查询
        by_category = defaultdict(list)
        by_length = defaultdict(list)
        by_type = defaultdict(list)
        
        for query in all_queries:
            by_category[query['expected_category']].append(query)
            by_length[query['length']].append(query)
            by_type[query['type']].append(query)
        
        selected_queries = []
        
        # 策略：选择查询以最大化多样性
        
        # 1. 确保类别多样性
        categories = list(by_category.keys())
        queries_per_category = target_count // len(categories)
        remainder = target_count % len(categories)
        
        for i, category in enumerate(categories):
            count = queries_per_category + (1 if i < remainder else 0)
            
            # 从此类别选择最佳查询 (按频率和长度)
            category_queries = sorted(by_category[category], 
                                    key=lambda x: (x['frequency_in_source'] * x['length']), 
                                    reverse=True)
            
            selected_queries.extend(category_queries[:count])
        
        # 2. 如果仍需要更多查询，按多样性选择
        while len(selected_queries) < target_count:
            remaining_queries = [q for q in all_queries if q not in selected_queries]
            if not remaining_queries:
                break
            
            # 选择具有最不同特征的查询
            best_query = max(remaining_queries, 
                           key=lambda x: (x['length'] * 2 + x['frequency_in_source']))
            selected_queries.append(best_query)
        
        # 为一致性重新分配查询ID
        for i, query in enumerate(selected_queries, 1):
            query['id'] = f'Q{i}'
        
        print("选择的查询:")
        for query in selected_queries:
            print(f"   {query['id']}: '{query['query']}' (来自 {query['source_document']}, {query['type']})")
        
        return selected_queries[:target_count]


class DocumentBasedEvaluator:
    """
    使用基于文档的查询的完整评估器
    """
    
    def __init__(self):
        self.query_generator = DocumentBasedQueryGenerator()
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
    
    def calculate_fusion_results(self, lsi_results, lm_results, bm25_results, weights=[0.25, 0.35, 0.4]):
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
        bm25_scores = normalize_scores(bm25_results)
        all_doc_indices = set(lsi_scores.keys()) | set(lm_scores.keys()) | set(bm25_scores.keys())
        
        fusion_results = []
        for doc_idx in all_doc_indices:
            lsi_score = lsi_scores.get(doc_idx, 0)
            lm_score = lm_scores.get(doc_idx, 0)
            bm25_score = bm25_scores.get(doc_idx, 0)
            fusion_score = (weights[0] * lsi_score + 
                           weights[1] * lm_score + 
                           weights[2] * bm25_score)
            fusion_results.append((doc_idx, fusion_score))
        
        fusion_results.sort(key=lambda x: x[1], reverse=True)
        return fusion_results
    
    def evaluate_single_query(self, query_info):
        """评估单个查询"""
        query_id = query_info['id']
        query_text = query_info['query']
        
        print(f"\n🔍 评估 {query_id}: '{query_text}' (来自 {query_info['source_document']})")
        
        # 从所有三种方法获取结果
        lsi_results = self.irs.information_retrieval_algorithm_1(query_text, top_k=5)
        lm_results = self.irs.information_retrieval_algorithm_2(query_text, top_k=5)
        bm25_results = self.irs.information_retrieval_algorithm_3(query_text, top_k=5)
        fusion_results = self.calculate_fusion_results(lsi_results, lm_results, bm25_results)
        
        # 获取相关性判断
        relevant_docs = self.relevance_judgments[query_id]
        
        # 计算指标
        result = {
            'query_id': query_id,
            'query_text': query_text,
            'query_type': query_info['type'],
            'source_document': query_info['source_document'],
            'expected_category': query_info['expected_category']
        }
        
        methods = [
            ('LSI', lsi_results),
            ('Language_Model', lm_results),
            ('BM25', bm25_results),
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
            
            # 显示被检索的内容
            if retrieved_docs:
                top_3 = [(self.irs.document_names[doc_id], score, relevant_docs.get(doc_id, 0)) 
                        for doc_id, score in retrieved_docs[:3]]
                print(f"      前3名: {[(name.split('_')[0], f'{score:.3f}', f'rel={rel}') for name, score, rel in top_3]}")
        
        return result
    
    def run_document_based_evaluation(self):
        """运行完整的基于文档的评估"""
        print("📄 基于文档的查询评估")
        print("DTS305TC Natural Language Processing - Part 4")
        print("=" * 80)
        
        print("🎯 方法:")
        print("  • 直接从文档内容生成查询")
        print("  • 基于内容分析的自动相关性判断")
        print("  • 确保自然查询和现实相关性")
        print("  • 有机地解决泛化问题")
        print()
        
        # 从文档生成查询
        all_queries = self.query_generator.generate_document_based_queries()
        
        if len(all_queries) < 5:
            print(f"⚠️  只生成了 {len(all_queries)} 个查询。需要至少5个。")
            return False
        
        # 选择5个多样化查询以符合课程作业要求
        self.selected_queries = self.query_generator.select_diverse_queries(all_queries, target_count=5)
        
        # 创建自动相关性判断
        self.relevance_judgments = self.query_generator.create_automatic_relevance_judgments(self.selected_queries)
        
        # 评估每个查询
        print(f"\n" + "="*80)
        print("Part 4.1-4.4: 评估结果")
        print("="*80)
        
        evaluation_results = []
        
        for query_info in self.selected_queries:
            result = self.evaluate_single_query(query_info)
            evaluation_results.append(result)
        
        # 计算汇总统计
        self.calculate_summary_statistics(evaluation_results)
        
        # 保存结果
        self.save_results(evaluation_results)
        
        print(f"\n✅ 基于文档的评估完成!")
        
        return True
    
    def calculate_summary_statistics(self, results):
        """计算和显示汇总统计"""
        print(f"\n📈 汇总统计")
        print("=" * 50)
        
        methods = ['LSI', 'Language_Model', 'BM25', 'Fusion']
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
        """将结果保存到CSV文件"""
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
                'BM25_Precision@5': result['BM25_Precision@5'],
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
                'BM25_Recall@5': result['BM25_Recall@5'],
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
                'BM25_nDCG@5': result['BM25_nDCG@5'],
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
    """运行基于文档的评估"""
    evaluator = DocumentBasedEvaluator()
    
    if evaluator.run_document_based_evaluation():
        print(f"\n🎉 基于文档的评估完成!")
        print(f"📊 从实际文档内容生成了自然查询")
        print(f"📄 自动相关性判断确保了现实评估")
        print(f"🎓 课程作业文件已在'results/'目录中准备就绪")
    else:
        print(f"\n⌐ 评估失败!")


if __name__ == "__main__":
    main()