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

# # Download required NLTK data
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

from main_system import InformationRetrievalSystem


class DocumentBasedQueryGenerator:
    """
    Generate queries directly from document content
    """
    
    def __init__(self):
        self.irs = InformationRetrievalSystem()
        self.stop_words = set(stopwords.words('english'))
        
        # Additional stopwords for better query quality
        self.stop_words.update(['said', 'also', 'would', 'could', 'one', 'two', 
                               'first', 'last', 'new', 'old', 'good', 'well',
                               'way', 'much', 'many', 'take', 'make', 'get'])
    
    def extract_key_phrases_from_document(self, doc_content, doc_name, max_phrases=10):
        """
        Extract meaningful phrases from a document
        
        Args:
            doc_content (str): Document text
            doc_name (str): Document filename
            max_phrases (int): Maximum phrases to extract
            
        Returns:
            list: List of key phrases with metadata
        """
        # Clean and tokenize
        sentences = sent_tokenize(doc_content)
        words = word_tokenize(doc_content.lower())
        
        # Remove stopwords and short words
        meaningful_words = [word for word in words 
                           if word.isalpha() and len(word) > 3 
                           and word not in self.stop_words]
        
        # Get word frequencies
        word_freq = Counter(meaningful_words)
        
        # Extract different types of phrases
        phrases = []
        
        # 1. High-frequency meaningful words (1-word queries)
        for word, freq in word_freq.most_common(5):
            if freq >= 2:  # Word appears at least twice
                phrases.append({
                    'phrase': word,
                    'type': 'single_word',
                    'frequency': freq,
                    'source_doc': doc_name,
                    'length': 1
                })
        
        # 2. Extract noun phrases and meaningful bigrams
        pos_tagged = nltk.pos_tag(meaningful_words)
        
        # Simple noun phrase extraction (Noun + Noun, Adjective + Noun)
        for i in range(len(pos_tagged) - 1):
            word1, pos1 = pos_tagged[i]
            word2, pos2 = pos_tagged[i + 1]
            
            # Noun + Noun or Adjective + Noun
            if ((pos1.startswith('NN') and pos2.startswith('NN')) or
                (pos1.startswith('JJ') and pos2.startswith('NN'))):
                
                bigram = f"{word1} {word2}"
                
                # Check if this bigram appears in original text
                if bigram in doc_content.lower():
                    phrases.append({
                        'phrase': bigram,
                        'type': 'noun_phrase',
                        'frequency': doc_content.lower().count(bigram),
                        'source_doc': doc_name,
                        'length': 2
                    })
        
        # 3. Extract meaningful trigrams from sentences
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence_words = [word.lower() for word in word_tokenize(sentence) 
                            if word.isalpha() and word.lower() not in self.stop_words]
            
            if len(sentence_words) >= 3:
                for trigram in ngrams(sentence_words, 3):
                    trigram_phrase = ' '.join(trigram)
                    
                    # Only include if it contains meaningful content
                    if any(word in word_freq and word_freq[word] >= 2 for word in trigram):
                        phrases.append({
                            'phrase': trigram_phrase,
                            'type': 'sentence_trigram',
                            'frequency': 1,
                            'source_doc': doc_name,
                            'length': 3
                        })
        
        # 4. Extract key sentences (simplified)
        for sentence in sentences:
            words_in_sentence = [word.lower() for word in word_tokenize(sentence) 
                               if word.isalpha()]
            
            # Score sentence by frequency of important words
            sentence_score = sum(word_freq.get(word, 0) for word in words_in_sentence)
            
            if sentence_score >= 5 and len(words_in_sentence) <= 8:
                # Create phrase from important words in sentence
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
        
        # Remove duplicates and sort by relevance
        unique_phrases = {}
        for phrase_info in phrases:
            phrase = phrase_info['phrase']
            if phrase not in unique_phrases:
                unique_phrases[phrase] = phrase_info
            elif phrase_info['frequency'] > unique_phrases[phrase]['frequency']:
                unique_phrases[phrase] = phrase_info
        
        # Sort by frequency and length (prefer more specific phrases)
        sorted_phrases = sorted(unique_phrases.values(), 
                              key=lambda x: (x['frequency'] * x['length']), 
                              reverse=True)
        
        return sorted_phrases[:max_phrases]
    
    def generate_document_based_queries(self):
        """
        Generate queries from all documents in the collection
        
        Returns:
            list: List of generated queries with metadata
        """
        print("📚 GENERATING QUERIES FROM DOCUMENT CONTENT")
        print("=" * 60)
        
        # Initialize IR system to load documents
        if not self.irs.initialize_system():
            print("Failed to load documents")
            return []
        
        all_queries = []
        query_id = 1
        
        # Extract phrases from each document
        for i, (doc_name, doc_content) in enumerate(zip(self.irs.document_names, self.irs.documents)):
            print(f"\n📄 Analyzing {doc_name}:")
            
            # Extract key phrases
            phrases = self.extract_key_phrases_from_document(doc_content, doc_name, max_phrases=5)
            
            print(f"   Extracted {len(phrases)} key phrases:")
            
            # Convert phrases to queries
            for phrase_info in phrases[:3]:  # Use top 3 phrases per document
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
        
        print(f"\n✅ Generated {len(all_queries)} queries from document content")
        
        return all_queries
    
    def create_automatic_relevance_judgments(self, queries):
        """
        Automatically create relevance judgments based on query source and content analysis
        
        Args:
            queries (list): List of query dictionaries
            
        Returns:
            dict: Relevance judgments for each query
        """
        print(f"\n🎯 CREATING AUTOMATIC RELEVANCE JUDGMENTS")
        print("-" * 50)
        
        relevance_judgments = {}
        
        for query_info in queries:
            query_id = query_info['id']
            query_text = query_info['query']
            source_doc_index = query_info['source_doc_index']
            expected_category = query_info['expected_category']
            
            # Initialize relevance scores
            relevance = {}
            
            for doc_idx in range(len(self.irs.documents)):
                doc_name = self.irs.document_names[doc_idx]
                doc_content = self.irs.documents[doc_idx].lower()
                doc_category = doc_name.split('_')[0] if '_' in doc_name else 'main'
                
                # Base relevance calculation
                relevance_score = 0
                
                # 1. Source document gets highest relevance
                if doc_idx == source_doc_index:
                    relevance_score = 3  # Highly relevant
                
                # 2. Same category documents
                elif doc_category == expected_category:
                    # Check if query words appear in document
                    query_words = query_text.lower().split()
                    word_matches = sum(1 for word in query_words if word in doc_content)
                    
                    if word_matches == len(query_words):
                        relevance_score = 2  # Relevant
                    elif word_matches > 0:
                        relevance_score = 1  # Partially relevant
                    else:
                        relevance_score = 0  # Not relevant
                
                # 3. Different category documents
                else:
                    # Check for semantic relevance (query words in content)
                    query_words = query_text.lower().split()
                    word_matches = sum(1 for word in query_words if word in doc_content)
                    
                    if word_matches == len(query_words):
                        relevance_score = 1  # Partially relevant (cross-domain)
                    elif word_matches > len(query_words) / 2:
                        relevance_score = 1  # Some relevance
                    else:
                        relevance_score = 0  # Not relevant
                
                relevance[doc_idx] = relevance_score
            
            relevance_judgments[query_id] = relevance
            
            # Show relevance judgment summary
            relevant_docs = sum(1 for score in relevance.values() if score > 0)
            highly_relevant = sum(1 for score in relevance.values() if score >= 2)
            
            print(f"   {query_id}: '{query_text}' - {relevant_docs} relevant docs ({highly_relevant} highly)")
        
        return relevance_judgments
    
    def select_diverse_queries(self, all_queries, target_count=5):
        """
        Select a diverse set of queries for evaluation
        
        Args:
            all_queries (list): All generated queries
            target_count (int): Number of queries to select
            
        Returns:
            list: Selected diverse queries
        """
        print(f"\n🎲 SELECTING {target_count} DIVERSE QUERIES")
        print("-" * 40)
        
        # Group queries by characteristics
        by_category = defaultdict(list)
        by_length = defaultdict(list)
        by_type = defaultdict(list)
        
        for query in all_queries:
            by_category[query['expected_category']].append(query)
            by_length[query['length']].append(query)
            by_type[query['type']].append(query)
        
        selected_queries = []
        
        # Strategy: Select queries to maximize diversity
        
        # 1. Ensure category diversity
        categories = list(by_category.keys())
        queries_per_category = target_count // len(categories)
        remainder = target_count % len(categories)
        
        for i, category in enumerate(categories):
            count = queries_per_category + (1 if i < remainder else 0)
            
            # Select best queries from this category (by frequency and length)
            category_queries = sorted(by_category[category], 
                                    key=lambda x: (x['frequency_in_source'] * x['length']), 
                                    reverse=True)
            
            selected_queries.extend(category_queries[:count])
        
        # 2. If still need more queries, select by diversity
        while len(selected_queries) < target_count:
            remaining_queries = [q for q in all_queries if q not in selected_queries]
            if not remaining_queries:
                break
            
            # Select query with most different characteristics
            best_query = max(remaining_queries, 
                           key=lambda x: (x['length'] * 2 + x['frequency_in_source']))
            selected_queries.append(best_query)
        
        # Re-assign query IDs for consistency
        for i, query in enumerate(selected_queries, 1):
            query['id'] = f'Q{i}'
        
        print("Selected queries:")
        for query in selected_queries:
            print(f"   {query['id']}: '{query['query']}' (from {query['source_document']}, {query['type']})")
        
        return selected_queries[:target_count]


class DocumentBasedEvaluator:
    """
    Complete evaluator using document-based queries
    """
    
    def __init__(self):
        self.query_generator = DocumentBasedQueryGenerator()
        self.irs = self.query_generator.irs
        self.selected_queries = []
        self.relevance_judgments = {}
        
    def calculate_precision_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculate Precision@K"""
        if not retrieved_docs or k == 0:
            return 0.0
        
        top_k_docs = [doc_id for doc_id, _ in retrieved_docs[:k]]
        relevant_retrieved = sum(1 for doc_id in top_k_docs 
                               if relevant_docs.get(doc_id, 0) > 0)
        
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculate Recall@K"""
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
        """Calculate DCG@K"""
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
        """Calculate Ideal DCG@K"""
        sorted_relevances = sorted(relevant_docs.values(), reverse=True)
        
        idcg = 0.0
        for i, relevance in enumerate(sorted_relevances[:k]):
            if i == 0:
                idcg += relevance
            else:
                idcg += relevance / math.log2(i + 2)
        
        return idcg
    
    def calculate_ndcg_at_k(self, retrieved_docs, relevant_docs, k=5):
        """Calculate nDCG@K"""
        dcg = self.calculate_dcg_at_k(retrieved_docs, relevant_docs, k)
        idcg = self.calculate_ideal_dcg_at_k(relevant_docs, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_fusion_results(self, lsi_results, lm_results, alpha=0.6):
        """Calculate fusion results"""
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
        """Evaluate a single query"""
        query_id = query_info['id']
        query_text = query_info['query']
        
        print(f"\n🔍 Evaluating {query_id}: '{query_text}' (from {query_info['source_document']})")
        
        # Get results from all methods
        lsi_results = self.irs.information_retrieval_algorithm_1(query_text, top_k=5)
        lm_results = self.irs.information_retrieval_algorithm_2(query_text, top_k=5)
        fusion_results = self.calculate_fusion_results(lsi_results, lm_results)
        
        # Get relevance judgments
        relevant_docs = self.relevance_judgments[query_id]
        
        # Calculate metrics
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
            
            # Show what was retrieved
            if retrieved_docs:
                top_3 = [(self.irs.document_names[doc_id], score, relevant_docs.get(doc_id, 0)) 
                        for doc_id, score in retrieved_docs[:3]]
                print(f"      Top 3: {[(name.split('_')[0], f'{score:.3f}', f'rel={rel}') for name, score, rel in top_3]}")
        
        return result
    
    def run_document_based_evaluation(self):
        """Run complete document-based evaluation"""
        print("📄 DOCUMENT-BASED QUERY EVALUATION")
        print("DTS305TC Natural Language Processing - Part 4")
        print("=" * 80)
        
        print("🎯 APPROACH:")
        print("  • Generate queries directly from document content")
        print("  • Automatic relevance judgments based on content analysis")
        print("  • Ensures natural queries and realistic relevance")
        print("  • Addresses generalizability concerns organically")
        print()
        
        # Generate queries from documents
        all_queries = self.query_generator.generate_document_based_queries()
        
        if len(all_queries) < 5:
            print(f"⚠️  Only generated {len(all_queries)} queries. Need at least 5.")
            return False
        
        # Select 5 diverse queries for coursework compliance
        self.selected_queries = self.query_generator.select_diverse_queries(all_queries, target_count=5)
        
        # Create automatic relevance judgments
        self.relevance_judgments = self.query_generator.create_automatic_relevance_judgments(self.selected_queries)
        
        # Evaluate each query
        print(f"\n" + "="*80)
        print("PART 4.1-4.4: EVALUATION RESULTS")
        print("="*80)
        
        evaluation_results = []
        
        for query_info in self.selected_queries:
            result = self.evaluate_single_query(query_info)
            evaluation_results.append(result)
        
        # Calculate summary statistics
        self.calculate_summary_statistics(evaluation_results)
        
        # Save results
        self.save_results(evaluation_results)
        
        print(f"\n✅ DOCUMENT-BASED EVALUATION COMPLETED!")
        
        return True
    
    def calculate_summary_statistics(self, results):
        """Calculate and display summary statistics"""
        print(f"\n📈 SUMMARY STATISTICS")
        print("=" * 50)
        
        methods = ['LSI', 'Language_Model', 'Fusion']
        metrics = ['Precision@5', 'Recall@5', 'nDCG@5']
        
        print(f"{'Method':<15} {'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
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
        
        # Find best methods
        print(f"\n🏆 BEST PERFORMING METHODS:")
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
        """Save results to CSV files"""
        print(f"\n💾 SAVING EVALUATION RESULTS")
        print("-" * 30)
        
        os.makedirs('results', exist_ok=True)
        
        # Save precision results
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
        
        # Save recall results
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
        
        # Save nDCG results
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
        
        # Save relevance judgments
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
        
        print(f"✅ Results saved:")
        print(f"   • precision.csv")
        print(f"   • recall.csv") 
        print(f"   • nDCG.csv")
        print(f"   • relevance_judgments.csv")


def main():
    """Run document-based evaluation"""
    evaluator = DocumentBasedEvaluator()
    
    if evaluator.run_document_based_evaluation():
        print(f"\n🎉 DOCUMENT-BASED EVALUATION COMPLETED!")
        print(f"📊 Generated natural queries from actual document content")
        print(f"📄 Automatic relevance judgments ensure realistic evaluation")
        print(f"🎓 Coursework files ready in 'results/' directory")
    else:
        print(f"\n❌ Evaluation failed!")


if __name__ == "__main__":
    main()