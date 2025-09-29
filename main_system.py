import os
import re
import csv
import math
import numpy as np
from collections import defaultdict, Counter

from algorithm1_lsi import LSIAlgorithm
from algorithm2_language_model import LanguageModelAlgorithm


# ====================== TEXT PREPROCESSOR ======================
class StandaloneTextPreprocessor:
    """
    Completely standalone text preprocessor
    """
    def __init__(self):
        # Comprehensive English stopwords list
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
        """Simple tokenization using regex"""
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split on whitespace and filter empty strings
        tokens = [token.strip().lower() for token in text.split() if token.strip()]
        return tokens
    
    def simple_stem(self, word):
        """Very basic stemming"""
        # Define common suffixes to remove
        suffixes = [
            'ings', 'ing', 'edly', 'ied', 'ies', 'ed', 'es', 's',
            'erly', 'er', 'est', 'ly', 'tion', 'ness', 'ment'
        ]
        
        word = word.lower()
        original_word = word
        
        # Try to remove suffixes
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                stemmed = word[:-len(suffix)]
                # Basic rules to avoid overstemming
                if len(stemmed) >= 3:
                    word = stemmed
                    break
        
        return word
    
    def preprocess_text(self, text):
        """Complete preprocessing pipeline"""
        # lowercase and tokenize
        tokens = self.simple_tokenize(text)
        
        # Remove stopwords and very short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        # Apply simple stemming
        stemmed_tokens = [self.simple_stem(token) for token in tokens]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tokens = []
        for token in stemmed_tokens:
            if token not in seen:
                unique_tokens.append(token)
                seen.add(token)
        
        return ' '.join(unique_tokens)


## main system
class InformationRetrievalSystem:
    def __init__(self):
        
        self.documents = []
        self.document_names = []
        self.processed_documents = []
        self.preprocessor = StandaloneTextPreprocessor()
        
        # use the imported algorithms
        self.lsi_algorithm = LSIAlgorithm(k_dimensions=10)
        self.language_model = LanguageModelAlgorithm(lambda_param=0.7)
        
        self.system_ready = False
            
    def database_input_function(self, folder_path="dataset"):
        """read the documents from the folder"""
        print(f"\nLoading documents from {folder_path}...")
        
        try:
            self.documents = []
            self.document_names = []
            
            if not os.path.exists(folder_path):
                print(f"Error: '{folder_path}' folder not found")
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
                            print(f"Failed to load {filename}: {e}")
            
            if total_files == 0:
                print("No documents found")
                return False
            
            print(f"Loaded {total_files} documents")
            return True
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False
    
    def text_preprocessing_function(self):
        """preprocess the documents"""        
        try:
            self.processed_documents = []
            
            for i, doc in enumerate(self.documents):
                processed = self.preprocessor.preprocess_text(doc)
                self.processed_documents.append(processed)
            
            # Stats
            all_words = ' '.join(self.processed_documents).split()
            print(f"Processing complete: {len(set(all_words))} unique terms")
            
            return True
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return False
    
    def information_retrieval_algorithm_1(self, query, top_k=5):
        """LSI algorithm"""
        try:
            processed_query = self.preprocessor.preprocess_text(query)
            if not processed_query.strip():
                return []
            
            results = self.lsi_algorithm.retrieve_documents(processed_query, top_k)
            return results
            
        except Exception as e:
            print(f"LSI algorithm error: {e}")
            return []
    
    def information_retrieval_algorithm_2(self, query, top_k=5):
        """Language Model algorithm"""
        try:
            processed_query = self.preprocessor.preprocess_text(query)
            if not processed_query.strip():
                return []
            
            results = self.language_model.retrieve_documents(processed_query, top_k)
            return results
            
        except Exception as e:
            print(f"Language Model algorithm error: {e}")
            return []
    
    def output_function(self, query, lsi_results, lm_results):
        """output the results of LSI and Language Model"""
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        print("LSI Algorithm Results:")
        for i, (doc_idx, score) in enumerate(lsi_results[:5]):  # Ensure top-5
            print(f"{i+1}. {self.document_names[doc_idx]} (Score: {score:.3f})")
        
        print("\nLanguage Model Algorithm Results:")
        for i, (doc_idx, score) in enumerate(lm_results[:5]):  # Ensure top-5
            print(f"{i+1}. {self.document_names[doc_idx]} (Score: {score:.3f})")
    
    def fusion_output_function(self, query, lsi_results, lm_results, alpha=0.6):
        """weighted fusion"""
        try:
            # Normalize scores to [0,1] range
            def normalize_scores(results):
                if not results:
                    return {}
                scores = [s for _, s in results]
                min_s, max_s = min(scores), max(scores)
                if max_s == min_s:
                    return {idx: 0.5 for idx, _ in results}
                return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}
            
            lsi_scores = normalize_scores(lsi_results)
            lm_scores = normalize_scores(lm_results)
            all_docs = set(lsi_scores.keys()) | set(lm_scores.keys())
            
            # Weighted fusion: alpha * LSI + (1-alpha) * Language Model
            fusion_results = []
            for doc_idx in all_docs:
                lsi_score = lsi_scores.get(doc_idx, 0)
                lm_score = lm_scores.get(doc_idx, 0)
                fusion_score = alpha * lsi_score + (1 - alpha) * lm_score
                fusion_results.append((doc_idx, fusion_score))
            
            fusion_results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nWeighted Fusion Results (α={alpha}):")
            for i, (doc_idx, score) in enumerate(fusion_results[:5]):
                print(f"{i+1}. {self.document_names[doc_idx]} (Fusion Score: {score:.3f})")
            
            return fusion_results[:5]
            
        except Exception as e:
            print(f"Fusion error: {e}")
            return []
    
    def user_input_function(self):
        """user input"""
        print("\nEnter queries (type 'quit' to exit):")
        
        while True:
            try:
                query = input("> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif not query:
                    continue
                
                # Process query using both algorithms
                lsi_results = self.information_retrieval_algorithm_1(query)
                lm_results = self.information_retrieval_algorithm_2(query)
                
                # Display results
                self.output_function(query, lsi_results, lm_results)
                self.fusion_output_function(query, lsi_results, lm_results)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
    
    def initialize_system(self, dataset_folder="dataset"):
        """Initialize the complete information retrieval system"""
        
        # Load documents
        if not self.database_input_function(dataset_folder):
            return False
        
        # Preprocess documents
        if not self.text_preprocessing_function():
            return False
        
        # Build algorithm indexes
        print("Building search indexes...")
        
        if not self.lsi_algorithm.build_index(self.processed_documents):
            print("Failed to build LSI index")
            return False
        
        if not self.language_model.build_index(self.processed_documents):
            print("Failed to build Language Model index")
            return False
        
        self.system_ready = True
        print("System ready!\n")
        
        # Display algorithm information
        print("Algorithm Information:")
        lsi_info = self.lsi_algorithm.get_algorithm_info()
        lm_info = self.language_model.get_algorithm_info()
        print(f"  • {lsi_info['name']}: {lsi_info['dimensions']} dimensions")
        print(f"  • {lm_info['name']}: λ={lm_info['lambda_parameter']}")
        
        return True


def main():
    """Main function to run the Information Retrieval System"""
    irs = InformationRetrievalSystem()
    
    if irs.initialize_system():
        irs.user_input_function()
    else:
        print("System initialization failed")


if __name__ == "__main__":
    main()