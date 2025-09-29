import math
import numpy as np
from collections import defaultdict, Counter


class LanguageModelAlgorithm:
    def __init__(self, lambda_param=0.7):
        """
        Initialize Language Model Algorithm with Jelinek-Mercer smoothing
        
        Args:
            lambda_param (float): Smoothing parameter (0 < lambda < 1)
                                 Higher values give more weight to document model
        """
        self.lambda_param = lambda_param
        self.document_language_models = []
        self.collection_language_model = {}
        self.documents = []
        self.vocabulary = set()
        self.collection_word_count = 0
        self.is_trained = False
        
    def build_index(self, processed_documents):
        """
        Build language model index from processed documents
        
        Args:
            processed_documents (list): List of preprocessed document strings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Building Language Model index with λ={self.lambda_param}...")
            
            if len(processed_documents) == 0:
                print("Error: No documents provided for indexing!")
                return False
            
            self.documents = processed_documents
            self.document_language_models = []
            collection_word_counts = defaultdict(int)
            self.collection_word_count = 0
            self.vocabulary = set()
            
            # count words in each document and in the collection
            doc_word_counts = []
            
            for doc_idx, doc in enumerate(processed_documents):
                words = doc.split()
                word_count = Counter(words)   
                doc_word_counts.append(word_count)   ## doc_word_counts is a list of word counts for each document
                
                # Update collection statistics
                for word, count in word_count.items():
                    collection_word_counts[word] += count
                    self.collection_word_count += count
                    self.vocabulary.add(word)  ## get all words and the number of words in the collection
                
                print(f"Document {doc_idx + 1}: {len(words)} tokens, {len(word_count)} unique words")
            
            # Build collection language model P(w|C)
            self.collection_language_model = {}
            for word, count in collection_word_counts.items():
                self.collection_language_model[word] = count / self.collection_word_count
            
            # Build document language models with smoothing
            for doc_idx, word_count in enumerate(doc_word_counts):
                doc_length = sum(word_count.values())
                doc_language_model = {}
                
                # For each word in vocabulary, calculate smoothed probability
                for word in self.vocabulary:
                    # Maximum likelihood estimate P(w|d)
                    ml_prob = word_count.get(word, 0) / doc_length if doc_length > 0 else 0
                    
                    # Collection probability P(w|C)
                    collection_prob = self.collection_language_model.get(word, 1e-10)
                    
                    # Jelinek-Mercer smoothing
                    smoothed_prob = (self.lambda_param * ml_prob + 
                                   (1 - self.lambda_param) * collection_prob)
                    
                    doc_language_model[word] = smoothed_prob
                
                self.document_language_models.append(doc_language_model)
            
            print(f"Vocabulary size: {len(self.vocabulary)}")
            print(f"Collection word count: {self.collection_word_count}")
            print(f"Number of document models: {len(self.document_language_models)}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error building Language Model index: {e}")
            return False
    
    def calculate_query_likelihood(self, query_text, doc_index):
        """
        Calculate the likelihood of generating the query from a document
        
        Args:
            query_text (str): User query
            doc_index (int): Document index
            
        Returns:
            float: Log likelihood score
        """
        try:
            if not self.is_trained or doc_index >= len(self.document_language_models):
                return float('-inf')
            
            query_words = query_text.split()
            if not query_words:
                return float('-inf')
            
            doc_model = self.document_language_models[doc_index]
            log_likelihood = 0.0
            
            # Calculate log P(q|d) = sum(log P(qi|d))
            for word in query_words:
                if word in doc_model:
                    prob = doc_model[word]
                else:
                    # Use collection probability for unseen words
                    prob = self.collection_language_model.get(word, 1e-10)
                
                # Add log probability (avoid log(0))
                if prob > 0:
                    log_likelihood += math.log(prob)
                else:
                    log_likelihood += math.log(1e-10)  # Very small probability
            
            return log_likelihood
            
        except Exception as e:
            print(f"Error calculating query likelihood: {e}")
            return float('-inf')
    
    def retrieve_documents(self, query_text, top_k=5):
        """
        Retrieve top-k most likely documents using Language Model
        
        Args:
            query_text (str): User query
            top_k (int): Number of documents to retrieve
            
        Returns:
            list: List of tuples (doc_index, log_likelihood_score)
        """
        try:
            if not self.is_trained:
                print("Error: Language Model not trained!")
                return []
            
            if not query_text.strip():
                print("Error: Empty query!")
                return []
            
            print(f"Processing query: '{query_text}'")
            
            # Calculate likelihood for each document
            doc_scores = []
            for doc_idx in range(len(self.document_language_models)):
                score = self.calculate_query_likelihood(query_text, doc_idx)
                doc_scores.append((doc_idx, score))
            
            # Sort by likelihood (descending order)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            return doc_scores[:top_k]
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_algorithm_info(self):
        """
        Get information about the Language Model algorithm
        
        Returns:
            dict: Algorithm information
        """
        return {
            'name': 'Query Likelihood Language Model',
            'type': 'Statistical Retrieval',
            'lambda_parameter': self.lambda_param,
            'smoothing_method': 'Jelinek-Mercer',
            'is_trained': self.is_trained,
            'vocabulary_size': len(self.vocabulary),
            'description': 'Estimates probability of generating query from each document language model'
        }
    
    def explain_score(self, query_text, doc_index):
        """
        Explain the language model score for a specific document
        
        Args:
            query_text (str): User query
            doc_index (int): Document index
            
        Returns:
            dict: Score explanation
        """
        try:
            if not self.is_trained or doc_index >= len(self.document_language_models):
                return None
            
            query_words = query_text.split()
            doc_model = self.document_language_models[doc_index]
            
            word_scores = []
            total_log_likelihood = 0.0
            
            for word in query_words:
                if word in doc_model:
                    prob = doc_model[word]
                    source = "document + collection (smoothed)"
                else:
                    prob = self.collection_language_model.get(word, 1e-10)
                    source = "collection only"
                
                log_prob = math.log(max(prob, 1e-10))
                total_log_likelihood += log_prob
                
                word_scores.append({
                    'word': word,
                    'probability': prob,
                    'log_probability': log_prob,
                    'source': source
                })
            
            explanation = {
                'total_log_likelihood': total_log_likelihood,
                'query_length': len(query_words),
                'average_log_prob': total_log_likelihood / len(query_words) if query_words else 0,
                'word_scores': word_scores,
                'lambda_parameter': self.lambda_param
            }
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining score: {e}")
            return None
    
    def get_document_statistics(self, doc_index):
        """
        Get statistics for a specific document
        
        Args:
            doc_index (int): Document index
            
        Returns:
            dict: Document statistics
        """
        try:
            if not self.is_trained or doc_index >= len(self.documents):
                return None
            
            doc = self.documents[doc_index]
            words = doc.split()
            word_count = Counter(words)
            
            # Calculate perplexity on the document itself
            doc_model = self.document_language_models[doc_index]
            log_prob_sum = sum(math.log(doc_model.get(word, 1e-10)) for word in words)
            perplexity = math.exp(-log_prob_sum / len(words)) if words else float('inf')
            
            return {
                'document_index': doc_index,
                'document_length': len(words),
                'unique_words': len(word_count),
                'vocabulary_coverage': len(set(words) & self.vocabulary) / len(self.vocabulary) if self.vocabulary else 0,
                'perplexity': perplexity,
                'top_words': word_count.most_common(5)
            }
            
        except Exception as e:
            print(f"Error getting document statistics: {e}")
            return None


# Test function for development
if __name__ == "__main__":
    # Test the Language Model algorithm
    test_docs = [
        "british hurdler sarah claxton confident medal european indoor championships madrid",
        "athlete smash british record hurdles season set new mark second",
        "claxton won national hurdles title past three year struggled international",
        "scotland born athlete equal fifth fastest time world year",
        "birmingham grand prix claxton left european medal favourite russian trailing"
    ]
    
    lm = LanguageModelAlgorithm(lambda_param=0.7)
    
    # Build index
    success = lm.build_index(test_docs)
    if success:
        print("\nLanguage Model built successfully!")
        
        # Test query
        query = "british athlete hurdles"
        results = lm.retrieve_documents(query, top_k=3)
        
        print(f"\nQuery: '{query}'")
        print("Language Model Results:")
        for i, (doc_idx, score) in enumerate(results):
            print(f"{i+1}. Document {doc_idx}: {score:.4f}")
            print(f"   Content: {test_docs[doc_idx][:100]}...")
            
        # Explain first result
        if results:
            explanation = lm.explain_score(query, results[0][0])
            print(f"\nScore explanation for top result:")
            print(f"Total log-likelihood: {explanation['total_log_likelihood']:.4f}")
            print(f"Average log-probability: {explanation['average_log_prob']:.4f}")