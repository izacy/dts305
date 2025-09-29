import numpy as np
import math
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LSIAlgorithm:
    def __init__(self, k_dimensions=80):
        """
        Initialize LSI Algorithm
        
        Args:
            k_dimensions (int): Number of semantic dimensions for SVD
        """
        self.k_dimensions = k_dimensions  ## k_dimensions <= number of documents
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.doc_vectors_semantic = None
        self.vocabulary = None
        self.is_trained = False
        
    def build_index(self, processed_documents):
        """
        Build LSI index from processed documents
        
        Args:
            processed_documents (list): List of preprocessed document strings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Building LSI index with {self.k_dimensions} semantic dimensions...")
            
            if len(processed_documents) == 0:
                print("Error: No documents provided for indexing!")
                return False
            
            # Build TF-IDF matrix
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,  # Limit vocabulary size
                min_df=1,           # Minimum document frequency
                max_df=0.8,         # Maximum document frequency
                ngram_range=(1, 1)  # Use unigrams only
            )
            
            # Transform documents to TF-IDF matrix
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_documents)
            self.vocabulary = self.tfidf_vectorizer.get_feature_names_out()
            
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            print(f"Vocabulary size: {len(self.vocabulary)}")
            
            # Apply SVD for dimensionality reduction
            # Adjust k_dimensions if it's larger than matrix dimensions
            max_components = min(self.tfidf_matrix.shape[0]-1, 
                               self.tfidf_matrix.shape[1], 
                               self.k_dimensions)
            
            self.svd_model = TruncatedSVD(
                n_components=max_components,
                random_state=42,
                algorithm='randomized'
            )
            
            # Step 3: Transform documents to semantic space
            self.doc_vectors_semantic = self.svd_model.fit_transform(self.tfidf_matrix)
            
            print(f"SVD decomposition completed with {max_components} components")
            print(f"Semantic document vectors shape: {self.doc_vectors_semantic.shape}")
            print(f"Explained variance ratio: {sum(self.svd_model.explained_variance_ratio_):.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            print(f"Error building LSI index: {e}")
            return False
    
    def query_processing(self, query_text):
        """
        Process query and transform to semantic space
        
        Args:
            query_text (str): User query string
            
        Returns:
            np.ndarray: Query vector in semantic space, or None if error
        """
        try:
            if not self.is_trained:
                print("Error: LSI model not trained yet!")
                return None
            
            # Transform query using the same TF-IDF vectorizer
            query_tfidf = self.tfidf_vectorizer.transform([query_text])
            
            # Project query to semantic space
            query_semantic = self.svd_model.transform(query_tfidf)
            
            return query_semantic[0]  # Return 1D array
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return None
    
    def retrieve_documents(self, query_text, top_k=5):
        """
        Retrieve top-k most similar documents using LSI
        
        Args:
            query_text (str): User query
            top_k (int): Number of documents to retrieve
            
        Returns:
            list: List of tuples (doc_index, similarity_score)
        """
        try:
            if not self.is_trained:
                print("Error: LSI model not trained!")
                return []
            
            # Process query to semantic space
            query_semantic = self.query_processing(query_text)
            if query_semantic is None:
                return []
            
            # Calculate cosine similarities
            similarities = cosine_similarity(
                query_semantic.reshape(1, -1), 
                self.doc_vectors_semantic
            )[0]
            
            # Get top-k document indices and scores
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                results.append((idx, float(score)))
                
            return results
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def get_algorithm_info(self):
        """
        Get information about the LSI algorithm
        
        Returns:
            dict: Algorithm information
        """
        return {
            'name': 'Latent Semantic Indexing (LSI)',
            'type': 'Semantic Retrieval',
            'dimensions': self.k_dimensions,
            'is_trained': self.is_trained,
            'vocabulary_size': len(self.vocabulary) if self.vocabulary is not None else 0,
            'description': 'Uses SVD to discover latent semantic relationships between terms and documents'
        }
    
    def explain_similarity(self, query_text, doc_index):
        """
        Explain why a document is similar to the query (for debugging)
        
        Args:
            query_text (str): User query
            doc_index (int): Document index
            
        Returns:
            dict: Explanation of similarity
        """
        try:
            if not self.is_trained or doc_index >= len(self.doc_vectors_semantic):
                return None
            
            query_semantic = self.query_processing(query_text)
            if query_semantic is None:
                return None
            
            doc_semantic = self.doc_vectors_semantic[doc_index]
            
            # Calculate similarity
            similarity = cosine_similarity(
                query_semantic.reshape(1, -1),
                doc_semantic.reshape(1, -1)
            )[0][0]
            
            # Get top contributing semantic dimensions
            contribution = query_semantic * doc_semantic
            top_dims = np.argsort(np.abs(contribution))[::-1][:5]
            
            explanation = {
                'similarity_score': float(similarity),
                'top_semantic_dimensions': [
                    {
                        'dimension': int(dim),
                        'contribution': float(contribution[dim]),
                        'query_weight': float(query_semantic[dim]),
                        'doc_weight': float(doc_semantic[dim])
                    }
                    for dim in top_dims
                ]
            }
            
            return explanation
            
        except Exception as e:
            print(f"Error explaining similarity: {e}")
            return None


# Test function for development
if __name__ == "__main__":
    # Test the LSI algorithm
    test_docs = [
        "british hurdler sarah claxton confident medal european indoor championships madrid",
        "athlete smash british record hurdles season set new mark second",
        "claxton won national hurdles title past three year struggled international",
        "scotland born athlete equal fifth fastest time world year",
        "birmingham grand prix claxton left european medal favourite russian trailing"
    ]
    
    lsi = LSIAlgorithm(k_dimensions=3)
    
    # Build index
    success = lsi.build_index(test_docs)
    if success:
        print("\nLSI Index built successfully!")
        
        # Test query
        query = "british athlete hurdles"
        results = lsi.retrieve_documents(query, top_k=3)
        
        print(f"\nQuery: '{query}'")
        print("LSI Results:")
        for i, (doc_idx, score) in enumerate(results):
            print(f"{i+1}. Document {doc_idx}: {score:.4f}")
            print(f"   Content: {test_docs[doc_idx][:100]}...")