import faiss
import numpy as np
import json
import torch
from transformers import AutoModel, AutoTokenizer

class FAISSDatabase:
    def __init__(self, json_file):
        self.json_file = json_file
        self.products = self.load_products()
        self.tokenizer, self.model = self.load_model()
        self.index, self.product_vectors = self.create_index()

    def load_products(self):
        """Load product data from the JSON file."""
        with open(self.json_file, 'r') as file:
            products = json.load(file)
        return products

    def load_model(self):
        """Load the tokenizer and model for generating embeddings."""
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        return tokenizer, model

    def create_index(self):
        """Create a FAISS index for the product vectors."""
        product_names = [product['name'] for product in self.products]
        product_vectors = self.get_embeddings(product_names)

        # Create the FAISS index
        index = faiss.IndexFlatL2(384)  # L2 distance metric; 384 is the embedding dimension
        index.add(product_vectors)  # Add vectors to the index

        return index, product_vectors

    def get_embeddings(self, product_names):
        """Generate embeddings for product names."""
        embeddings = []
        for name in product_names:
            inputs = self.tokenizer(name, return_tensors='pt')
            with torch.no_grad():
                embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
                embeddings.append(embedding.numpy())
        return np.vstack(embeddings).astype('float32')

    def search(self, query_vector, k=5):
        """Search for the k nearest products to the query vector."""
        distances, indices = self.index.search(query_vector, k)
        return distances, indices

    def get_product_recommendations(self, query_vector, k=5):
        """Get product recommendations based on the query vector."""
        distances, indices = self.search(query_vector, k)
        recommendations = []
        for idx in indices[0]:
            recommendations.append(self.products[idx])
        return recommendations
