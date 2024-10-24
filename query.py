import os
import json
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate product queries
def generate_product_queries(num_queries):
    product_queries = []
    
    # Example product attributes to vary
    product_names = [
        "Wireless Router", "Bluetooth Speaker", "Smartphone", "Laptop", 
        "Smartwatch", "Tablet", "Camera", "Wireless Headphones"
    ]
    
    product_descriptions = [
        "High-speed wireless router with extended range.", 
        "Portable Bluetooth speaker with deep bass.", 
        "Latest smartphone with cutting-edge features.", 
        "Sleek and powerful laptop for all your needs.", 
        "Stylish smartwatch with fitness tracking.", 
        "Compact tablet for on-the-go entertainment.", 
        "Professional camera for stunning photos.", 
        "Noise-cancelling wireless headphones."
    ]

    for i in range(num_queries):
        # Randomly select product attributes
        name = random.choice(product_names)
        description = random.choice(product_descriptions)
        price = round(random.uniform(50, 300), 2)  # Random price between 50 and 300
        order_number = f"ORD{str(i + 1).zfill(3)}"
        tracking_status = random.choice(["Shipped", "Delivered", "Processing", "Returned"])
        similar_recommendations = random.sample(product_names, 3)  # Randomly select 3 similar products
        
        # Create product JSON object
        product_query = {
            "name": name,
            "description": description,
            "price": price,
            "order_number": order_number,
            "tracking_status": tracking_status,
            "similar_recommendations": similar_recommendations
        }
        
        product_queries.append(product_query)

    return product_queries

# Generate 50 product queries
product_queries = generate_product_queries(50)

# Save the generated queries to data/product.json
output_file = 'data/products.json'
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Create the directory if it doesn't exist

with open(output_file, 'w') as f:
    json.dump(product_queries, f, indent=4)

print(f"Saved {len(product_queries)} product queries to {output_file}.")
