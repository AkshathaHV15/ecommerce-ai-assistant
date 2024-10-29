import os
import json
import streamlit as st
import google.generativeai as genai
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from faiss_vector import FAISSDatabase
import re

# Set your Gemini API key
api_key = os.environ.get("GEMINI_API_KEY", "API_KEY")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Load the model and tokenizer for embedding generation
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

# Create the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the generative model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Start a chat session
chat_session = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [
                "You are an eCommerce chatbot designed to assist users with their shopping-related inquiries."
            ]
        },
        {
            "role": "model",
            "parts": [
                "I'm here to assist with eCommerce-related queries. Please ask about products, orders, or services related to shopping!"
            ],
        },
    ]
)

# Load the FAISS index and product data
faiss_db = FAISSDatabase('data/products.json')

st.title("eCommerce Chatbot")

# Function to generate embeddings
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.numpy()

def parse_order_number(user_input):
    """Extracts the order number (assuming format like 'ORD001') from the user's input."""
    match = re.search(r'ORD\d+', user_input)  # Adjust pattern if order number format changes
    return match.group(0) if match else None

def get_product_details(order_number):
    """Look up the product details for the order number in the product data."""
    for product in faiss_db.products:
        if product.get('order_number') == order_number:
            return product  # Return the entire product object
    return None

# Function to construct prompt based on user query type
def construct_prompt(user_input, product_details):
    return f"""
    You are an eCommerce chatbot. Answer the user's query based on the following product details only. 

    Product Details:
    - **Product Name:** {product_details['name']}
    - **Description:** {product_details['description']}
    - **Price:** ${product_details['price']}
    - **Tracking Status:** {product_details['tracking_status']}
    - **Estimated Delivery:** {product_details.get('estimated_delivery', 'N/A')}
    - **Carrier:** {product_details.get('carrier', 'N/A')}
    - **Similar Recommendations:** {", ".join(product_details['similar_recommendations'])}

    User Query: "{user_input}"

    Instructions:
    1. If the user asks for the **price**, respond with "The price of this item is ${product_details['price']}".
    2. If the user asks for a **description**, respond with: "{product_details['description']}".
    3. If the user asks for the **product name**, respond with: "{product_details['name']}".
    4. For **order tracking**, give the tracking status: "{product_details['tracking_status']}".
    5. For **estimated delivery**, state the estimated delivery date if available.
    6. If the user’s query is not specific to any of the above, generate a relevant and informative response within the eCommerce context.
    7. If the query is outside of eCommerce topics, respond politely with: "I'm here to assist with eCommerce-related queries. Please ask about products, orders, or services related to shopping."
    """

# Create a form for user input and button
with st.form("chat_form"):
    user_input = st.text_input("You:", key="user_input_input")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Check for order number and retrieve product details
        order_number = parse_order_number(user_input)
        response_text = ""

        if order_number:
            product_details = get_product_details(order_number)
            if product_details:
                # Construct response based on user query
                prompt = construct_prompt(user_input, product_details)
                response = model.generate_content(prompt)
                response_text = response.candidates[0].content.parts[0].text
            else:
                response_text = "Order number not found."
        else:
            # Process general queries with the chatbot
            response = chat_session.send_message(user_input)  # Ensure response is from the current input
            response_text = response.text

        # Display the user input and assistant response directly
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response_text)
