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
api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")

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
        "You are an eCommerce chatbot designed to assist users with their shopping-related inquiries. Your primary functions include:\n"
        "1. Providing detailed information about products, including descriptions, prices, availability, and specifications.\n"
        "2. Assisting with order-related queries such as tracking, order status, and returns.\n"
        "3. Making recommendations based on user preferences and past purchases.\n"
        "4. Answering questions related to payment options, shipping methods, and customer service.\n"
        "\n"
        "Ensure that your responses are strictly within the eCommerce domain. If a user asks something outside of this context, politely respond with: 'I'm here to assist with eCommerce-related queries. Please ask about products, orders, or services related to shopping.'\n"
        "\n"
        "Avoid providing information unrelated to shopping, products, or services. Do not engage in casual conversation or answer general knowledge questions unless they are part of the eCommerce domain.\n"
        "\n"
        "Stick to the eCommerce context, and ensure that all responses are informative, relevant, and helpful for the user's shopping experience."
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
                if "description" in user_input.lower():
                    response_text = f"**Description:** {product_details['description']}"
                elif "price" in user_input.lower():
                    response_text = f"**Price:** *${product_details['price']}*"
                elif "tracking status" in user_input.lower():
                    response_text = (
                        f"**Order Status for Order Number: {order_number}**\n"
                        f"- **Tracking Status:** *{product_details['tracking_status']}*\n"
                        f"- **Estimated Delivery:** *{product_details.get('estimated_delivery', 'N/A')}*\n"
                        f"- **Carrier:** *{product_details.get('carrier', 'N/A')}*\n"
                    )
                else:
                    response_text = (
                        f"**Product Details for Order Number: {order_number}**\n"
                        f"- **Product Name:** *{product_details['name']}*\n"
                        f"- **Description:** *{product_details['description']}*\n"
                        f"- **Price:** *${product_details['price']}*\n"
                        f"- **Tracking Status:** *{product_details['tracking_status']}*\n"
                        f"- **Similar Recommendations:**\n- " + "\n- ".join(product_details['similar_recommendations'])
                    )
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