# eCommerce Chatbot

This is an eCommerce chatbot designed to assist users with shopping-related inquiries. The chatbot leverages the Gemini API for generating responses and integrates a vector database for efficient product information retrieval.

## Features

- Product Information: Provides detailed information about products, including descriptions, prices, and specifications.
- Order Tracking: Assists with order-related queries such as tracking and order status.
- Recommendations: Suggests similar products based on user preferences.
- User-Friendly Interface: Built using Streamlit for easy interaction.

## Technologies Used

- Streamlit: For building the user interface.
- Google Gemini API: For natural language understanding and response generation.
- FAISS: For storing and retrieving product information efficiently.
- Hugging Face Transformers: For generating embeddings from product descriptions.

## Requirements

To run this project, you need the following:

- Python 3.9 or higher
- Necessary Python packages listed in `requirements.txt`

## Setting Up the Environment

1. Create a virtual environment:

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

2. Install the required packages

   pip install -r requirements.txt

3. Set up your Gemini API key in your environment:
    
   export GEMINI_API_KEY="your_api_key"  # On Windows use `set GEMINI_API_KEY=your_api_key`

4. Data generation
   
   python query.py

   (This script generates a diverse set of product queries for the e-commerce chatbot. This data generation process helps create a more realistic and varied dataset for testing and demonstrating the chatbot's capabilities.)

5. Running the Chatbot
   
   streamlit run app.py


##Usage
You can ask the chatbot questions related to product information, order tracking, and recommendations. The chatbot will respond within the eCommerce context and avoid casual conversation.


