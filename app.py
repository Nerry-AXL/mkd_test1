import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import easyocr
from PIL import Image
import io
import cv2
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
API_KEY = "pcsk_47pLVx_C2oDHPCxoUWz7vLtVA6ureN96YEyNTesQXHmDMPja5p6m5rPGAu9uhBwZCVKsUF"

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize Pinecone
if not API_KEY:
    raise ValueError("Missing Pinecone API Key! Set it in the environment variables.")

pc = Pinecone(api_key=API_KEY)

# Connect to or create Pinecone index
index_name = "product-data"
existing_indexes = [index["name"] for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )

index = pc.Index(index_name)

# Initialize sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# ✅ Function to convert query into a vector
def get_query_vector(query):
    """Converts a natural language query into an embedding vector."""
    try:
        vector = embedding_model.encode(query).tolist()
        if not vector or not all(isinstance(x, float) for x in vector):
            raise ValueError("Invalid vector generated.")
        return vector
    except Exception as e:
        logging.error(f"Error generating query vector: {e}")
        raise


# ✅ Function to generate product descriptions
def generate_description(product_name):
    """Generates a short product description using NLP."""
    try:
        prompt = f"Describe the product '{product_name}' in a simple and informative way."
        description = embedding_model.encode(prompt)  # Simulating text generation
        logging.info(f"Generated description for '{product_name}'")
        return f"{product_name} is a high-quality product designed for optimal performance and customer satisfaction."
    except Exception as e:
        logging.error(f"Error generating product description: {e}")
        return "No description available."


# ✅ Search function with description generation
def search_products(query, top_k=5):
    """Search for products based on a query string."""
    try:
        query_vector = get_query_vector(query)

        # Query Pinecone with metadata retrieval
        response_data = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        matches = response_data.get("matches", [])

        # Extract all relevant metadata fields and generate descriptions if missing
        products = []
        for match in matches:
            metadata = match.get("metadata", {})
            product_name = metadata.get("name", "Unknown Product")
            description = metadata.get("Description", "")

            # If no description is available, generate one
            if not description:
                description = generate_description(product_name)

            products.append({
                "StockCode": metadata.get("StockCode", "N/A"),
                "Description": description,
                "UnitPrice": metadata.get("UnitPrice", 0.0),
                "Country": metadata.get("Country", "Unknown"),
            })

        return products
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        return []


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/product-recommendation', methods=['GET', 'POST'])
def product_recommendation():
    """Fetches product recommendations based on a natural language query."""
    query = request.form.get('query', '')

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    try:
        products = search_products(query)
        response_text = f"Recommendations for '{query}':"
    except Exception as e:
        logging.error(f"Pinecone query failed: {e}")
        return jsonify({"error": f"Failed to fetch recommendations: {str(e)}"}), 500

    return jsonify({"products": products, "response": response_text})


@app.route('/ocr-query', methods=['GET', 'POST'])
def ocr_query():
    """Handles OCR-based queries for product recommendations."""
    response_text = None
    products = []

    if request.method == 'POST':
        if 'image_data' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files['image_data']

        if image_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            # ✅ Read image file correctly
            image_bytes = image_file.read()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)  # Convert to OpenCV image

            # ✅ Perform OCR using EasyOCR
            result = reader.readtext(image, detail=0)  # detail=0 returns only text

            # ✅ Extract text
            query_text = " ".join(result).strip()

            if not query_text:
                return jsonify({"error": "No readable text found in image"}), 400

            # ✅ Process the extracted text as a query
            products = search_products(query_text)

            # ✅ Generate response
            response_text = f"Recognized text: '{query_text}'"
            if products:
                response_text = f"Search results for '{query_text}':"
            else:
                response_text = "Sorry, no matching products were found."

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

    return render_template('ocr_query.html', response=response_text, products=products)


@app.route('/image-product-search', methods=['GET', 'POST'])
def image_product_search():
    """Processes an uploaded product image and provides recommendations."""
    if 'product_image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    product_image = request.files['product_image']

    try:
        # Placeholder for actual image processing logic
        products = [
            {"name": "Smartphone", "price": "$800"},
            {"name": "Smartwatch", "price": "$200"}
        ]
        response_text = "Detected product and suggested related items."
    except Exception as e:
        logging.error(f"Image product search failed: {e}")
        return jsonify({"error": "Image processing error"}), 500

    return jsonify({"response": response_text, "products": products})


@app.route('/text-query', methods=['GET', 'POST'])
def text_query():
    """Handles text-based queries for product recommendations."""
    response_text = None
    products = []

    if request.method == 'POST':
        query = request.form.get('query', "").strip()

        if not query:
            return jsonify({"error": "Query cannot be empty"}), 400

        products = search_products(query)

        # ✅ Generate response including all metadata
        if products:
            response_text = f"Search results for '{query}':"
        else:
            response_text = "Sorry, no matching products were found."

    return render_template('text_query.html', response=response_text, products=products)


if __name__ == '__main__':
    app.run(debug=True)
