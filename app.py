from flask import Flask, request, jsonify, render_template
import easyocr
import cv2
import numpy as np
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize Pinecone
API_KEY = "pcsk_47pLVx_C2oDHPCxoUWz7vLtVA6ureN96YEyNTesQXHmDMPja5p6m5rPGAu9uhBwZCVKsUF"
pc = Pinecone(api_key=API_KEY)

# Connect to the Pinecone index
index_name = "ecommerce-data"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(index_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/product-recommendation', methods=['GET', 'POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    if request.method == 'POST':
        query = request.form.get('query', '')

        # Query the Pinecone index
        try:
            response = index.query(queries=[query], top_k=5)
            products = [{"name": match["metadata"]["name"], "price": match["metadata"]["price"]} for match in response["matches"]]
            response_text = f"Based on your query '{query}', here are some recommended products."
        except Exception as e:
            products = []
            response_text = f"An error occurred while processing your query: {str(e)}"

        return jsonify({"products": products, "response": response_text})
    else:
        return render_template('product_recommendation_form.html')

@app.route('/ocr-query', methods=['GET', 'POST'])
def ocr_query():
    response = None
    products = []
    if request.method == 'POST':
        image_file = request.files.get('image_data')
        if not image_file:
            return jsonify({"error": "No image uploaded"}), 400

        # Convert image to numpy array
        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Extract text using EasyOCR
        result = reader.readtext(image)
        extracted_text = " ".join([text[1] for text in result]) if result else "No text found"

        # Simulated response (should be replaced with NLP logic)
        products = [{"name": "Notebook", "price": "$5"}] if "notebook" in extracted_text.lower() else []
        response = f"Extracted text: {extracted_text}"

    return render_template('ocr_query.html', response=response, products=products)

@app.route('/image-product-search', methods=['GET', 'POST'])
def image_product_search():
    response = None
    products = []
    if request.method == 'POST':
        product_image = request.files.get('product_image')
        if not product_image:
            return jsonify({"error": "No image uploaded"}), 400

        # Process the image (Placeholder logic, replace with actual ML model)
        products = [
            {"name": "Smartphone", "price": "$800"},
            {"name": "Smartwatch", "price": "$200"}
        ]
        response = "Detected product and suggested related items."

    return render_template('image_product_search.html', response=response, products=products)

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

@app.route('/text-query', methods=['GET', 'POST'])
def text_query():
    response = None
    products = []
    if request.method == 'POST':
        query = request.form['query']
        # Process the query and get the response and products
        response = "Search Results"
        products = [{'name': 'Product 1', 'price': '$10'}, {'name': 'Product 2', 'price': '$20'}]
    return render_template('text_query.html', response=response, products=products)

if __name__ == '__main__':
    app.run(debug=True)
