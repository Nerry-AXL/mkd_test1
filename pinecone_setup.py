import pinecone

# Initialize Pinecone
pinecone.init(api_key="pcsk_47pLVx_C2oDHPCxoUWz7vLtVA6ureN96YEyNTesQXHmDMPja5p6m5rPGAu9uhBwZCVKsUF", environment="us-west1-gcp")

# Create index (if not exists)
index_name = "ecommerce-data"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")

index = pinecone.Index(index_name)

# Load cleaned data
import pandas as pd
df = pd.read_csv("cleaned_ecommerce_data.csv")

# Upload product vectors
for i, row in df.iterrows():
    index.upsert([(str(i), row["embeddings"], {"product_name": row["product_name"]})])

print("✅ Vector database populated successfully!")
