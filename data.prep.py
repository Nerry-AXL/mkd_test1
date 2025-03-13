import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone

# Load and clean dataset
df = pd.read_csv("ecommerce_data.csv")
df.drop_duplicates(inplace=True)
df.dropna(subset=["product_name", "description"], inplace=True)

# Initialize text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text data to vectors
df["embeddings"] = df["description"].apply(lambda text: model.encode(text).tolist())

# Save cleaned data
df.to_csv("cleaned_ecommerce_data.csv", index=False)
