import os
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# Define directories
processed_chunks_dir = "ProcessedChunks"

# Initialize variables
documents = []

print("Loading documents...")
# Read processed chunks
for filename in os.listdir(processed_chunks_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(processed_chunks_dir, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Building embeddings...")
# Generate embeddings for all documents
embeddings = model.encode(documents, show_progress_bar=True)
np.save('embeddings.npy', embeddings)

print("Training model...")
# Train a single BERTopic model on all documents
topic_model = BERTopic()
topics, probabilities = topic_model.fit_transform(documents, embeddings)

# Save the unified BERTopic model
topic_model.save("unified_bertopic_model.pkl")

print("Unified BERTopic model has been saved.")
