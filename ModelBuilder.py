import os
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import json

# Define directories
processed_chunks_dir = "CleanedChunks"
collected_dir = "CollectedChunks"

# Define the filename for saving the document names
document_names_file = "document_names.json"

# Initialize variables
documents = []
document_names = []  # Keep track of document names

print("Loading documents...")
# Read processed chunks
for filename in os.listdir(collected_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(collected_dir, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())
            document_names.append(filename)  # Append the filename


# Save the document names to a JSON file
with open(document_names_file, 'w', encoding='utf-8') as f:
    json.dump(document_names, f, ensure_ascii=False, indent=4)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Building embeddings...")
# Generate embeddings for all documents
embeddings = model.encode(documents, show_progress_bar=True)
np.save('embeddings.npy', embeddings)

print("Training model...")
# Train a single BERTopic model on all documents
topic_model = BERTopic(embedding_model=model, calculate_probabilities=True, verbose=True)
topics, probabilities = topic_model.fit_transform(documents, embeddings)

# Assuming `topics` is the output from topic_model.fit_transform(documents)
with open("topic_assignments.json", 'w', encoding='utf-8') as f:
    json.dump(topics, f)


# After training, get the most representative documents for each topic
representative_docs = topic_model.get_representative_docs()

# Save the unified BERTopic model
topic_model.save("unified_bertopic_model.pkl")

print("Unified BERTopic model has been saved.")
