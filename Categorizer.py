from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import json
import os

# Define the path to your processed book chunks and the model
chunks_dir = "DRACEXAMPLE"
model_path = "unified_bertopic_model.pkl"
embeddings_path = "embeddings.npy"

# Load document names
with open("document_names.json", 'r', encoding='utf-8') as f:
    document_names = json.load(f)

# Load the saved topic assignments
with open("topic_assignments.json", 'r', encoding='utf-8') as f:
    saved_topics = json.load(f)


# Load the pre-trained BERTopic model
topic_model = BERTopic.load(model_path)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Collect all chunk texts into a list
new_book_texts = []
for filename in sorted(os.listdir(chunks_dir)):
    if filename.endswith(".txt"):
        with open(os.path.join(chunks_dir, filename), 'r', encoding='utf-8') as file:
            chunk_text = file.read()
        new_book_texts.append(chunk_text)

# Use the transform method directly with texts to infer topics
new_book_topics = topic_model.transform(new_book_texts)

# Output topics for the new book's chunks
# print("Inferred topics for the new book's chunks:", new_book_topics)


# Assuming `new_book_topics` is the variable holding your output
assigned_topics, probabilities = new_book_topics

# Get the indices of the documents that are most representative of the top topic
rep_doc_indices = topic_model.get_representative_docs(new_book_topics[0])

# Iterate over the unique topics in `assigned_topics` (excluding -1 if you wish)
unique_topics = set(assigned_topics)
if -1 in unique_topics:
    unique_topics.remove(-1)  # Optional: remove if you don't want the outlier topic

# For each unique topic, get and print the top words and their scores
for topic in unique_topics:
    topic_words = topic_model.get_topic(topic)
    topic_words_list = [word[0] for word in topic_words]
    topic_words_str = ', '.join(topic_words_list)
    print(f"Topic {topic} top words: {topic_words_str}\n")


for topic in unique_topics:
    # Find indices of documents in this topic
    doc_indices = [i for i, t in enumerate(saved_topics) if t == topic]
    print(f"Documents for Topic {topic}:")
    for idx in doc_indices:
        print(f"\t{document_names[idx]}")

print("--------------------------------------------------------------------------------------------")

