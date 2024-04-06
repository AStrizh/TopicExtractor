import os
import pickle

from bertopic import BERTopic

books_directory = './ProcessedBooks'
book_files = os.listdir(books_directory)
documents = []

# Reads all the books into memory (cannot be avoided)
for book_file in book_files:
    try:
        with open(os.path.join(books_directory, book_file), encoding='utf-8') as f:
            documents.append(f.read())
    except UnicodeDecodeError:
        print(f"Skipping {book_file} due to UnicodeDecodeError")

# Initialize BERTopic
# embedding_model - have not seen other types besides sentence-transformers
# calculate_probabilities - allows finer data points but takes considerably more time
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", calculate_probabilities=False, verbose=True)

# Fit the model to your documents
topics, probabilities = topic_model.fit_transform(documents)

# # Save the trained model for use later.  Uses a library called pickle
# with open('../BookTrainer/bertopic_model.pkl', 'wb') as f:
#     pickle.dump(topic_model, f)

# Get an overview of the topics
topic_info = topic_model.get_topic_info()
print(topic_info)
