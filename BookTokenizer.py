import os
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from concurrent.futures import ProcessPoolExecutor



# Ensure necessary NLTK data packages are downloaded before processing in parallel
def download_nltk_resources():
    nltk_packages = ['punkt', 'wordnet']
    for package in nltk_packages:
        nltk.download(package)

download_nltk_resources()

def clean_and_split_text(filename, output_folder, chunk_size=1000, lemmatize=True):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    text = re.sub(r'\n+', '\n', text).strip()
    sentences = sent_tokenize(text)

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        sentences = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(sentence)]) for sentence in sentences]

    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    for i, chunk in enumerate(chunks, start=1):
        chunk_filename = f"{base_filename}_chunk_{i}.txt"
        with open(os.path.join(output_folder, chunk_filename), 'w', encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)

def process_files_parallel(input_folder, output_folder, chunk_size=1000, lemmatize=True):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(clean_and_split_text, filename, output_folder, chunk_size, lemmatize) for filename in filenames]
        for future in futures:
            future.result()

if __name__ == "__main__":
    # Example usage
    # input_folder = 'ProcessedBooks'
    # output_folder = 'ProcessedChunks'
    input_folder = 'CleanCategorize'
    output_folder = 'CleanChunks'
    process_files_parallel(input_folder, output_folder)
