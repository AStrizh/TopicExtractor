import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def preprocess_text(document):
    # Tokenize, remove stop words, and lemmatize the document
    doc = nlp(document)
    cleaned_tokens = []
    for token in doc:
        # Check if the token is not a stop word and is not a punctuation mark
        if token.text.lower() not in STOP_WORDS and token.text not in ['``', "''", '`', ',', ';']:
            # Append the lemma of the token
            cleaned_tokens.append(token.lemma_)
    # Join the cleaned tokens to form a cleaned document
    cleaned_document = " ".join(cleaned_tokens)
    # Remove extra whitespaces and return the cleaned document
    return " ".join(cleaned_document.split())

def process_file(input_file, output_directory):
    # Read the content of the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        document = f.read()
    # Preprocess the text
    cleaned_text = preprocess_text(document)
    # Extract the filename from the input file path
    filename = os.path.basename(input_file)
    # Define the output file path
    output_file = os.path.join(output_directory, filename)
    # Write the cleaned text to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

def clean_files_in_directory(input_directory, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Get the list of input files in the input directory
    input_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith('.txt')]
    # Process the input files using multithreading
    with ThreadPoolExecutor() as executor:
        # Use the partial function to fix the output_directory argument for the process_file function
        executor.map(partial(process_file, output_directory=output_directory), input_files)

if __name__ == '__main__':
    input_directory = 'CleanChunks'
    output_directory = 'DRACEXAMPLE'
    clean_files_in_directory(input_directory, output_directory)
