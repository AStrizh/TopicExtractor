import os
from chardet import detect


def detect_and_convert_encoding(original_file_path, clean_file_path=None):
    """
    Detects a file's encoding and converts it to UTF-8.

    :param original_file_path: Path to the original file.
    :param clean_file_path: Path to save the cleaned file. If None, overwrites the original.
    """
    # Read the file's binary data
    with open(original_file_path, 'rb') as file:
        binary_data = file.read()

    # Detect encoding
    detected_encoding = detect(binary_data)['encoding']
    print(f"Detected encoding for {original_file_path}: {detected_encoding}")

    # Decode the binary data using the detected encoding
    try:
        text_data = binary_data.decode(detected_encoding)
    except UnicodeDecodeError:
        print(f"Failed to decode {original_file_path} using {detected_encoding}.")
        return

    # Re-encode the text to UTF-8
    utf8_data = text_data.encode('utf-8')

    # Write the UTF-8 encoded text to a new file or overwrite the original
    output_file_path = clean_file_path if clean_file_path else original_file_path
    with open(output_file_path, 'wb') as clean_file:
        clean_file.write(utf8_data)
    print(f"Cleaned file saved to {output_file_path}")


def clean_directory(input_directory, output_directory=None):
    """
    Cleans all text files in a directory to ensure they are UTF-8 encoded.

    :param input_directory: Directory containing the original files.
    :param output_directory: Directory to save cleaned files. If None, overwrites the originals.
    """
    # Ensure the output directory exists
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            original_file_path = os.path.join(input_directory, filename)
            clean_file_path = os.path.join(output_directory, filename) if output_directory else None

            # Clean each file
            detect_and_convert_encoding(original_file_path, clean_file_path)


# Example usage
# input_dir = 'ProcessedBooks'
# output_dir = 'ProcessedBooksClean'  # Set to None to overwrite originals

input_dir = 'DirtyCategorize'
output_dir = 'CleanCategorize'  # Set to None to overwrite originals
clean_directory(input_dir, output_dir)
