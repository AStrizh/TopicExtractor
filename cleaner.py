import os
import re

books_directory = './FreshBook'
output_directory = './DirtyCategorize'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Regex pattern for start and end markers, capturing the title directly from the start marker
start_pattern = re.compile(r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK (.+?) \*\*\*', re.IGNORECASE)
end_pattern = re.compile(r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK', re.IGNORECASE)

def clean_filename(title):
    """Clean and format the title to a valid filename."""
    filename = title.strip().replace(' ', '_').replace(':', '').replace('"', '').replace('/', '').replace('\\', '').replace('?', '').replace('*', '')
    invalid_chars = '<>|\n'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename[:200] + '.txt'  # Limit filename length

for book_file in os.listdir(books_directory):
    book_path = os.path.join(books_directory, book_file)
    try:
        with open(book_path, encoding='utf-8') as f:
            content = f.read()

        # Extract the title from the start marker
        start_match = start_pattern.search(content)
        if start_match:
            title = start_match.group(2)  # The title captured from the start pattern
            content_start_index = start_match.end()
        else:
            print(f"Start marker not found in {book_file}, skipping.")
            continue

        # Find the end marker and extract content up to that point
        end_match = end_pattern.search(content, content_start_index)
        if end_match:
            content = content[content_start_index:end_match.start()]
        else:
            print(f"End marker not found in {book_file}, using content from start marker to end of file.")
            content = content[content_start_index:]

        # Save the cleaned content to a new file with the title as its name
        new_filename = clean_filename(title)
        output_path = os.path.join(output_directory, new_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Processed and saved: {new_filename}")

    except UnicodeDecodeError as e:
        print(f"Skipping {book_file} due to UnicodeDecodeError: {e}")

    except Exception as e:
        print(f"An error occurred while processing {book_file}: {e}")
