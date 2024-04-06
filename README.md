# TopicExtractor
FGCU Hackathon 2024 project repository.

This project aims to create a Python Web application which will allow a user to extract topics from books and categorize them with related texts.  
The Hackathon project will be split into two parts. One part will be the scrips needed to pre-process, clean the data, and train the model.   
The second part will be the web application that will use a trained model.  
The books themselves are only used as a proof of concept. The wider goal of the project is to be able to classify different types of documents, including regulatory data.

This is a continuation of the project that was started on Dirk Johnson's GitHub.
https://github.com/DirkJohnson4/Three-Amigos-Hackathon-Project

## BERTopic 

BERTopic, developed by Maarten Grootendorst, is used as the basis of the project to extract topics from textual data.
https://github.com/MaartenGr/BERTopic

At this time we have run a demo to classify randomly selected English language texts from Project Guttenberg.

## Installation

The goal of this project is to produce a program that would be simple to build and install. 
However, as this project is currently in the design phase, the set up procedure is complicated.

1. Gather textual data. For the demo, we are using English text from Project Guttenberg.  
   This guide was used for retreiving the texts:  
   https://www.exratione.com/2014/11/how-to-politely-download-all-english-language-text-format-files-from-project-gutenberg/  
   However, as it was written in 2014 we reduced the delay between downloads as we expect the organization has updated its infrastructure over the past ten years.
   The unaltered bash script is added to this project as "download_guttenberg.sh"

3. pyTorch with NVIDIA CUDA
  This command downloads pyTorch and the 12.1 release of CUDA (latest at this time)

   pip3 install torch torchvision torchaudio --index-url `https://download.pytorch.org/whl/cu121`

4. Install BERTopic  
   pip install bertopic

Additional libraries installed with pip:
datasets, nltk

### Authors
Aleksandr Strizhevskiy  
