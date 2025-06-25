# code for Task B for assignment 1

import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def file_parser(file_path):
    documents = {}
    current_id = None
    capture_text = False
    current_text = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('.I'):
                if current_id:
                    documents[current_id] = ' '.join(current_text)
                current_id = line.split()[1]
                current_text = []
                capture_text = False
            elif line.startswith('.W'):
                capture_text = True
            elif line.startswith('.X'):
                capture_text = False
            elif capture_text:
                current_text.append(line.strip())

        if current_id:
            documents[current_id] = ' '.join(current_text)

    return documents


def main():
    query_file_path = sys.argv[1]
    documents = file_parser(query_file_path)
    with open("queries_22EE30035.txt", 'w') as f:
        for doc_id, text in documents.items():
            tokens = preprocess_text(text)
            line = f"{doc_id}\t"
            for token in tokens:
                line += f"{token} "
            f.write(f"{line}\n")

if __name__=="__main__":
    main()
    


