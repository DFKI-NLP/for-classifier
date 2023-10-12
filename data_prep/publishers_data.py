import pandas as pd
import numpy as np

import string
from string import digits
import torch

def clean_publisher(publisher):
    """
    A function that gets the "publisher" string and returns it after the following pre-processing steps: 
    1. Lowercase
    2. Remove digits
    3. Remove punctuation
    4. Remove white spaces
    """
    if type(publisher) != float:
        # lowercase
        clean_publisher = publisher.lower()
        # remove digits
        remove_digits = str.maketrans('', '', digits)
        clean_publisher = clean_publisher.translate(remove_digits)
        # remove punctuation
        remove_punctuation = str.maketrans('', '', string.punctuation)
        clean_publisher = clean_publisher.translate(remove_punctuation)
        # remove white spaces
        clean_publisher = " ".join(clean_publisher.split())
        
        return clean_publisher
    else:
        return publisher
    
def get_publisher_dict(data: pd.DataFrame) -> list:

    default_list = []
    publisher_dict = {key: default_list[:] for key in set(data['clean_publisher'].values)}

    for index, row in data.iterrows():
        publisher = row['clean_publisher']
        publisher_dict[publisher].append(index)

    return publisher_dict


def main():

    data = pd.read_csv('~/documents/forc_I_dataset_FINAL_September.csv')
    # Add a column with clean (i.e. preprocessed) publishers
    data['clean_publisher'] = [clean_publisher(row['publisher']) for index, row in data.iterrows()]

    # Make a dictionary where the keys are the unique publishers 
    # and the values are a list of the indices where the publisher appears in the data
    publisher_dict = get_publisher_dict(data)

    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
    model = AutoModel.from_pretrained('malteos/scincl')

    # A dictionary of {unique_publisher: their embedding (the average of the title+abstract embedding of all their papers)}
    author_embedding_dict = get_author_embeddings(author_dict, tokenizer, model)

    # updated data with author embeddings for each row
    data = get_data_with_authors_embedding(data, author_embedding_dict)

    # get binary data (prepared in data_for_classifier.py)
    binary_data = torch.load('../../data/classifier/binary_data.pt')

    # list of author embeddings according to binary dataset, to be used as input for the binary classifier
    author_embeddings_for_binary_classifier = get_embeddings_for_binary_classifier(data, binary_data)

    torch.save(author_embeddings_for_binary_classifier, '../../data/classifier/authors_embeddings.pt')     

if __name__ == '__main__':
    main()