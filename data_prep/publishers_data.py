import pandas as pd
import numpy as np

import string
from string import digits
import torch

from transformers import AutoTokenizer, AutoModel


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


def get_publisher_embeddings(data: pd.DataFrame,
                             author_dict: dict,
                             tokenizer: AutoTokenizer,
                             model: AutoModel) -> dict:
    default_list = []
    publisher_embedding_dict = {key: default_list[:] for key in set(author_dict)}

    for publisher, indices in author_dict.items():
        indices_embeddings = []
        for index in indices:
            title = data['title'][index]
            abstact = None
            if type(data['abstract'][index]) == str:
                abstract = data['abstract'][index]
            title_abs = title + tokenizer.sep_token + abstract
            inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = model(**inputs)
            embedding = result.last_hidden_state[:, 0, :]
            indices_embeddings.append(embedding)

        # the embedding representing each author is the average embedding of all the publications they wrote
        publisher_embedding_dict[publisher] = sum(indices_embeddings) / len(indices_embeddings)

    return publisher_embedding_dict


def get_data_with_publisher_embedding(data: pd.DataFrame,
                                      publisher_embedding_dict: dict) -> pd.DataFrame:
    publisher_author_embeddings = []

    for idx, row in data.iterrows():
        row_publisher = row['clean_publisher']
        row_publisher_embedding = []
        for author, embedding in publisher_embedding_dict.items():
            if author in row_publisher:
                row_publisher_embedding.append(embedding)
        # the author embedding of each row is the average of all of its authors' embeddings
        publisher_author_embeddings.append(sum(row_publisher_embedding) / len(row_publisher_embedding))

    data['publishers_embedding'] = publisher_author_embeddings

    return data


def get_embeddings_for_binary_classifier(data, binary_data):
    publisher_embeddings_for_binary_classifier = []

    for data_point in binary_data:
        row_idx = data_point[0][0]
        publisher_embeddings_for_binary_classifier.append(data['publishers_embedding'][row_idx])

    return publisher_embeddings_for_binary_classifier


def main():
    data = pd.read_csv('~/documents/forc_I_dataset_FINAL_September.csv')
    # Add a column with clean (i.e. preprocessed) publishers
    data['clean_publisher'] = [clean_publisher(row['publisher']) for index, row in data.iterrows()]
    print('Cleaned publisher data...')

    # Make a dictionary where the keys are the unique publishers 
    # and the values are a list of the indices where the publisher appears in the data
    publisher_dict = get_publisher_dict(data)
    print(f'Got list of {len(publisher_dict)} unique publishers...')

    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
    model = AutoModel.from_pretrained('malteos/scincl')

    # A dictionary of {unique_publisher: their embedding (the average of the title+abstract embedding of all their
    # papers)}

    publisher_embedding_dict = get_publisher_embeddings(data, publisher_dict, tokenizer, model)
    print('Got embeddings of unique publishers...')

    # updated data with author embeddings for each row
    data = get_data_with_publisher_embedding(data, publisher_embedding_dict)

    # get binary data (prepared in data_for_classifier.py)
    binary_data = torch.load('data/classifier/binary_data.pt')

    print('Getting publisher embeddings for binary classifier...')
    # list of author embeddings according to binary dataset, to be used as input for the binary classifier
    publisher_embeddings_for_binary_classifier = get_embeddings_for_binary_classifier(data, binary_data)

    torch.save(publisher_embeddings_for_binary_classifier, 'data/classifier/publisher_embeddings.pt')
    print('Successfully got embeddings and saved under data/classifier/publisher_embeddings.pt...')


if __name__ == '__main__':
    main()
