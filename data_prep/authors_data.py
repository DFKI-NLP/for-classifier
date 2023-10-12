import pandas as pd
import numpy as np
import ast

from nameparser import HumanName
import torch

from transformers import AutoTokenizer, AutoModel


def parse_author(name:str) -> list:
    """
    Parse author names and return them in a list according to the template:
    [last name, first + middle name, title, suffix]
    """
    if name.endswith('et al'):
        name = name[:-6]
        return [name, ' ', ' ', ' ']
    
    last = HumanName(name).last
    first = HumanName(name).first
    middle = HumanName(name).middle
    title = HumanName(name).title
    suffix = HumanName(name).suffix
    
    if last == '':
        last = ' '
    if first == '':
        first = ' '

    return [HumanName(name).last, HumanName(name).first + ' ' + HumanName(name).middle, HumanName(name).title,
            HumanName(name).suffix]

def parse_authors(orkg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the orkg_df and adds the column 'authors_parsed' with the same authors parsed in a list.
    """
    orkg_df['authors_parsed'] = ''
    for index, row in orkg_df.iterrows():
        
        if not pd.isna(row['author']):
            
            if row['author'].startswith('['):
                author_list = ast.literal_eval(row['author'])
                authors_list_parsed = []
                for author in author_list:
                    authors_list_parsed.append(parse_author(author))
                orkg_df.at[index, 'authors_parsed'] = authors_list_parsed
                
            elif ',' in row['author']:
                author_list = row['author'].split(',')
                authors_list_parsed = []
                for author in author_list:
                    authors_list_parsed.append(parse_author(author))
                orkg_df.at[index, 'authors_parsed'] = authors_list_parsed
                
            else:
                orkg_df.at[index, 'authors_parsed'] = parse_author(row['author'])
                
    return orkg_df

def change_author_format(authors_parsed):
    
    if type(authors_parsed) == float:
        return np.nan
    if authors_parsed == '':
        return np.nan
    
    authors_new_format = []
    if type(authors_parsed[0]) == list: #this means there is more than one author i.e. authors_parsed is a list of lists
        for author in authors_parsed:
            authors_new_format.append(author[0].lower() + author[1][0].lower()) # Last Name + First letter of First Name
    else: # if there is only one author i.e. authors_parsed is one list
        authors_new_format = authors_parsed[0].lower() + authors_parsed[1][0].lower()
    
    return authors_new_format

def get_authors_dict(data: pd.DataFrame) -> dict:

    # step 1: make a flat list of all authors and then a set of all unique authors
    all_authors_new_format = data['authors_new_format'].values
    all_authors_new_format = ['nan' if type(item) == float else item for item in all_authors_new_format]

    # get a set of all unique authors
    all_authors = []
    for item in all_authors_new_format:
        if type(item) == str:
            all_authors.append(item)
        elif type(item) == list:
            for subitem in item:
                all_authors.append(subitem)

    # Step 2: make a dictionary where the keys are the unique authors 
    # and the values are a list of the indices where the authors appears in the data
    default_list = []
    author_dict = {key: default_list[:] for key in set(all_authors)}

    for index, row in data.iterrows():
        authors = row['authors_new_format']
        if type(authors) == list: # i.e. there is more than one author
            for author in authors: 
                author_dict[author].append(index)
        elif type(authors) == str: # i.e. there is only one author
            author_dict[authors].append(index)
        else: # i.e. authors is nan
            author_dict['nan'].append(index)

    return author_dict

def get_author_embeddings(data: pd.DataFrame, 
                          author_dict: dict, 
                          tokenizer: AutoTokenizer, 
                          model: AutoModel) -> dict:

    default_list = []
    author_embedding_dict = {key: default_list[:] for key in set(author_dict)}

    for author, indices in author_dict.items():
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
        author_embedding_dict[author] = sum(indices_embeddings)/len(indices_embeddings)

    return author_embedding_dict

def get_data_with_authors_embedding(data: pd.DataFrame, 
                                    author_embedding_dict: dict) -> pd.DataFrame:

    article_author_embeddings = []

    for idx, row in data.iterrows():
        row_authors = row['authors_new_format']
        row_authors_embedding = []
        for author, embedding in author_embedding_dict.items():
            if author in row_authors:
                row_authors_embedding.append(embedding)
        # the author embedding of each row is the average of all of its authors' embeddings
        article_author_embeddings.append(sum(row_authors_embedding)/len(row_authors_embedding))

    data['authors_embedding'] = article_author_embeddings

    return data

def get_embeddings_for_binary_classifier(data, binary_data):

    author_embeddings_for_binary_classifier = []

    for data_point in binary_data:
        row_idx = data_point[0][0]
        author_embeddings_for_binary_classifier.append(data['authors_embedding'][row_idx])
    
    return author_embeddings_for_binary_classifier


def main():

    data = pd.read_csv('~/documents/forc_I_dataset_FINAL_September.csv')
    data = parse_authors(data)

    # add column to data with the following structure: ['Author1LastName FirstLetterOfFirstName.', 
    #                                                   'Author2LastName FirstLetterOfFirstName.', ..... ]
    data['authors_new_format'] = [change_author_format(row['authors_parsed']) for index, row in data.iterrows()]

    # This is the created dictionary of {unique_author: a list the papers that are written by them}
    # The idea is to take this and for each author, its embedding would be the average SciNCL embedding of all the papers 
    # (the embedding of each paper = its title+abstract embedding)
    author_dict = get_authors_dict(data)

    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
    model = AutoModel.from_pretrained('malteos/scincl')

    # A dictionary of {unique_author: their embedding (the average of the title+abstract embedding of all their papers)}
    author_embedding_dict = get_author_embeddings(data, author_dict, tokenizer, model)

    # updated data with author embeddings for each row
    data = get_data_with_authors_embedding(data, author_embedding_dict)

    # get binary data (prepared in data_for_classifier.py)
    binary_data = torch.load('../../data/classifier/binary_data.pt')

    # list of author embeddings according to binary dataset, to be used as input for the binary classifier
    author_embeddings_for_binary_classifier = get_embeddings_for_binary_classifier(data, binary_data)

    torch.save(author_embeddings_for_binary_classifier, '../../data/classifier/authors_embeddings.pt')     

if __name__ == '__main__':
    main()