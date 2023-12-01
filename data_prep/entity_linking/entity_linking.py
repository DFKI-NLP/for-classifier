import os
import ast
import json
import string
import pandas as pd
import time

import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from collections import Counter
import requests
from bs4 import BeautifulSoup
from thefuzz import fuzz
from linking_APIs import EntityLinkingAPIs
import torch

import sparql_dataframe
from http.client import IncompleteRead, RemoteDisconnected
import http.client
http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'


def get_academicDisciplines():
    endpoint = "http://dbpedia.org/sparql"

    query = """
        PREFIX :     <http://dbpedia.org/resource/>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo:  <http://dbpedia.org/ontology/>

        SELECT DISTINCT ?discipline ?label ?abstract

        WHERE {
        ?subject dbo:academicDiscipline ?discipline .
        ?discipline rdfs:label ?label ;
                        dbo:abstract ?abstract .
        FILTER (LANG(?label)="en") .
        FILTER (LANG(?abstract)="en") .
        }
    """

    # use sparql_dataframe; a library that return results of SPARQL queries as pandas DataFrames
    try:
        academicDisciplines = sparql_dataframe.get(endpoint, query)
    except RemoteDisconnected:
        time.sleep(60)
        academicDisciplines = sparql_dataframe.get(endpoint, query)
    except IncompleteRead:
        time.sleep(60)
        academicDisciplines = sparql_dataframe.get(endpoint, query)

    return academicDisciplines


def divide_labels(for_list):
    complex_labels = []
    non_complex_labels = []

    for label in for_list:

        if "," in label or "and" in label:
            complex_labels.append(label)
        else:
            non_complex_labels.append(label)

    return complex_labels, non_complex_labels


def get_labels_list(complex_label_doc):  # input is the label after it gets processed with spacy. E.g. doc = nlp(label)
    """ 
    A function that gets a spacy nlp document as input (which represent the complex label)
    and returns a list of strings that contains the simplified labels
    """
    labels = []
    labels_children = []
    stop_words = stopwords.words("english")

    # for each token in the label, check if it appears in a compound or as a modifier (noun or adjective)
    for token in complex_label_doc:
        if token.dep_ == 'compound' or token.dep_ == 'amod' or token.dep_ == 'nmod':
            # If so, add it to the list of new simplified labels
            labels.append(token.text + ' ' + token.head.text)
            # Add its childred to the list of children of compounds/modifiers
            labels_children.append([str(child) for child in token.children])

    # get the original complex label text
    doc_text = complex_label_doc.text
    # remove punctuation from it
    doc_text = doc_text.translate(str.maketrans('', '', string.punctuation))

    # for each token in the label, check if it is used in one of the simplified labels
    for token in doc_text.split(" "):
        if not any(token in label for label in labels) and (token.lower() not in stop_words):
            # This is reached is the word is not contained in any of the simplified labels
            if any(token in child for child in labels_children):
                # If the token is the child of one of the compounds/modifiers
                for idx, child in enumerate(labels_children):
                    # Add it to the list of simplified labels by concatenating it to the first word in the compound/modifier
                    if token in child:
                        labels.append(token + ' ' + labels[idx].split(" ")[-1])
            else:
                # If it is not one of the children and does not appear in conjunction to any of the 
                # first words in other simplified labels, Add it as a separate token
                labels.append(token)

    return labels


def simplify_complex_labels(complex_labels):
    nlp = spacy.load("en_core_web_lg")  # lg gives better results than sm or md

    # Iterate ove the complex labels and for each one get a list of its simplified labels
    default_list = []
    complex_labels_dict = {key: default_list[:] for key in complex_labels}

    for key in complex_labels_dict.keys():
        doc = nlp(key)
        complex_labels_dict[key] = get_labels_list(doc)

    return complex_labels_dict


def get_for_values(data: pd.DataFrame) -> list:
    """
     A function that takes the FoRC dataset as input and returns a flat list of its label values.
     Additionally, the function cleans values with chars such as "/" and "&" so that they can be appropriately linked. 
     """

    # Change the label 'Other Quantitative Biology' to its parent because of its ambiguity
    for idx, row in data.iterrows():
        if row['label'] == 'Other Quantitative Biology':
            data.at[idx, 'label'] = 'Biology, Integrated Biology, Integrated Biomedical Sciences'

    # list of FoR
    for_flat_list = list(set(data['label'].values))
    for_cleaned_list = []

    for label in for_flat_list:
        cleaned_label = label
        if '/' in label:
            cleaned_label = cleaned_label.replace("/", ", ")
        if '&' in label:
            cleaned_label = cleaned_label.replace("&", "and")

        for_cleaned_list.append(cleaned_label)

    return for_cleaned_list


def main():
    data = pd.read_csv('/netscratch/abu/forc_I_dataset_FINAL_September.csv')

    # list of FoR
    for_flat_list = get_for_values(data)

    complex_labels, non_complex_labels = divide_labels(for_flat_list)
    print("Divided taxonomy labels...")

    complex_labels_dict = simplify_complex_labels(complex_labels)
    print("Simplified complex labels...")

    simplified_labels = complex_labels_dict.values()

    simplified_labels = [item for sublist in simplified_labels for item in sublist]

    # Gather all labels into one list. These will be 1. the non-complex labels and 2. the simplified complex labels
    all_labels = [*non_complex_labels, *simplified_labels]

    # Remove duplicates
    all_labels = list(set(all_labels))

    # get dataframe of dbo:academicDiscipline
    academicDisciplines = get_academicDisciplines()
    print("Got list of dbo:academicDisciplines...")

    entity_linker = EntityLinkingAPIs(academicDisciplines, all_labels, for_flat_list, complex_labels_dict)

    linked_taxonomy = entity_linker.run()
    print('Sucessfully linked taxonomy!')
    print(linked_taxonomy)

    torch.save(linked_taxonomy, '../../../data/linked_taxonomy.pt')
    print('Saved in "/data/linked_taxonomy.pt"')


if __name__ == '__main__':
    main()
