import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch

from zipfile import ZipFile
import fileinput

def get_list_of_entities(linked_taxonomy: dict) -> list:
    """
    A function that gets the linked taxonomy dictionary as an input and outputs a list of all unique DBpedia entities
    """

    entities = []

    for key, value in linked_taxonomy.items():

        for key2, value2 in value.items():
            entities.append(key2)

    entities = list(set(entities))

    return entities

def get_pretrained_embeddings(linked_taxonomy: dict, input_file: str) -> dict:

    entities = []

    for key, value in for_linking.items():
        entities.append(list(value.keys()))

    flat_entities = [item for sublist in entities for item in sublist]
    flat_entities = list(set(flat_entities))

    entities_embeddings = []

    for i in range(100000):
        for lines in range(i*1000, 1000*(i+1)):
            line = input_file.readline()
            line = line.split()
            if line[0] in flat_entities:
                entities_embeddings.append(line)

    entity_embeddings_dict = {}

    for index, element in enumerate(flat_entities):
        entity_embeddings_dict[element] = []

    for entity in entities_embeddings:
        embedding_str = entity[1:]
        embedding_fl = [float(i) for i in embedding_str]
        entity_embeddings_dict[entity[0]] = embedding_fl

    return entity_embeddings_dict


def get_taxonomy_embeddings(dbpedia_embeddings: pd.DataFrame, linked_taxonomy: dict) -> dict:
    # Create an empty dictionary with taxonomy labels as keys
    default_list = []
    taxonomy_embeddings = {key: default_list[:] for key in linked_taxonomy}

    for for_label, linked_entities in linked_taxonomy.items():

        # instantiate an empty list to include embeddings of all connected DBpedia entities to the ORKG for_label
        for_entities_embeddings = []
        # instantiate a weight sum in order to later use it for weighted average calculations
        weights_sum = 0

        # iterate over all DBpedia entities linked to the ORKG label, add their embeddings to the list multiplied by
        # their weight update the weight sum
        for entity, weight in linked_entities.items():
            for_entities_embeddings.append(
                weight * (dbpedia_embeddings[dbpedia_embeddings['entity'] == entity]['embedding'].values[0]))
            weights_sum = weights_sum + weight

        # sum all the embeddings that are connected to the same ORKG for_label (this already includes the
        # multiplication with their weight)
        for_entities_sum = sum(for_entities_embeddings)

        # save the weighted average in a dict
        taxonomy_embeddings[for_label] = for_entities_sum / weights_sum

    return taxonomy_embeddings


def main():
    linked_taxonomy = torch.load('../../data/linked_taxonomy.pt')

    entities = get_list_of_entities(linked_taxonomy)

    print("Getting pre-trained embeddings...")
    # Get dataset of pretrained embeddings from Zenodo: https://zenodo.org/records/6384728
    zip = ZipFile('../../data/embeddings.zip')
    zip.extractall('../../data/')
    zip.close()

    print("Getting KG embeddings...")

    # Read data
    input_file = open('../../data/vectors.txt','r')

    taxonomy_embeddings = get_pretrained_embeddings(linked_taxonomy, input_file)
    print('Sucessfully embedded taxonomy labels!')

    torch.save(taxonomy_embeddings, '../../data/taxonomy_embeddings_pretrained.pt')
    print('Saved in "/data/taxonomy_embeddings_pretrained.pt"')


if __name__ == '__main__':
    main()
