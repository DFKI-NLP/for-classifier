import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from cachetools import MRUCache
import numpy as np

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

import torch

RANDOM_STATE = 22

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


def get_taxonomy_embeddings(dbpedia_embeddings: pd.DataFrame, linked_taxonomy: dict) -> dict:

	# Create an empty dictionary with taxonomy labels as keys
	default_list = []
	taxonomy_embeddings = {key: default_list[:] for key in linked_taxonomy}

	for for_label, linked_entities in linked_taxonomy.items():
    
    	# instantialte an empty list to include embeddings of all connected DBpedia entities to the ORKG for_label
    	for_entities_embeddings = []
    	# instantiate a weight sum in order to later use it for weighted average calculations
    	weights_sum = 0
    
    	# iterate over all DBpedia entities linked to the ORKG label, add their embeddings to the list multiplied by their weight
    	# update the weight sum
    	for entity, weight in linked_entities.items():
        	taxonomy_embeddings.append(weight*(dbpedia_embeddings[dbpedia_embeddings['FoR']==entity]['embeddings'].values[0]))
        	weights_sum = weights_sum + weight
    
    	# sum all the embeddings that are connected to the same ORKG for_label (this already includes the multiplication with their weight)
    	for_entities_sum = sum(for_entities_embeddings)
    
    	# save the weighted average in a dict
    	taxonomy_embeddings[for_label] = for_entities_sum/weights_sum


def main():

	linked_taxonomy = torch.load('/data/linked_taxonomy.pt')

	entities = get_list_of_entities(linked_taxonomy)

	# Initiate the RDF2VecTransformer
	transformer = RDF2VecTransformer(
    # Use one worker threads for Word2Vec to ensure random determinism.
    Word2Vec(workers=1),
    # Extract a maximum of 10 walks of a maximum depth of 4 for each entity
    # using two processes and use a random state to ensure that the same walks
    # are generated for the entities.
    walkers=[RandomWalker(10, 500, n_jobs=2, random_state=RANDOM_STATE)],
    verbose=1,
	)

	# Get embeddings for each entity
	embeddings_pyrdf2vec, _ = transformer.fit_transform(
    # Defined the DBpedia endpoint server, as well as a set of predicates to
    # exclude from this KG.
    KG("https://dbpedia.org/sparql"),
    [entity for entity in entities]
	)

	dbpedia_embeddings = pd.DataFrame(list(zip(entities, embeddings_pyrdf2vec)),
              columns=['FoR','embedding'])

	taxonomy_embeddings = get_taxonomy_embeddings(dbpedia_embeddings, linked_taxonomy)

	torch.save(taxonomy_embeddings, '/data/taxonomy_embeddings.pt')

if __name__ == '__main__':
    main()
