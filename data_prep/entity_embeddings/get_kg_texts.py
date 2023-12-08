import os
from transformers import AutoTokenizer
import pandas as pd
import torch

import sparql_dataframe


def get_academicDisciplines() -> pd.DataFrame:
    """
    A function that gets the academic disciplines from DBpedia
    """
    endpoint = "http://dbpedia.org/sparql"

    query = """
        PREFIX :     <http://dbpedia.org/resource/>
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo:  <http://dbpedia.org/ontology/>

        SELECT DISTINCT ?discipline ?label ?comment

        WHERE {
        ?subject dbo:academicDiscipline ?discipline .
        ?discipline rdfs:label ?label ;
                        rdfs:comment ?comment .
        FILTER (LANG(?label)="en") .
        FILTER (LANG(?comment)="en") .
        }
    """

    academicDisciplines = sparql_dataframe.get(endpoint, query)

    return academicDisciplines


def get_label_texts(dbpedia_disciplines: pd.DataFrame, for_linking: dict, tokenizer: AutoTokenizer) -> dict:
    """
    A function that gets a pd.DataFrame of dbo:academicDisciplines, 
    a dictionary of {FoR: embedding(s) of DBpedia entit(ies)}, 
    and a Transformers tokenizer object
    and returns a dictionary of 
    { FoR: textual features of the label }

    The textual features consist of:
    1. The label from the linked DBpedia entity
    2. The comment from the linked DBpedia entity

    If the taxonomy label is linked to more than one DBpedia entity, (1) and (2) are repeated.
    Each two texts are seperated by the token [SEP]. 
    """

    default_list = []
    for_texts = {key: default_list[:] for key in for_linking}

    for for_label, linked_entities in for_linking.items():
        for_textual_info = [for_label]

        for entity, weight in linked_entities.items():
            entity_label = dbpedia_disciplines[dbpedia_disciplines['discipline'] == entity]['label'].values[0]
            for_textual_info.append(entity_label)

            entity_comment = dbpedia_disciplines[dbpedia_disciplines['discipline'] == entity]['comment'].values[0]
            for_textual_info.append(entity_comment)

            text_for_tokenizer = [text + tokenizer.sep_token for text in for_textual_info]
            text_for_tokenizer = tokenizer.cls_token + ''.join(text_for_tokenizer)[:-5]

        for_texts[for_label] = text_for_tokenizer

    return for_texts


def main():
    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')

    academicDisciplines = get_academicDisciplines()

    for_linking = torch.load('../../data/linked_taxonomy.pt')

    print("Getting textual features of taxonomy labels...")
    for_texts = get_label_texts(academicDisciplines, for_linking, tokenizer)

    torch.save(for_texts, '../../data/taxonomy_texts.pt')
    print('Saved in "/data/taxonomy_texts.pt"')


if __name__ == '__main__':
    main()
