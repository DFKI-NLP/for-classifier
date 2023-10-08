import os
from transformers import AutoTokenizer
import pandas as pd
import torch

from data_prep.utils import get_academicDisciplines


def get_label_texts(dbpedia_disciplines: pd.DataFrame, for_linking: dict, tokenizer: AutoTokenizer) -> dict:
    """
    A function that gets a pd.DataFrame of dbo:academicDisciplines, 
    a dictionary of {FoR: embedding(s) of DBpedia entit(ies)}, 
    and a Transformers tokenizer object
    and returns a dictionary of 
    { FoR: textual features of the label }

    The textual features consist of: 
    1. The label from the ORKG taxonomy 
    2. The label from the linked DBpedia entity
    3. The abstract from the linked DBpedia entity

    If the taxonomy label is linked to more than one DBpedia entity, (2) and (3) are repeated. 
    Each two texts are seperated by the token [SEP]. 
    """

    default_list = []
    for_texts = {key: default_list[:] for key in for_linking}

    for for_label, linked_entities in for_linking.items():
        for_textual_info = []
    
    for_textual_info.append(for_label)
    
    for entity, weight in linked_entities.items():
        entity_label = dbpedia_disciplines[dbpedia_disciplines['discipline']==entity]['label'].values[0]
        for_textual_info.append(entity_label)
        
        entity_abstract = dbpedia_disciplines[dbpedia_disciplines['discipline']==entity]['abstract'].values[0]
        for_textual_info.append(entity_abstract)
        
        text_for_tokenizer = [text + tokenizer.sep_token for text in for_textual_info]
        text_for_tokenizer = ''.join(text_for_tokenizer)[:-5]
    
    for_texts[for_label] = text_for_tokenizer

def main():
    
    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')

    academicDisciplines = get_academicDisciplines()

    for_linking = torch.load('../../data/taxonomy_embeddings.pt')

    print("Getting textual features of taxonomy labels...")
    for_texts = get_label_texts(academicDisciplines, for_linking, tokenizer)

    torch.save(for_texts, '../../data/taxonomy_texts.pt')
    print('Saved in "/data/taxonomy_texts.pt"')


if __name__ == '__main__':
    main()