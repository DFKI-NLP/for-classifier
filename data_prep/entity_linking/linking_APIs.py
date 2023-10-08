import os
import ast
import json
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from collections import Counter
import requests as req
from bs4 import BeautifulSoup
from thefuzz import fuzz

import pandas as pd

from wat import WATAnnotation, wat_entity_linking, get_wat_annotations

class EntityLinkingAPIs:
    """
    A class that is used to link taxonomy labels to entities in DBpedia using 3 different APIs:
    1. WAT API (https://sobigdata.d4science.org/web/tagme/wat-api)
    2. DBpedia Lookup (https://github.com/dbpedia/dbpedia-lookup)
    3. FALCON API (https://labs.tib.eu/falcon/falcon2/api-use)
    """

    def __init__(self, academicDisciplines, labels):

        self.academicDisciplines = academicDisciplines
        self.labels = labels

    def run(self) -> dict:

        wat_results = self._get_wat_results(self.labels)
        print('Got data from WAT API...')
        lookup_results = self._get_lookup_results(self.labels)
        print('Got data from DBpedia Lookup...')
        falcon_results = self._get_falcon_results(self.labels)
        print('Got data from FALCON API...')

        # link entities that exist as 1:1 matches in academicDisciplines
        linked_entities = self.link_identical_entities(self.labels, wat_results, lookup_results, falcon_results, self.academicDisciplines)
        print('Linked identical entities...')

        # link entities that don't exist as 1:1 matches
        linked_entities = self.link_non_identical_entities(linked_entities, wat_results, lookup_results, falcon_results, self.academicDisciplines)
        print('Linked non-identical entities...')

        # link remaining entities based on fuzzy matching
        linked_entities = self.link_entities_fuzz(linked_entities)
        print('Linked fuzzy entities...')

        linked_entities_orkg = self.link_taxonomy(linked_entities)

        return linked_entities_orkg
        
    def _get_wat_results(self, labels) -> dict:
        """ 
        A function that gets a list of labels and returns 
        a dictionary of {label: list of linked DBpedia entity(ies)} using WAT API 
        """
        
        wat_api_results = {el: [] for el in labels}
        
        for rf in labels:
            wat_annotations = wat_entity_linking(rf)
            wat_api_results[rf] = ast.literal_eval(get_wat_annotations(wat_annotations))
            
        wat_results = {el: [] for el in labels}
        
        for key, value in wat_results.items():
            if  wat_api_results[key] != []:
                wat_results[key] = ['http://dbpedia.org/resource/'+wat_api_results[key][0]['wiki_title']]
        
        return wat_results
    
    def _get_lookup_results(self, labels) -> dict:
        """ A function that gets a list of labels and returns 
        a dictionary of {label: list of linked DBpedia entity(ies)} using the DBpedia Lookup service
        """
        
        lookup_api_results = {el: [] for el in labels}
        
        for label in labels:
            
            # Query API
            resp = req.get("https://lookup.dbpedia.org/api/search?query="+label)
            resp_text = resp.text
            # Convert it to BeautifulSoup XML
            beautsoup_rf = BeautifulSoup(resp_text, "xml")
            # Get all the result of all <resource>s
            beautsoup_resource = beautsoup_rf.find_all('resource')
            # Save it in the dictionary    
            lookup_api_results[label] = beautsoup_resource
    
    
        # remove <resource> and </resource>
        for key, value in lookup_api_results.items():
            resources = []
            if value != []:
                for resource in value:
                    resources.append(str(resource)[10:-11])
                    lookup_api_results[key] = resources
        
        return lookup_api_results
    
    def _get_falcon_results(self, labels) -> dict:
        """ A function that gets a list of labels and returns 
        a dictionary of {label: list of linked DBpedia entity(ies)} using the FALCON API
        """
        
        falcon_api_results = {el: [] for el in labels}
        
        for rf in labels:
            json_data = {
                'text': rf,
            }
            resp = req.post('https://labs.tib.eu/falcon/falcon2/api?mode=long&k=5&db=1', json=json_data)
            falcon_api_results[rf] = ast.literal_eval(resp.text)
            
        falcon_results_dbpedia = {el: [] for el in labels}
        
        for key, value in falcon_api_results.items():
            for element in value['entities_dbpedia']:
                falcon_results_dbpedia[key].append(element['URI'])
        
        return falcon_results_dbpedia
    
    def get_identical_entity(self, label: str, api_result: list) -> dict:
        """
        Input:
        label: str that represents the label we want to check
        api_result: the list of results for this label that was returned from one of the APIs
        academicDisciplines: a pandas DataFrame of entities that are objects of the dbo:academicDiscipline predicate
        Output:
        a dictionary that contains the resource that has the same label as the input 'label' and is also in 
        academicDisciplines. An empty dictionary is returned otherwise
        """
        
        # for lemmatization
        wnl = WordNetLemmatizer()
        # dict for final linking results
        linked_labels = {}
        # for punctuation removal
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        # for stop words removal
        stop_words = stopwords.words("english")
        
        # lower label and make it singular
        label = [wnl.lemmatize(word, 'n') for word in label.lower().split()]
        # remove stop words
        label = [word for word in label if not word in stop_words]
        
        label = ' '.join(label)
        
        for resource in api_result:
            # remove the string 'http://dbpedia.org/resource/' to compare labels
            entity = resource[28:]
            # remove punctuation
            entity = entity.translate(translator)
            # lower entity label and make it singular
            entity = [wnl.lemmatize(word, 'n') for word in entity.lower().split()]
            # remove stop words
            entity = [word for word in entity if not word in stop_words]
            
            entity = ' '.join(entity)
            
            if label == entity:
                if resource in self.academicDisciplines['discipline'].values:
                    linked_labels[resource] = 1
                    
        return linked_labels
    
    def link_identical_entities(self, labels: list, wat_results: dict, lookup_results: dict, falcon_results: dict) -> dict:
        """
        A function that iterated through the API results of 1. WAT API, 2. DBpedia Lookup, and 3. FALCON API
        and gets identical resources from DBpedia that also exist in the dbo:academicDisciplines list.
        The output is a dictionary of {label: [DBpedia resource]}
        """
    
        # this will be the returned dictionary which contains all linked identical entities
        linked_entities = {el: [] for el in all_labels}
    
        # get identical entities from WAT API
        for key, value in wat_results.items():
            if linked_entities[key] == []:
                linked_entities[key] = get_identical_entity(key, value, self.academicDisciplines)
            
        # get identical entities from DBpedia Lookup
        for key, value in lookup_results.items():
            if linked_entities[key] == []:
                linked_entities[key] = get_identical_entity(key, value, self.academicDisciplines)
            
        # get identical entities from FALCON API
        for key, value in falcon_results.items():
            if linked_entities[key] == []:
                linked_entities[key] = get_identical_entity(key, value, self.academicDisciplines)
                
        return linked_entities
    
    def link_non_identical_entities(self, linked_entities: dict, wat_results: dict, lookup_results: dict, falcon_results: dict, academicDisciplines: pd.DataFrame) -> dict:
        """
        A function that iterated through the API results of 1. WAT API, 2. DBpedia Lookup, and 3. FALCON API
        and gets non-identical resources from DBpedia that also exist in the dbo:academicDisciplines list.
        The output is a dictionary of {label: [DBpedia resources]}
        """
        
        updated_linked_entities = linked_entities
        
        for key, value in linked_entities.items():
            # for entities that are still not linked
            if linked_entities[key] == {}:
                # append results from all APIs
                all_api_results = [*wat_results[key], *lookup_results[key], *falcon_results[key]]
                
                all_linked_resources = {}
                for resource in set(all_api_results):
                    if resource in academicDisciplines['discipline'].values:
                        weight = all_api_results.count(resource)
                        all_linked_resources[resource] = weight
                        
                updated_linked_entities[key] = all_linked_resources
                
        return updated_linked_entities
    
    def get_fuzzy_match(self, label: str) -> str:
        """
        A function that gets a label (string) and a pd.DataFrame of the dbo:academicDiscipline list, and returns
        the top matched resource with the label based on fuzzy ration between the input label + the DBpedia resources 
        labels.
        """
        
        fuzz_ratio = []
        for idx, row in self.academicDisciplines.iterrows():
            fuzz_ratio.append(fuzz.ratio(label, row['label']))
        
        self.academicDisciplines['fuzz_ratio'] = fuzz_ratio
        
        linked_resource = self.academicDisciplines['discipline'][self.academicDisciplines['fuzz_ratio'].idxmax()]
        self.academicDisciplines = self.academicDisciplines.drop(columns=['fuzz_ratio'])
        
        return linked_resource


    def link_entities_fuzz(self, linked_entities: dict) -> dict:
        """
        A function that iterates over linked entities. For those that were not linked neither by identical nor non-identical
        matching; it links them with the most similar label in dbo:academicDiscipline based on fuzzy matching. 
        """
        
        for key, value in linked_entities.items():
            # for entities that are still not linked
            if linked_entities[key] == {}:
                linked_entities[key] = {
                    self.get_fuzzy_match(key, self.academicDisciplines): 1
                }
                
        return linked_entities
    
    def link_taxonomy(self, linked_entities: dict) -> dict:
        """
        A function that links the original ORKG taxonomy labels based on the results of the complex + non-complex labels.
        The output is a dictionary of {orkg-taxonomy-label: {linked-dbpedia-resource1: weight1, 
                                                        linked-dbpedia-resource2: weight2, etc.} }
                                                          """
        
        linked_entities_orkg = {el: [] for el in self.labels}
        
        for key, value in linked_entities_orkg.items():
            if key in linked_entities.keys():
                linked_entities_orkg[key].append(linked_entities[key])
            else:
                for label in complex_labels_dict[key]:
                    linked_entities_orkg[key].append(linked_entities[label])

        # merge dictionaries if one class has more than one
        for key, value in linked_entities_orkg.items():
            if len(value) > 1:
                input_dicts = [item for item in value]
                linked_entities_orkg[key] = dict(sum((Counter(dict(x)) for x in input_dicts), Counter()))
            else:
                linked_entities_orkg[key] = value[0]
                
        return linked_entities_orkg






