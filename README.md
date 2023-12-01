

# Investigating Approaches for Field of Research Classification of Scholarly Articles 
![header](https://github.com/ryabhmd/for-classifier/assets/77779090/d06d9eb0-a581-4982-84b2-d68d3e09c72e)


## Description

This repository holds the code for my master's thesis project, which investigates classifying scholarly articles into research fields by exploring knowledge injection approaches. 

There are different models in the ```models``` directory that utilise different features from scholarly articles: 
- Titles + abstracts
- Authors
- Publishers
- Full metadata
  
The models also have different methods to semantically represent fields of research: 
- Categorical (baseline)
- Using taxonomy labels (the taxonomy used is https://orkg.org/fields)
- Linking labels to DBpedia entities and using the text under rdfs:label + rdfs:comment
- Linking labels to DBpedia entities and using knowledge graph embeddings (pre-trained embeddings from https://zenodo.org/records/6384728)


## Dataset



## To create the classification dataset from the FoRC pre-prepared dataset: 

1. Link the taxonomy to DBpedia entities:

```commandline
python data_prep/entity_linking/entity_linking.py
```

2. Create KGEs of taxonomy labels:

```commandline
python data_prep/entity_embeddings/get_kges.py
```

3. Create textual representations of taxonomy labels:

```commandline
python data_prep/entity_embeddings/get_kg_texts.py
```

4. Create binary dataset for classifier:

```commandline
python data_prep/data_for_classifier.py
```

## To run models:

```commandline
python models/kge_classifier.py
```

## Results
<p align="center">
  <img src="https://github.com/ryabhmd/for-classifier/assets/77779090/a0038c81-e08e-415b-a542-c01ae95c2938" height="600" width="600"/>

</p>


