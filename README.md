

# Investigating Approaches for Field of Research Classification of Scholarly Articles 
![header](https://github.com/ryabhmd/for-classifier/assets/77779090/d06d9eb0-a581-4982-84b2-d68d3e09c72e)


## Description

This repository holds the code for my master's thesis project, which investigates classifying scholarly articles into research fields by exploring knowledge injection approaches.


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

## To run classifier:

```commandline
python models/kge_classifier.py
```
