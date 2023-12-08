

# Investigating Knowledge Injection Approaches for Field of Research Classification of Scholarly Articles 
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

#### *All data required for running the classifiers are available for download at: https://zenodo.org/records/10245830.*



However, this repository also contains the code for creating the data (including linking ORKG labels to DBpedia entities) under ```data_prep``` directory. 
The data is prepared by using the nfdi4ds dataset for the field of research classification (FoRC) shared task. The code for creating this dataset can be found [here](https://github.com/ryabhmd/nfdi4ds-forc). A link to download the dataset can be provided in order to run the steps below.  

1. Link the ORKG taxonomy to DBpedia entities:

```commandline
python data_prep/entity_linking/entity_linking.py
```

2. Create KGEs of taxonomy labels:

  Note that this step includes downloading a pre-trained DBpedia embeddings dataset from Zenodo (https://zenodo.org/records/6384728) and thus requires enough space. Additionally, it will take up to 2 hours to download and process.
  In order to run the code, the dataset from Zenodo should be downloaded by running ```zenodo_get -d '10.5281/zenodo.6384728'```.
  After obtaining the dataset, it should be saved under ```/data/embeddings.zip``` in order to run the code below.

```commandline
python data_prep/entity_embeddings/get_kges_pretrained.py
```

  Alternatively, KGEs can be constructed using [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec), which is a process that takes less time. However, the models perform better using the pre-trained embeddings as opposed to using pyRDF2Vec.
  
```commandline
python data_prep/entity_embeddings/get_kges_pyrdf2vec.py
```

3. Create textual representations from DBpedia of taxonomy labels:

```commandline
python data_prep/entity_embeddings/get_kg_texts.py
```

4. Create the binary dataset for the classifier:

```commandline
python data_prep/data_for_classifier.py
```

## Models

1. Categorical baseline:
  ```commandline
python models/categorical_baseline.py
```

2. Pairwise text classifier (class features either ORKG labels or DBpedia entities text):
```commandline
python models/text_classifier_trainer.py
```

3. KGEs only:
```commandline
python models/kge-only-classifier.py
```

4. Adding author embeddings:
```commandline
python models/kge-authors-classifier.py
```

5. Adding publishers embeddings:
```commandline
python models/kge-publishers-classifier.py
```

6. Full metadata:
```commandline
python models/kge-authors-publishers-classifier.py
```

## Results

| Publication Features                      | Class Features      | Precision | Recall    | F1        | Accuracy  |
|-------------------------------------------|---------------------|-----------|-----------|-----------|-----------|
| **Baseline**                              |                     |           |           |           |           |
| Titles + Abstracts                        | Categorical Encoder | 0.0       | 0.0       | 0.0       | 74.85     |
| **Embedding Class Labels with SciNCL**    |                     |           |           |           |           |
| Titles + Abstracts                        | ORKG Labels Text    | 93.54     | 93.80     | 93.67     | 96.83     |
| **Injecting DBpedia Class Features**      |                     |           |           |           |           |
| Titles + Abstracts                        | DBpedia Text        | **93.55** | **94.11** | **93.83** | **96.91** |
| Titles + Abstracts                        | KGEs                | 75.83     | 29.39     | 42.36     | 80.00     |
| Titles + Abstracts                        | DBpedia Text + KGEs | 93.18     | 93.19     | 93.18     | 96.60     |
| **Adding Publication Metadata**           |                     |           |           |           |           |
| Titles + Abstracts + Authors              | DBpedia Text + KGEs | 93.20     | 92.02     | 92.61     | 96.32     |
| Titles + Abstracts + Publishers           | DBpedia Text + KGEs | 92.25     | 93.52     | 92.88     | 96.43     |
| Titles + Abstracts + Authors + Publishers | DBpedia Text + KGEs | 93.28     | 92.51     | 92.90     | 96.43     |

Additional graphs and comparisons between the models can be seen at: https://api.wandb.ai/links/raya-abu-ahmad/ykbq4ke4. 


