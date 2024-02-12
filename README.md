

# Investigating Knowledge Injection Approaches for Field of Research Classification of Scholarly Articles 

## Description

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

#### Download pre-prepared dataset

All data required for running the classifiers are available for download at: [https://zenodo.org/records/10245830](https://zenodo.org/records/10649951). After downloading, please save all ```.pt``` files under ```/data/classifier``` in order to be able to train and test the models.

#### Construct dataset

This repository also contains the code for creating the data in the link above (including linking ORKG labels to DBpedia entities) under the ```data_prep``` directory. 
The data is prepared by using the nfdi4ds dataset for the field of research classification (FoRC) shared task. The code for creating this dataset can be found [here](https://github.com/ryabhmd/nfdi4ds-forc). A link to download the dataset can be provided in order to run the steps below.  

1. Link the ORKG taxonomy to DBpedia entities:

```commandline
python data_prep/entity_linking/entity_linking.py
```

2. Create KGEs of taxonomy labels:

  Note that this step includes downloading a pre-trained DBpedia embeddings dataset from Zenodo (https://zenodo.org/records/6384728) and thus requires enough space. It will ca. 3 hours to download and ca. 1 hour to run the code in order to get the embeddings.
  In order to run the code, the dataset from Zenodo should be downloaded by running ```zenodo_get -d '10.5281/zenodo.6384728'```.
  After obtaining the dataset, it should be saved under ```/data/embeddings.zip```.

```commandline
python data_prep/entity_embeddings/get_kges_pretrained.py
```

  Alternatively, KGEs can be constructed using [pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec). This process will take 2-3 hours and does not need an external dataset. However, the models perform better using the pre-trained embeddings as opposed to the ones constructed using pyRDF2Vec.
  
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

5. Create authors and publishers embeddings:
The code below creates embeddings for authors and publishers that can be used in the classifiers below. Note that both of these scripts use SciNCL to create embeddings of each title and abstract in the dataset and thus require enough system memory to run. Each code will take ca. 3 hours to run. 

```commandline
python data_prep/authors_data.py
```

```commandline
python data_prep/publishers_data.py
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

