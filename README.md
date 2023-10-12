# for-classifier

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
```

## To run classifier:

```commandline
```
