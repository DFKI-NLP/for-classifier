import pandas as pd
import numpy as np
import random
import torch

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

random.seed(42)


def get_binary_dataset(dataset, negative_samples=3):
    """
    A function that takes the input .csv FoRC dataset and returns a new dataset that
    consists of a list of tuples. Each tuple is defined as follows:
    ((row_number, category_number), binary_label), where:
    
    row_number denotes the number of the data point in the original FoRC dataset, 
    category_number denotes the categorical number of the label as encoded by LabelEncoder, 
    binary_label is 1 if the data point row_number is labeled with the category_number, and 0 otherwise

    To create the dataset, 1 positive samples for each data point is created (i.e. the real label the data point has in FoRC), 
    followed by N negative samples chosen randomly from the list of ORKG labels. The default of N = 3. 
    """

    binary_dataset = []
    labels, labels_mapping = get_categorical_labels(dataset)

    for idx, row in dataset.iterrows():

        # positive sample (1 per paper)
        label = labels[idx]
        binary_dataset.append(((idx, label), 1))

        # negative samples (according to the defined variable)
        unique_labels_list = list(set(labels))
        unique_labels_list.remove(label)
        negative_labels = random.sample(unique_labels_list, negative_samples)

        for category in negative_labels:
            binary_dataset.append(((idx, category), 0))

    return binary_dataset, labels_mapping


def clean_data(data: pd.DataFrame):
    texts_rows = []
    tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')

    for idx, row in data.iterrows():
        # Substitute 'Other Quantitative Biology' with its parent node due to ambiguity
        if row['label'] == 'Other Quantitative Biology':
            data.at[idx, 'label'] = 'Biology, Integrated Biology, Integrated Biomedical Sciences'
        # Clean labels that include "&" or "/"
        if '/' in row['label']:
            cleaned_label = row['label'].replace("/", ", ")
            data.at[idx, 'label'] = cleaned_label
        if '&' in row['label']:
            cleaned_label = row['label'].replace("&", "and")
            data.at[idx, 'label'] = cleaned_label

        # Append title[SEP]abstract to add it in a new column
        if type(row['abstract']) == str:
            text = row['title'] + tokenizer.sep_token + row['abstract']
        else:
            text = row['title']

        texts_rows.append(text)

    data['document_text'] = texts_rows

    return data


def get_categorical_labels(data):
    """
    Returns an array of categorical labels with the same indices as the FoRC dataset.
    E.g. array([89, 89, 89, ...,  7,  7,  7]).

    Additionally, saved the labels mapping to their category number for later use.
    """

    label_encoder = LabelEncoder()
    labels = data['label'].to_list()
    labels = label_encoder.fit_transform(labels)

    labels_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return labels, labels_mapping


def get_dataset(binary_datapoint, dataset, taxonomy_texts, taxonomy_embeddings, classes_mapping_rev):
    """
    input data point will be in the format of ((26, 89), 1) denoting ((idx in dataset, idx of FoR class), binary label)
    """
    dataset_idx = binary_datapoint[0][0]
    class_idx = binary_datapoint[0][1]
    label = binary_datapoint[1]

    # 1. get document text
    document_text = dataset['document_text'][dataset_idx]

    # 2. get DBpedia class text (i.e. DBpedia label(s)[SEP]DBpedia comment(s))
    class_text = taxonomy_texts[classes_mapping_rev[class_idx]]

    # 3. get ORKG class label 
    orkg_class_text = classes_mapping_rev[class_idx]

    # 4. get class DBpedia embedding
    class_dbpedia_KGE = taxonomy_embeddings[classes_mapping_rev[class_idx]].astype(np.float64)

    # 5. get class categorical number (for baseline)
    class_cat = np.asarray([class_idx])


    return document_text, class_text, orkg_class_text, class_dbpedia_KGE, class_cat, label


def get_classifier_data(binary_dataset, dataset, taxonomy_texts, taxonomy_embeddings, labels_mapping):
    document_text_list = []
    class_dbpedia_text_list = []
    class_orkg_text_list = []
    class_kge_list = []
    class_cat_list = []
    label_list = []

    classes_mapping_rev = dict((v, k) for k, v in labels_mapping.items())

    for data_point in binary_dataset:
        document_text, class_text, orkg_class_text, class_dbpedia_KGE, class_cat, label = get_dataset(data_point,
                                                                          dataset,
                                                                          taxonomy_texts,
                                                                          taxonomy_embeddings,
                                                                          classes_mapping_rev)
        document_text_list.append(document_text)
        class_dbpedia_text_list.append(class_text)
        class_orkg_text_list.append(orkg_class_text)
        class_kge_list.append(class_dbpedia_KGE)
        class_cat_list.append(class_cat)
        label_list.append(label)

    return document_text_list, class_dbpedia_text_list, class_orkg_text_list, class_kge_list, class_cat_list, label_list


def main():
    data = pd.read_csv('data/forc_I_dataset_FINAL_September.csv')
    cleaned_data = clean_data(data)
    print("Cleaned data...")

    # get binary data that consists of ((row_number, category_number), binary_label)
    binary_data, labels_mapping = get_binary_dataset(cleaned_data)
    # shuffle data
    random.shuffle(binary_data)
    torch.save(binary_data, 'data/classifier/binary_data.pt')
    print("Constructed binary dataset and saved in data/classifier/binary_data.pt...")

    taxonomy_texts = torch.load('data/taxonomy_texts.pt')
    taxonomy_embeddings = torch.load('data/taxonomy_embeddings.pt')

    document_text_list, class_dbpedia_text_list, class_orkg_text_list, class_kge_list, class_cat_list, label_list = get_classifier_data(binary_data,
                                                                                          cleaned_data,
                                                                                          taxonomy_texts,
                                                                                          taxonomy_embeddings,
                                                                                          labels_mapping)

    print("Constructed input for binary classifier successfully!")

    torch.save(document_text_list, 'data/classifier/document_text_list.pt')
    torch.save(class_dbpedia_text_list, 'data/classifier/class_texts_dbpedia_only.pt')
    torch.save(class_orkg_text_list, 'data/classifier/class_texts_orkg_only.pt')
    torch.save(class_kge_list, 'data/classifier/class_new_KGEs.pt')
    torch.save(class_cat_list, 'data/classifier/class_category_list.pt')
    torch.save(label_list, 'data/classifier/labels.pt')

    print("Saved data under /data/classifier/")


if __name__ == '__main__':
    main()
