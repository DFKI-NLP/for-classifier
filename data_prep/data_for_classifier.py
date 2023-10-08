import pandas as pd
import random
import torch

from sklearn.preprocessing import LabelEncoder

def get_binary_dataset(dataset):

    binary_dataset = []
    negative_samples = 3

    label_encoder = LabelEncoder()
    labels = dataset['label'].to_list()
    labels = label_encoder.fit_transform(labels)

def main():

    data = pd.read_csv('~/documents/for-classifier/data/forc_I_dataset_FINAL_September.csv')


if __name__ == '__main__':
    main()