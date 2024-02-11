import os

import wandb
import torch
import numpy as np

from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import BertForSequenceClassification, AutoTokenizer

from torch.utils.data import (Dataset, DataLoader)

import datasets
import evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')

def tokenize_function(example):
    return tokenizer(example["document_text"],
                     example["class_text"],
                     truncation=True,
                     max_length=512)


def prepare_dataset(document_text, class_text, labels):
    # Define dataset dictionary
    my_data = {
        'document_text': document_text,
        'class_text': class_text,
        'label': labels,
        'idx': [float(num) for num in range(len(labels))]
    }
    # Make it a datasets.Dataset object
    my_data_dataset = datasets.Dataset.from_dict(my_data)
    # Tokenize it
    tokenized_dataset = my_data_dataset.map(tokenize_function, batched=True)
    # Split dataset
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)

    return tokenized_dataset


def compute_metrics(eval_pred):
    metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels), \
        precision_metric.compute(predictions=predictions, references=labels), \
        recall_metric.compute(predictions=predictions, references=labels), \
        f1_metric.compute(predictions=predictions, references=labels)


def main():
    # Load dataset
    document_text = torch.load('data/classifier/document_text_list.pt')
    # change to 'class_texts_orkg_only.pt' to run the model with only ORKG class labels
    class_text = torch.load('data/classifier/class_texts_dbpedia_only.pt')
    labels = torch.load('data/classifier/labels.pt')
    labels = [float(label) for label in labels]

    # Define DataCollocator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare dataset
    tokenized_dataset = prepare_dataset(document_text, class_text, labels)

    # Define model
    model = BertForSequenceClassification.from_pretrained('malteos/scincl', num_labels=1)
    model.to(device)

    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "pairwise-text-classification"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="results/models",
        report_to="wandb",
        logging_steps=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train model
    trainer.train()

    # Evaluate model
    predictions = trainer.predict(tokenized_dataset["test"])
    accuracy_metric = evaluate.load('accuracy')
    precision_metric = evaluate.load('precision')
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')

    preds = []
    for pred in predictions.predictions:

        if pred >= 0.5:
            preds.append(1)
        else:
            preds.append(0)

    accuracy = accuracy_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Accuracy: {accuracy}')
    precision = precision_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Precision: {precision}')
    recall = recall_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'Recall: {recall}')
    f1 = f1_metric.compute(predictions=preds, references=predictions.label_ids)
    print(f'F1: {f1}')

    # Save model
    trainer.save_model("results/models")


if __name__ == '__main__':
    print(device)
    main()
