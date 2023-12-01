import os

import wandb
import torch
import torch.nn as nn
import numpy as np

from transformers import Trainer, AutoModel, TrainingArguments, AutoTokenizer, DataCollatorWithPadding, \
    PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import NextSentencePredictorOutput

from torch.utils.data import (Dataset, DataLoader)

import datasets
import evaluate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
wandb.init(project="kge-only-classifier-trainer", entity="raya-abu-ahmad")


class MyConfig(PretrainedConfig):
    model_type = 'mymodel'

    def __init__(self, labels_count=1, hidden_dim=768, mlp_dim=100, kg_dim=200, publisher_dim=768, dropout=0.1):
        super().__init__()

        self.labels_count = labels_count
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.kg_dim = kg_dim
        self.publisher_dim = publisher_dim
        self.dropout = dropout


class ExtraBertClassifier(PreTrainedModel):
    config_class = MyConfig

    def __init__(self, config_class):
        config = MyConfig()
        super().__init__(config)

        self.model = AutoModel.from_pretrained('malteos/scincl')
        self.model.to(device)
        self.config = config
        self.dropout = nn.Dropout(self.config.dropout)
        self.dropout.to(device)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.hidden_dim + self.config.kg_dim, self.config.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.config.mlp_dim, self.config.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.config.mlp_dim, self.config.labels_count)
        )
        self.mlp.to(device)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(device)

    def forward(self, input_ids, attention_mask, token_type_ids, class_kge, labels):
        text_results = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        text_embeddings = text_results.last_hidden_state[:, 0, :]
        text_embeddings = text_embeddings.to(device)

        kg_embeddings = torch.as_tensor(class_kge)
        kg_embeddings = kg_embeddings.to(device)

        concat_output = torch.cat((text_embeddings, kg_embeddings), dim=1)
        mlp_output = self.mlp(concat_output)
        prob = self.sigmoid(mlp_output)

        labels = labels.unsqueeze(1)

        loss = None

        if labels is not None:
            loss_funct = nn.BCELoss()
            loss = loss_funct(prob, labels)

        return NextSentencePredictorOutput(
            loss=loss,
            logits=prob
        )


def tokenize_function(example):
    return tokenizer(example["document_text"],
                     truncation=True, max_length=512,
                     return_tensors="pt",
                     padding=True)


def prepare_dataset(document_text, class_kges, labels):
    # Define dataset dictionary
    my_data = {
        'document_text': document_text,
        'class_kge': class_kges,
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


def main():
    # Load dataset
    document_text = torch.load('../../data/document_text_list.pt')
    class_kges = torch.load('../../data/class_new_KGEs.pt')
    labels = torch.load('../../data/labels.pt')
    labels = [float(label) for label in labels]

    # Define DataCollocator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare dataset
    tokenized_dataset = prepare_dataset(document_text, class_kges, labels)

    # Define model
    model = AutoModel.from_pretrained('malteos/scincl')
    model.to(device)

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="../../results/models",
        report_to="wandb",
        logging_steps=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3
    )

    config = MyConfig()

    # Define Trainer
    trainer = Trainer(
        model=ExtraBertClassifier(config),
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Train model
    trainer.train()

    # Evaluate model
    labels = tokenized_dataset["test"]['label']

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

    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    print(f'Accuracy: {accuracy}')
    precision = precision_metric.compute(predictions=preds, references=labels)
    print(f'Precision: {precision}')
    recall = recall_metric.compute(predictions=preds, references=labels)
    print(f'Recall: {recall}')
    f1 = f1_metric.compute(predictions=preds, references=labels)
    print(f'F1: {f1}')

    # Save model
    trainer.save_model("../../results/models")


if __name__ == '__main__':
    print(device)
    main()
