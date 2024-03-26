# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:19:53 2023

@author: enriq
"""

import pandas as pd
import torch
import os
from helpers import representative_sample
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
label_column = 'label'
text_column = 'text'
file_path = (r'C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\files\rows_clean.csv')


def train_and_evaluate(training_args):
    sample = 10000
    df = representative_sample(sample,file_path,label_column, text_column)
    df.to_csv(r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\files\sample.csv")


    num_labels = len(df.label.unique())

    # Carga del modelo entrenado
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=num_labels,ignore_mismatched_sizes=True)
    model.to(device)

    # Inicialización del tokenizador
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Obteniendo las etiquetas del dataframe
    labels = df[label_column].tolist()

    # Convert dataframe to list
    texts = df[text_column].tolist()

    # Tokenize the texts
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Convert labels to tensor
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels = torch.tensor(labels_encoded)


    # Split the dataset into train, validation, and test sets
    train_size = int(0.6 * len(df))
    val_size = int(0.2 * len(df))
    test_size = len(df) - train_size - val_size
    train_dataset, eval_dataset, test_dataset = random_split(TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels), [train_size, val_size, test_size])



    # Define a data collator function
    def data_collator(batch):
        # Stack the input_ids, attention_mask, and labels of each example in the batch
        input_ids = torch.stack([example[0] for example in batch])
        attention_mask = torch.stack([example[1] for example in batch])
        labels = torch.stack([example[2] for example in batch])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )


    # Start training
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate(eval_dataset)

    # Print the evaluation results
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print("{}: {}".format(key, value))


    # Predictions of the model
    y_pred = trainer.predict(test_dataset)
    y_pred =le.inverse_transform(y_pred[1])

    # Create y_test as a list of true labels for the test set
    y_test = [label.numpy().item() for i, (_, _, label) in enumerate(test_dataset)]
    y_test = le.inverse_transform(y_test)

    # Calculation of metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    confusion = confusion_matrix(y_test, y_pred)

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)
    print("Confusion Matrix: \n", confusion)


    # Make predictions on test dataset
    predictions = trainer.predict(test_dataset)

    #Decode the labels back to their original form
    decoded_labels = le.inverse_transform(predictions[1])

    # Save the LabelEncoder object to use later
    import pickle
    with open("label_encoder.pkl", 'wb') as f:
        pickle.dump(le, f)

    # You can later load the LabelEncoder object with the following code:
    with open("label_encoder.pkl", 'rb') as f:
        le = pickle.load(f)
    decoded_labels = le.inverse_transform(predictions[1])


    # Guardado de los nuevos pesos
    
    path = r'C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\results'
    dir_path = os.path.join(path, "model-{}".format(sample))
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)

    model_save_path = os.path.join(dir_path, "new_weights")
    tokenizer_save_path = os.path.join(dir_path)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
 
    # Calculate F1-score and return it along with the evaluation results
    f1 = f1_score(y_test, y_pred, average='macro')
    eval_results["eval_f1"] = f1

    return eval_results


training_args = TrainingArguments(
    output_dir= r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001", 
    evaluation_strategy='steps',     
    save_strategy='steps',           
    save_total_limit=10,             
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_dir='./logs',            
    logging_steps=500,               
    load_best_model_at_end=True,
    gradient_accumulation_steps=2,  
)

train_and_evaluate(training_args)
