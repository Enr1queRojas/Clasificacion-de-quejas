

import pandas as pd
import torch
from helpers import SimpleNN
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_column = 'label'
text_column = 'text'
file_path = r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\files\rows_clean.csv"


sample = pd.read_csv(file_path).sample(n=50000)
resultados = pd.DataFrame(columns=["label", "predicted_label", "predicted_label_O", "label_probs", "model"])


# Cargar el modelo y el tokenizador
folder = r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\master_weight"
modelo_entrenado = RobertaForSequenceClassification.from_pretrained(folder, num_labels=sample.label.nunique(),ignore_mismatched_sizes=True).to(device)
tokenizer = RobertaTokenizer.from_pretrained(folder)

# Crear un mapeo de etiquetas
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(sample['label'].unique())
label_mapping_df = pd.DataFrame({'label': sample['label'].unique(), 'code_label': encoded_labels})

pbar = tqdm(sample.iterrows(), total=len(sample), desc=f'mod-3000')

# Realizar clasificación para cada sample
for index, row in sample.iterrows():
    inputs = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = modelo_entrenado(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_index = torch.argmax(probs, dim=1).item()
    predicted_label = modelo_entrenado.config.id2label[predicted_index]
    label_probs = probs[0][predicted_index].item()
    predicted_label_O = label_mapping_df[label_mapping_df['code_label'] == predicted_index]['label'].values[0]

    temp_df = pd.DataFrame([[row['label'], predicted_label, predicted_label_O, label_probs, 'mod-3000']], columns=["label", "predicted_label", "predicted_label_O", "label_probs", "model"])
    resultados = pd.concat([resultados, temp_df])
    pbar.update()

pbar.close()


# Guardar y reportar resultados
print('Procesado')
# Calculando Accuracy
resultados['is_correct'] = (resultados['label'] == resultados['predicted_label_O']).astype(int)
accuracy = resultados['is_correct'].mean() * 100
resultados['Accuracy'] = (resultados['label'] == resultados['predicted_label_O']).astype(int)
accuracy_summary = resultados.groupby(['label', 'Accuracy']).size().unstack(fill_value=0)
accuracy_summary['Total'] = accuracy_summary.sum(axis=1)
accuracy_summary = accuracy_summary.div(accuracy_summary['Total'], axis=0) 
accuracy_summary.columns = ['FALSE', 'TRUE', 'Total']

# Usar classification_report para obtener precision, recall y f1-score
report = pd.DataFrame(classification_report(resultados['label'], resultados['predicted_label_O'], zero_division=0, output_dict=True)).transpose()
report = report.iloc[:(sample.label.nunique())]
report.insert(0,'accuracy',accuracy_summary['TRUE'],allow_duplicates=False)



# Mostrando los resultados
print(f"Overall Accuracy: {accuracy:.2f}%\n")
print(report)
resultados.to_csv(r'C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\results\results_5r10k.csv')

import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer



models_folder = r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\results\Benchmark"
model_names = ["logistic_regression", "svm", "random_forest", "neural_network"]
model_objects = {}
# Cargar el vectorizer y metadatos para la red neuronal
vectorizer = joblib.load(os.path.join(models_folder, "tfidf_vectorizer.pkl"))
nn_input_dim = joblib.load(os.path.join(models_folder, "nn_input_dim.pkl"))

svd = joblib.load(os.path.join(models_folder, "svd_object.pkl"))


# Cargar el encoder
encoder = joblib.load(models_folder)


for model_name in model_names:
    if model_name == "neural_network":
        predicted_label = get_nn_prediction(model, text)
    elif model_name == "random_forest":
        transformed_text = vectorizer.transform([text])
        reduced_text = svd.transform(transformed_text)
        predicted_label = model.predict(reduced_text)[0]
    else:
        transformed_text = vectorizer.transform([text])
        predicted_label = model.predict(transformed_text)[0]



# Función para obtener la predicción de la red neuronal
def get_nn_prediction(model, text):
    inputs = torch.FloatTensor(vectorizer.transform([text]).toarray()).to(device)
    outputs = model(inputs)
    predicted_index = torch.argmax(outputs, dim=1).item()
    return encoder.inverse_transform([predicted_index])[0]

# Iterando sobre cada fila en la muestra y haciendo predicciones con cada modelo
for index, row in tqdm(sample.iterrows(), total=len(sample), desc="Processing"):
    text = row[text_column]
    for model_name, model in model_objects.items():
        if model_name == "neural_network":
            predicted_label = get_nn_prediction(model, text)
        else:
            transformed_text = vectorizer.transform([text])
            predicted_label = model.predict(transformed_text)[0]
        
        # Guardar los resultados
        temp_df = pd.DataFrame({
            "label": [row[label_column]],
            "predicted_label": [predicted_label],
            "model": [model_name]
        })
        resultados = pd.concat([resultados, temp_df])

        
