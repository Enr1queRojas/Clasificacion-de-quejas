# -*- coding: utf-8 -*-

"""
Created on Wed Jan 11 12:53:01 2023

@author: enriq
"""



import random
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
import warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Module, Linear, functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
from nltk.corpus import stopwords

#%%
loss_fn = torch.nn.CrossEntropyLoss()

# Función para entrenar el modelo
def train(model, dataloader, loss_fn, optimizer, device):
    """Entrena el modelo usando el dataloader proporcionado."""
    model.train()
    for inputs, attention_mask, labels in dataloader:
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

# Función para entrenar una época
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Entrena el modelo para una época y devuelve la pérdida y precisión promedio."""
    model.train()
    total_loss, total_accuracy = 0, 0
    for inputs, attention_mask, labels in dataloader:
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += (torch.argmax(logits, dim=1) == labels).sum().item()
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

# Función para evaluar el modelo
def evaluate(model, dataloader, device):
    """Evalúa el modelo usando el dataloader proporcionado."""
    model.eval()
    total_loss, total_accuracy = 0, 0
    for inputs, masks, labels in dataloader:
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(inputs, masks)[0]
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            total_accuracy += (torch.argmax(logits, dim=1) == labels).sum().item()
    return total_loss / len(dataloader), total_accuracy / len(dataloader)

#%%
# Función para obtener una muestra representativa
def representative_sample(observaciones,file_path,label_column, text_column):
    """Obtiene una muestra representativa del archivo proporcionado."""
    
    # Lee los datos 
    df = pd.read_csv(file_path)
    df = df[[label_column,text_column]]
    data = df.dropna()   
    
    # Define el número de observaciones a utilizar
    if observaciones > len(data[label_column]):
        observaciones = len(data[label_column])
        print(f"El archivo solo cuenta con {observaciones} observaciones.")
    else:
        observaciones = observaciones
        
    # Define el nivel de confianza y el error
    confidence_level = 0.95
    # Define un umbral para el número mínimo de observaciones por etiqueta
    min_obs = 10
    
    # Obtiene la proporción de etiquetas en la población total
    population_proportions = data.groupby(label_column).size() / len(data)
    
    # Selecciona las observaciones de manera aleatoria
    sample_indices = random.sample(range(0, len(data)), observaciones)
    sample_data = data.iloc[sample_indices]
    
    # Verifica la proporción de etiquetas en la muestra
    sample_proportions = sample_data.groupby(label_column).size() / len(sample_data)
    
    # Define un umbral para el número mínimo de observaciones por etiqueta
    min_obs = 10
    
    # Crea un nuevo dataframe para almacenar los datos balanceados
    representative_sample_data = pd.DataFrame(columns=data.columns)
    
    # Itera a través de cada etiqueta y verifica si la muestra es representativa
    for label, population_prop in population_proportions.items():
        
        if label not in sample_proportions.index:
            print(f"Etiqueta {label} no encontrada en sample_proportions.")
            continue
            
        sample_prop = sample_proportions[label]
        count = sample_data.groupby(label_column).size()[label]
        nobs = len(sample_data)
        z_value, p_value = proportions_ztest(count, nobs, population_prop, alternative='two-sided')
        lower_bound, upper_bound = proportion_confint(count, nobs, alpha=1-confidence_level)
    
        if count < min_obs:
            continue
            
        elif (sample_prop < lower_bound) or (sample_prop > upper_bound):
            continue
        else:
            # Agrega las filas con esta etiqueta al nuevo dataframe
            label_data = data[data[label_column] == label]
            sample_indices = random.sample(range(0, len(label_data)), int(observaciones*population_prop))
            representative_sample_data = pd.concat([representative_sample_data, label_data.iloc[sample_indices]], ignore_index=True)
    
    return representative_sample_data
#%%

# Función para obtener una muestra balanceada
def balanced_sample(observaciones, file_path, label_column, text_column):
    """Obtiene una muestra equilibrada del archivo proporcionado."""
    
    # Lee los datos
    df = pd.read_csv(file_path)
    df = df[[label_column, text_column]]
    data = df.dropna()
    
    # Número de etiquetas únicas
    num_labels = data[label_column].nunique()
    
    # Determina el número mínimo de observaciones disponibles para cualquier etiqueta
    min_obs_for_label = min(data[label_column].value_counts())
    
    # Si el número deseado de observaciones para cada etiqueta es mayor de lo disponible
    # Ajusta las observaciones por etiqueta al mínimo disponible e imprime un mensaje
    if observaciones // num_labels > min_obs_for_label:
        obs_per_label = min_obs_for_label
        total_obs = obs_per_label * num_labels
        print(f"No había suficientes observaciones para una muestra equilibrada. Se creó una muestra de {total_obs} observaciones.")
    else:
        obs_per_label = observaciones // num_labels
    
    # Crea un dataframe vacío para almacenar la muestra balanceada
    balanced_data = pd.DataFrame(columns=data.columns)
    
    # Para cada etiqueta, toma 'obs_per_label' número de observaciones
    for label in data[label_column].unique():
        label_data = data[data[label_column] == label]
        sample_data = label_data.sample(n=obs_per_label, replace=False)
        balanced_data = pd.concat([balanced_data, sample_data])
    
    # Devuelve la muestra balanceada
    return balanced_data

#%%
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)  # Primera capa oculta con 512 neuronas
        self.fc2 = nn.Linear(512, 256)        # Segunda capa oculta con 256 neuronas
        self.fc3 = nn.Linear(256, num_classes) # Capa de salida con 'num_classes' neuronas

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Aplicar ReLU a la salida de la primera capa oculta
        x = F.relu(self.fc2(x))  # Aplicar ReLU a la salida de la segunda capa oculta
        x = self.fc3(x)          # Salida de la capa final
        return x
#%%
def train_and_evaluate_nn(X_train, X_test, y_train, y_test, num_epochs=100, learning_rate=0.0001, batch_size=64):
    
    # Convertir a tensores 
    X_train_tensor = torch.FloatTensor(X_train.toarray())
    X_test_tensor = torch.FloatTensor(X_test.toarray())

    
    # Codificar etiquetas a números
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)
    
    y_train_tensor = torch.LongTensor(y_train_encoded)
    y_test_tensor = torch.LongTensor(y_test_encoded)
    
    # Definir datasets y dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    
    # Definir dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Definir modelo, criterio y optimizador
    model = SimpleNN(X_train.shape[1], len(encoder.classes_)).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Entrenar el modelo
    for epoch in range(num_epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluar el modelo y recolectar predicciones y etiquetas verdaderas
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Guardar el encoder en el directorio especificado
    encoder_save_path = r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\results\Benchmark\label_encoder.pkl"
    joblib.dump(encoder, encoder_save_path)

    # Obtener el reporte de clasificación
    NN_report = classification_report(all_labels, all_predictions, target_names=encoder.classes_)

    # Obtener y devolver el reporte de clasificación
    return NN_report,model
#%%
def preprocess_text(text,nlp,stop_words):

    # Convertir a minúsculas
    text = text.lower()
    # Procesar el texto con spaCy
    doc = nlp(text)
    # Lematizar y eliminar stopwords
    tokens = [token.lemma_ for token in doc if token.lemma_ not in stop_words and not token.is_punct]
    # Reconstruir el texto procesado
    return " ".join(tokens)