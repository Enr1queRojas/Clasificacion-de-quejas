from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from helpers import train_and_evaluate_nn,preprocess_text
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib
import os
import torch
import nltk
import spacy
from nltk.corpus import stopwords



data = pd.read_csv(r'C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\files\sample.csv')

# Cargar spaCy model para inglés
nlp = spacy.load("en_core_web_sm")
# Lista de stopwords de NLTK
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))



# Aplicar preprocesamiento al DataFrame
data['text'] = data['text'].apply(lambda x: preprocess_text(x, nlp, stop_words))


# Vectorizar el texto usando TF-IDF 
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Verificar las dimensiones de los conjuntos de entrenamiento y prueba
X_train.shape, X_test.shape


###               Regresión Logística              ###

# Entrenar un modelo de regresión logística
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)
# Evaluar el rendimiento del modelo
accuracy_train = accuracy_score(y_test, y_pred)
classification_rep_train = classification_report(y_test, y_pred)
print(classification_rep_train)

###               Regresión SVM              ###
# Entrenar modelo SVM
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train, y_train)
# Evaluar el modelo SVM en el conjunto de prueba
svm_predictions = svm_model.predict(X_test)
svm_report = classification_report(y_test, svm_predictions, target_names=data['label'].factorize()[1])
print(svm_report)

###               Random Forest             ###
# Reducir la dimensionalidad del conjunto de datos
svd = TruncatedSVD(n_components=1000, random_state=42)
X_train_reduced = svd.fit_transform(X_train)
X_test_reduced = svd.transform(X_test)
# Entrenar modelo de Bosques Aleatorios con los datos reducidos
rf_model_reduced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_reduced.fit(X_train_reduced, y_train)
# Evaluar el modelo de Bosques Aleatorios en el conjunto de prueba reducido
rf_predictions_reduced = rf_model_reduced.predict(X_test_reduced)
rf_report_reduced = classification_report(y_test, rf_predictions_reduced, target_names=data['label'].factorize()[1])
print(rf_report_reduced)



###               Red Neuronal             ###
'Input (1000) -> FC (512) -> ReLU -> FC (256) -> ReLU -> FC (num_classes)'
NN_report, trained_nn_model = train_and_evaluate_nn(X_train, X_test, y_train, y_test, num_epochs=100, learning_rate=0.0001, batch_size=64)
print(NN_report)

# Definir la carpeta donde se guardarán los modelos
save_folder = r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\results\Benchmark"

# Verificar si la carpeta existe; si no, crearla
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Guardar los modelos de scikit-learn
joblib.dump(clf, os.path.join(save_folder, "logistic_regression_model.pkl"))
joblib.dump(svm_model, os.path.join(save_folder, "svm_model.pkl"))
joblib.dump(rf_model_reduced, os.path.join(save_folder, "random_forest_model.pkl"))

# Guardar el modelo de PyTorch
torch.save(trained_nn_model.state_dict(), os.path.join(save_folder, "neural_network_model.pt"))

# Guardar el vectorizer y metadatos para la red neuronal
joblib.dump(vectorizer, os.path.join(save_folder, "tfidf_vectorizer.pkl"))
joblib.dump(X_train.shape[1], os.path.join(save_folder, "nn_input_dim.pkl"))

print("Models saved successfully!")