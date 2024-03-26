"""
Created on Tue Mar  7 12:58:52 2023

@author: enriq
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Carga del conjunto de datos
df = pd.read_csv(r"C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\Files\complaints_processed.csv")

# Análisis de Valores Faltantes
missing_values = df.isnull().sum()

print(len(df))
print(missing_values)

print(len(df))
print(df.groupby(['label'])['label'].count())

#plt.figure(figsize=(12, 8))
#missing_values.plot(kind='bar', color='green')
#plt.title('Valores Faltantes en el Conjunto de Datos')
#plt.ylabel('Número de Valores Faltantes')
#plt.show()


#  Limpieza y preprocesamiento de los datos
df = df[["label", "text"]]
df = df.dropna()

date_pattern = r'XX/XX/\d{4}'
confidential_pattern = r'X{4}'
d_pattern = r'XX/XX/'

df['text'] = df['text'].str.replace(date_pattern, '', regex=True)
df['text'] = df['text'].str.replace(d_pattern, '', regex=True)
df['text'] = df['text'].str.replace(confidential_pattern, '', regex=True)


# 3. Análisis Descriptivo Básico y Visualizaciones

issue_counts = df['label'].value_counts()

plt.figure(figsize=(20, 15))  
issue_counts.plot(kind='barh', color='#004270', )
plt.title('Distribuciones de Categorías', fontsize=28)  
plt.xticks(fontsize=24)  
plt.yticks(fontsize=28)  

plt.tight_layout()  # Ajustar el diseño para que todo encaje bien
plt.savefig("modified_figure.png", dpi=600)  # Guardar la figura con alta resolución
plt.show()


# 5. Análisis de Texto
df['complaint_length'] = df['text'].apply(len)

# Calcula la media y mediana
mean_length = df['complaint_length'].mean()
median_length = df['complaint_length'].median()

# Histograma y KDE 
plt.figure(figsize=(12, 8))
sns.histplot(df['complaint_length'], bins=60, color='#004270', kde=True)
plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1.5)
plt.axvline(median_length, color='blue', linestyle='dashed', linewidth=1.5)
plt.legend({'Media':mean_length, 'Mediana':median_length}, fontsize=20)
plt.title('Distribución de la Longitud de Queja', fontsize=28)
plt.xlabel('Longitud de Queja', fontsize=24)
plt.ylabel('Número de Quejas', fontsize=24)
plt.xlim(0, 10000)
plt.show()

skewness = df['complaint_length'].skew()
print(f"Sesgo de la distribución: {skewness:.2f}")

# Nube de palabras para una muestra de quejas de consumidores
sample_text = ' '.join(df.sample(5000)['text'].tolist())
wordcloud = WordCloud(background_color='white', max_words=200, width=800, height=400).generate(sample_text)

plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Palabras Más Comunes en Quejas de Consumidores (Muestra)')
plt.show()

df.to_csv(r'C:\Users\Usuario\OneDrive - ITESO\Iteso\Ciencia de datos\IDI\IDI3\ds_code_001\files\rows_clean.csv')