import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization


# Loads the dataset 
df = pd.read_csv("../data/SMSSpamCollection.csv", encoding = 'latin-1')

# Cleans dataset 
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
df = df.rename(columns={'v1': 'label', 'v2': 'Text'})

# Label Encoding
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data and convert tpo NumPy arrays
X_train, X_test, y_train, y_test = train_test_split(
    df['Text'],
    df['label_enc'],
    test_size=0.2,
    random_state=42
)
X_train_np = X_train.to_numpy()
X_test_np  = X_test.to_numpy()
y_train_np = y_train.to_numpy()
y_test_np  = y_test.to_numpy()

# Text statistics for vectorization
avg_words_per_message = round(df['Text'].str.split().str.len().mean())
total_unique_words = len(set(" ".join(df['Text']).split()))

# helper functions for training and evaluation 
def compile_and_fit(model, epochs=5):
    model.compile(
        optimizer= 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )
    history = model.fit(
        X_train_np,
        y_train_np,
        epochs = epochs,
        validation_data = (X_test_np, y_test_np)
    )

    return history

def get_metrics(model, X, y):
    y_preds = np.round(model.predict(X))
    return {
        'accuracy': accuracy_score(y, y_preds),
        'precision': precision_score(y, y_preds),
        'recall': recall_score(y, y_preds),
        'f1-score': f1_score(y, y_preds),
    }

# Text vectorization layer
text_vec  = TextVectorization(
    max_tokens = total_unique_words,
    standardize = 'lower_and_strip_punctuation',
    output_sequence_length = avg_words_per_message
)

text_vec.adapt(X_train_np)

"""
Model 1 
Dense embedding model
(Build and Train)
"""
input_layer = layers.Input(shape=(1,), dtype=tf.string)
x = text_vec(input_layer)
x = layers.Embedding(input_dim = total_unique_words, output_dim= 128)(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(32, activation= 'relu')(x)
output_layer =  layers.Dense(1, activation = 'sigmoid')(x)

model_1 = keras.Model(input_layer, output_layer, name = "Dense_Model")
history_1 = compile_and_fit(model_1)

model_1.save('spam_dense_model.keras')