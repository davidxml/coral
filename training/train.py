import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

def train_model():
    print("Loading dataset...")
    df = pd.read_csv("../data/SMSSpamCollection.csv", encoding="latin-1")
    
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42  # This makes it reproducible
    )

    print("Vectorizing text (TF-IDF)...")
    # TfidfVectorizer mathematically weighs the rarity and importance of each word
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("Training Naive Bayes classifier...")
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test_vectorized)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    # F1 score requires us to specify which label is the "positive" class (spam)
    f1 = f1_score(y_test, predictions, pos_label='spam')

    print("-" * 30)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("-" * 30)

    print("Exporting artifacts...")
    # Save the trained vectorizer and model to disk so FastAPI can load them
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(model, "models/naive_bayes_model.joblib")
    print("Done! Artifacts saved to the /models directory.")

if __name__ == "__main__":
    train_model()