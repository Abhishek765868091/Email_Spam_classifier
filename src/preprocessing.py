import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv("data/enron_spam_data.csv")
    print("Dataset load with shape:", df.shape)
    print("Columns:", df.columns)

    # Combine subject + message into one text field
    df['text'] = df['Subject'].fillna('') + " " + df['Message'].fillna('')

    # Normalize labels to lowercase first
    df['Spam/Ham'] = df['Spam/Ham'].str.lower()

    # Convert labels (ham → 0, spam → 1)
    df['Spam/Ham'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})

    # Drop rows where label mapping failed (NaN)
    df = df.dropna(subset=['Spam/Ham'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['Spam/Ham'], test_size=0.2, random_state=42
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

if __name__ == "__main__":
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess()
    print("Training data shape:", X_train_tfidf.shape)
    print("Testing data shape:", X_test_tfidf.shape)
    print("Sample labels:", y_train[:10].values)
