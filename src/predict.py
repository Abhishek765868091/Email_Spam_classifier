import torch
import os
from src.model import SpamClassifier
from src.preprocessing import load_and_preprocess

def load_model():
    # Path to saved model
    save_path = os.path.join(os.getcwd(), "saved_models", "spam_model.pth")

    # Load preprocessed data (for vectorizer)
    _, _, _, _, vectorizer = load_and_preprocess()

    # Initialize model
    input_dim = len(vectorizer.get_feature_names_out())
    model = SpamClassifier(input_dim=input_dim)

    # Load weights
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print("✅ Model loaded successfully from:", save_path)
    return model, vectorizer

def predict_email(text):
    model, vectorizer = load_model()

    # Transform input text using same TF-IDF vectorizer
    tfidf_features = vectorizer.transform([text])
    input_tensor = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).float().item()

    return "Spam" if prediction == 1.0 else "Ham"

if __name__ == "__main__":
    # Example test email
    sample_email = "Congratulations! You have won a lottery. Click here to claim."
    result = predict_email(sample_email)
    print("Prediction:", result)
