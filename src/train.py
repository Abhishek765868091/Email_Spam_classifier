# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# from src.model import SpamClassifier
# from src.preprocessing import load_and_preprocess

# def train_model():
#     # 1. Load preprocessed data
#     X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess()

#     # 2. Convert to tensors
#     x_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
#     x_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
#     y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

#     # 3. Initialize model
#     input_dim = x_train_tensor.shape[1]
#     model = SpamClassifier(input_dim=input_dim)

#     # 4. Loss + optimizer
#     criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # 5. Training loop
#     epochs = 5
#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(x_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

#     # 6. Evaluation
#     with torch.no_grad():
#         model.eval()
#         test_outputs = model(x_test_tensor)
#         predicted = (test_outputs >= 0.5).float()
#         accuracy = (predicted.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
#         print(f"Test Accuracy: {accuracy*100:.2f}%")

#     # 7. Save trained model
#     save_dir = os.path.join(os.getcwd(), "saved_models")
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "spam_model.pth")

#     torch.save(model.state_dict(), save_path)
#     print("✅ Model saved successfully at:", save_path)

# if __name__ == "__main__":
#     train_model()
import torch
import torch.nn as nn
import torch.optim as optim
import os
import joblib
from src.model import SpamClassifier
from src.preprocessing import load_and_preprocess

def train_model():
    # 1. Load preprocessed data (TF-IDF features + labels + vectorizer)
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = load_and_preprocess()

    # 2. Convert to tensors
    x_train_tensor = torch.tensor(X_train_tfidf.toarray(), dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # 3. Initialize model
    input_dim = x_train_tensor.shape[1]
    model = SpamClassifier(input_dim=input_dim)

    # 4. Loss + optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 6. Evaluation
    with torch.no_grad():
        model.eval()
        test_outputs = model(x_test_tensor)
        predicted = (test_outputs >= 0.5).float()
        accuracy = (predicted.eq(y_test_tensor).sum().item()) / y_test_tensor.size(0)
        print(f"Test Accuracy: {accuracy*100:.2f}%")

    # 7. Save trained model + vectorizer
    save_dir = os.path.join(os.getcwd(), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "spam_model.pth")
    vectorizer_path = os.path.join(save_dir, "vectorizer.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("✅ Model saved at:", model_path)
    print("✅ Vectorizer saved at:", vectorizer_path)

if __name__ == "__main__":
    train_model()
