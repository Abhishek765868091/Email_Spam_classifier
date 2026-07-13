from flask import Flask, request, render_template
import torch
import joblib
import json
import plotly
import plotly.graph_objs as go
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

app = Flask(__name__)

# Path to Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model architecture (same as training)
class SpamClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)   # 👈 match training
        self.fc2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Load vectorizer
vectorizer = joblib.load("saved_models/vectorizer.pkl")

# Load model weights
model = SpamClassifier(input_dim=vectorizer.max_features)
model.load_state_dict(torch.load("saved_models/spam_model.pth"))
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = ""

    # Textarea input
    if "email" in request.form and request.form["email"].strip() != "":
        email_text = request.form["email"]

    # File upload input
    elif "file" in request.files:
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename.lower()

        if filename.endswith(".pdf"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            for page in doc:
                email_text += page.get_text()

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(uploaded_file)
            email_text = pytesseract.image_to_string(image)

    if not email_text.strip():
        return render_template("index.html", prediction_text="⚠️ No text found!", chart_json=None)

    # Vectorize text
    tfidf_features = vectorizer.transform([email_text])
    input_tensor = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)

    prob = float(output.item())

    if prob >= 0.5:
        result = "🚨 Spam Detected!"
    else:
        result = "✅ Safe (Ham)"

    # Risk chart
    risk_score = prob * 100
    chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Spam Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green" if prob < 0.5 else "red"}
        }
    ))
    chart_json = json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("index.html",
                           prediction_text=result,
                           chart_json=chart_json)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
