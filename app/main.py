# from flask import Flask, request, render_template
# import torch
# import joblib
# import json
# import plotly
# import plotly.graph_objs as go
# import pytesseract
# from PIL import Image
# import fitz  # PyMuPDF

# app = Flask(__name__)

# # Path to Tesseract OCR
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Model architecture (same as training)
# class SpamClassifier(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(SpamClassifier, self).__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 128)   # 👈 match training
#         self.fc2 = torch.nn.Linear(128, 1)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = self.sigmoid(self.fc2(x))
#         return x

# # Load vectorizer
# vectorizer = joblib.load("saved_models/vectorizer.pkl")

# # Load model weights
# model = SpamClassifier(input_dim=vectorizer.max_features)
# model.load_state_dict(torch.load("saved_models/spam_model.pth"))
# model.eval()

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     email_text = ""

#     # Textarea input
#     if "email" in request.form and request.form["email"].strip() != "":
#         email_text = request.form["email"]

#     # File upload input
#     elif "file" in request.files:
#         uploaded_file = request.files["file"]
#         filename = uploaded_file.filename.lower()

#         if filename.endswith(".pdf"):
#             doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#             for page in doc:
#                 email_text += page.get_text()

#         elif filename.endswith((".png", ".jpg", ".jpeg")):
#             image = Image.open(uploaded_file)
#             email_text = pytesseract.image_to_string(image)

#     if not email_text.strip():
#         return render_template("index.html", prediction_text="⚠️ No text found!", chart_json=None)

#     # Vectorize text
#     tfidf_features = vectorizer.transform([email_text])
#     input_tensor = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)

#     # Prediction
#     with torch.no_grad():
#         output = model(input_tensor)

#     prob = float(output.item())

#     if prob >= 0.5:
#         result = "🚨 Spam Detected!"
#     else:
#         result = "✅ Safe (Ham)"

#     # Risk chart
#     risk_score = prob * 100
#     chart = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=risk_score,
#         title={"text": "Spam Risk (%)"},
#         gauge={
#             "axis": {"range": [0, 100]},
#             "bar": {"color": "green" if prob < 0.5 else "red"}
#         }
#     ))
#     chart_json = json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder)

#     return render_template("index.html",
#                            prediction_text=result,
#                            chart_json=chart_json)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)
from flask import Flask, request, render_template
import torch
import joblib
import json
import plotly
import plotly.graph_objs as go
import pytesseract
from PIL import Image
import fitz
import os

app = Flask(__name__)

# Cross-platform Tesseract path
if os.name == "nt":  # Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:  # Linux (Render)
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# main.py is inside app/, so go one level up to reach project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SpamClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

vectorizer = joblib.load(os.path.join(BASE_DIR, "saved_models", "vectorizer.pkl"))
input_dim = len(vectorizer.get_feature_names_out())
model = SpamClassifier(input_dim=input_dim)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "saved_models", "spam_model.pth"), map_location=torch.device("cpu")))
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = ""
    error_msg = None

    if "email" in request.form and request.form["email"].strip() != "":
        email_text = request.form["email"]
    elif "file" in request.files and request.files["file"].filename != "":
        uploaded_file = request.files["file"]
        filename = uploaded_file.filename.lower()
        try:
            if filename.endswith(".pdf"):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc:
                    email_text += page.get_text()
                doc.close()
            elif filename.endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(uploaded_file)
                email_text = pytesseract.image_to_string(image)
                image.close()
            else:
                error_msg = "Unsupported file type. Please upload PDF, PNG, or JPG."
        except pytesseract.TesseractNotFoundError:
            error_msg = "OCR engine not available on server. Please paste text instead."
        except Exception as e:
            error_msg = f"Could not process file: {str(e)}"

    if error_msg:
        return render_template("index.html", prediction_text=error_msg, chart_json=None)

    if not email_text.strip():
        return render_template("index.html", prediction_text="No text found!", chart_json=None)

    tfidf_features = vectorizer.transform([email_text])
    input_tensor = torch.tensor(tfidf_features.toarray(), dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    prob = float(output.item())
    result = "Spam Detected!" if prob >= 0.5 else "Safe (Ham)"
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
    return render_template("index.html", prediction_text=result, chart_json=chart_json)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)