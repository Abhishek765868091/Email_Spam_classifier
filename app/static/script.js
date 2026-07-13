import plotly.graph_objs as go
import plotly
import json

@app.route("/predict", methods=["POST"])
def predict():
    ...
    risk_score = float(output.item()) * 100  # model output ko percentage me lo

    chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Spam Risk (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green" if prediction == 0 else "red"}
        }
    ))

    # Proper JSON encoding for Plotly chart
    chart_json = json.dumps(chart, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("index.html",
                           prediction_text=result,
                           chart_json=chart_json)
