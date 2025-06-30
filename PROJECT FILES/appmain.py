from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("rf_model_10features.pkl")
scaler = joblib.load("scaler_10features.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("REQUEST METHOD:", request.method)
        print("FORM DATA RECEIVED:", request.form)

        age = float(request.form.get('age', 0))
        sex = int(request.form.get('sex', 0))
        ascites = int(request.form.get('ascites', 0))
        hepatomegaly = int(request.form.get('hepatomegaly', 0))
        spiders = int(request.form.get('spiders', 0))
        edema = request.form.get('edema', 'N')
        bilirubin = float(request.form.get('bilirubin', 0))
        albumin = float(request.form.get('albumin', 0))
        alk_phos = float(request.form.get('alk_phos', 0))
        sgot = float(request.form.get('sgot', 0))

        edema_map = {'N': 0, 'S': 0.5, 'Y': 1}
        edema_val = edema_map.get(edema, 0)
        features = np.array([[age, sex, ascites, hepatomegaly, spiders, edema_val, bilirubin, albumin, alk_phos, sgot]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]

        result = "Low Risk of Cirrhosis" if prediction == 0 else "High Risk of Cirrhosis"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print("❌ ERROR:", str(e))
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)