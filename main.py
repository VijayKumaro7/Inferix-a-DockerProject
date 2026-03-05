from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os
import time

app = Flask(__name__, template_folder="/app/templates")

# ── Load Models ───────────────────────────────────────────
models = {}

def load_models():
    model_dir = "/app/models"
    for fname in os.listdir(model_dir):
        if fname.endswith(".pkl"):
            name = fname.replace("_model.pkl", "")
            with open(os.path.join(model_dir, fname), "rb") as f:
                models[name] = pickle.load(f)
    print(f"✅ Loaded models: {list(models.keys())}")

load_models()

# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "models_loaded": list(models.keys()), "timestamp": time.time()})

@app.route("/models")
def list_models():
    result = {}
    for name, data in models.items():
        result[name] = {
            "accuracy": round(data["accuracy"] * 100, 2),
            "classes": data["classes"],
        }
    return jsonify(result)

@app.route("/predict/iris", methods=["POST"])
def predict_iris():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        model_data = models["iris"]
        prediction = model_data["model"].predict(features)[0]
        probabilities = model_data["model"].predict_proba(features)[0]
        class_name = model_data["classes"][prediction]

        return jsonify({
            "prediction": int(prediction),
            "class_name": class_name,
            "confidence": round(float(max(probabilities)) * 100, 2),
            "probabilities": {
                model_data["classes"][i]: round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/cancer", methods=["POST"])
def predict_cancer():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        model_data = models["cancer"]
        features_scaled = model_data["scaler"].transform(features)
        prediction = model_data["model"].predict(features_scaled)[0]
        probabilities = model_data["model"].predict_proba(features_scaled)[0]
        class_name = model_data["classes"][prediction]

        return jsonify({
            "prediction": int(prediction),
            "class_name": class_name,
            "confidence": round(float(max(probabilities)) * 100, 2),
            "probabilities": {
                model_data["classes"][i]: round(float(p) * 100, 2)
                for i, p in enumerate(probabilities)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
