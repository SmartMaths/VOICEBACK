
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "voiceback_model.joblib"

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": [
            "http://127.0.0.1:5500",
            "http://localhost:5500"
        ]
    }
})

# Load the trained model once when the server starts.
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "app": "VOICEBACK",
        "status": "online",
        "message": "Prediction API is ready."
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object."}), 400

    landmarks = data.get("landmarks")

    if not isinstance(landmarks, list) or len(landmarks) != 63:
        return jsonify({
            "error": "Expected 63 hand-coordinate values."
        }), 400

    try:
        features = np.asarray(landmarks, dtype=np.float32)
    except (TypeError, ValueError):
        return jsonify({"error": "Coordinates must be numbers."}), 400

    if not np.isfinite(features).all():
        return jsonify({"error": "Coordinates must be finite."}), 400

    prediction = model.predict(features.reshape(1, -1))[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features.reshape(1, -1))[0]
        confidence = float(np.max(probabilities))

    return jsonify({
        "prediction": str(prediction),
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
