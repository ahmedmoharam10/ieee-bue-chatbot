from flask import Flask, render_template, request, jsonify
import joblib, os, pandas as pd, numpy as np
import time

# Paths
OUTPUTS_DIR = r"E:\Ahmed Sameh Work\Projects\ieee bue chatbot\outputs"
PIPELINE_PATH = os.path.join(OUTPUTS_DIR, "intent_classifier_pipeline.joblib")
METADATA_PATH = os.path.join(OUTPUTS_DIR, "chatbot_metadata.joblib")
TRAINING_DATA_PATH = os.path.join(OUTPUTS_DIR, "training_data.csv")

# Load pipeline and metadata
intent_classifier = joblib.load(PIPELINE_PATH)
metadata = joblib.load(METADATA_PATH)
df = pd.read_csv(TRAINING_DATA_PATH)

# Extract mappings
id2label = metadata["id2label"]
intent_thresholds = metadata["intent_thresholds"]
base_threshold = metadata["BASE_CONFIDENCE_THRESHOLD"]

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_text = request.json.get("message", "")

    # Simulate bot thinking (typing effect)
    time.sleep(1.2)

    # Predict intent
    probabilities = intent_classifier.predict_proba([user_text])[0]
    pred_id = np.argmax(probabilities)
    confidence = float(probabilities[pred_id])
    intent = id2label[pred_id]

    # Threshold check
    threshold = intent_thresholds.get(intent, base_threshold)
    if confidence < threshold:
        bot_response = "🤔 I'm not sure what you mean. Can you rephrase?"
    else:
        intent_responses = df[df["Intent"] == intent]["Bot Response"].tolist()
        bot_response = intent_responses[0] if intent_responses else "Sorry, I don’t have an answer."

    return jsonify({
        "user": user_text,
        "intent": intent,
        "confidence": round(confidence, 2),
        "bot": bot_response
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
