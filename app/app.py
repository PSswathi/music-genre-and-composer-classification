import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import joblib
import uuid

import numpy as np

from glob import glob
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for, json

from tensorflow import keras

from utils import preprocess_midi_for_bilstm, custom_objects, VALID_COMPOSERS


UPLOAD_FOLDER = "tmp"

app = Flask(__name__)

model = keras.models.load_model("./model/final_bi_lstm_attention_model.keras", compile=False, custom_objects=custom_objects)

scaler = joblib.load("./model/scaler.pkl")
classes = VALID_COMPOSERS

program_freq = {}
with open("./model/program_freq.json", "r") as f:
        program_freq = json.load(f)
        # keys may be strings in JSON, cast to int
        program_freq = {int(k): float(v) for k,v in program_freq.items()}

def list_samples():
    test_dir = os.path.join(app.static_folder, "test")
    if not os.path.isdir(test_dir):
        return []
    paths = []
    for pat in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        paths += glob(os.path.join(test_dir, pat))
    paths.sort()
    return [
        {"name": os.path.basename(p),
         "url": url_for("static", filename=f"test/{os.path.basename(p)}")}
        for p in paths
    ]

@app.get("/")
def index():
    return render_template("index.html", classes=classes, samples=list_samples())

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify(error="Upload a MIDI file with form field 'file'"), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify(error="Empty filename"), 400

    top_k = min(request.form.get("top_k", type=int), len(classes))

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_mid_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    file.save(tmp_mid_path)

    try: 
        X = preprocess_midi_for_bilstm(tmp_mid_path, scaler=scaler, program_freq=program_freq, chunk_size=200)
        if len(X) == 0:
            os.remove(tmp_mid_path)
            return jsonify(error="File has fewer than 200 usable notes after filtering"), 400

        probs = model.predict(X, verbose=0)
        mean_probs = probs.mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()

        idx_sorted = np.argsort(mean_probs)[::-1]
        top_k = max(1, min(top_k, len(idx_sorted)))
        top = []
        for i in idx_sorted[:top_k]:
            label = classes[i] if i < len(classes) else f"class_{i}"
            top.append({"index": int(i), "label": label, "prob": float(mean_probs[i])})

        return jsonify(
            top=top,
            classes=classes,
            probs=[float(p) for p in mean_probs],
            chunks=int(X.shape[0]),
            aggregation="mean",
        )
    except Exception as e:
        try:
            os.remove(tmp_mid_path)
        except Exception as cleanup_error:
            app.logger.warning(f"Failed to clean up temp MIDI file: {cleanup_error}")
        app.logger.error(f"Prediction failed: {e}")
        return jsonify(error=f"Failed to process MIDI file: {e}"), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)