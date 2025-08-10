import joblib
import json
import os


import tempfile
import numpy as np

from glob import glob
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for

from tensorflow import keras

from utils import preprocess_midi_for_bilstm, custom_objects, midi_to_wav_data_url, VALID_COMPOSERS


UPLOAD_FOLDER = "tmp"

app = Flask(__name__)
# app.secret_key = "AVerySecretKey"

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

    # Read the uploaded file into memory (no persistent save)
    midi_bytes = file.read()

    # Use a temp dir for PrettyMIDI preprocessing only during this request
    with tempfile.TemporaryDirectory() as td:
        tmp_mid = os.path.join(td, secure_filename(file.filename) or "upload.mid")
        with open(tmp_mid, "wb") as f:
            f.write(midi_bytes)

        X = preprocess_midi_for_bilstm(tmp_mid, scaler=scaler, program_freq=program_freq, chunk_size=200)
        if X.shape[0] == 0:
            # nothing persisted; temp dir auto-deletes here
            return jsonify(error="File has fewer than 200 usable notes after filtering"), 400

        probs = model.predict(X, verbose=0)
        mean_probs = probs.mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()

    # Optionally render audio preview to a data URL (also uses its own temp dir)
    audio_data_url = midi_to_wav_data_url(app, midi_bytes)

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
        audio_data_url=audio_data_url 
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)