import joblib
import json
import os
import uuid
import base64

import numpy as np

from glob import glob
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, url_for

from midi2audio import FluidSynth

from tensorflow import keras

from utils import preprocess_midi_for_bilstm, custom_objects, VALID_COMPOSERS


UPLOAD_FOLDER = "tmp"

app = Flask(__name__)
app.secret_key = "AVerySecretKey"

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

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_mid_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    
    with open(tmp_mid_path, "wb") as f:
        f.write(midi_bytes)

    try: 
        X = preprocess_midi_for_bilstm(tmp_mid_path, scaler=scaler, program_freq=program_freq, chunk_size=200)
        if len(X) == 0:
            # nothing persisted; temp dir auto-deletes here
            os.remove(tmp_mid_path)
            return jsonify(error="File has fewer than 200 usable notes after filtering"), 400

        probs = model.predict(X, verbose=0)
        mean_probs = probs.mean(axis=0)
        mean_probs = mean_probs / mean_probs.sum()

        # render audio preview to a data URL (also uses its own temp dir)
        audio_data_url = midi_to_wav_data_url(tmp_mid_path)

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
    except Exception as e:
        os.remove(tmp_mid_path)
        app.logger.error(f"Prediction failed: {e}")
        return jsonify(error="Failed to process MIDI file"), 500

def midi_to_wav_data_url(filename: str) -> str | None:
    """Render MIDI bytes to WAV and return a data: URL. Uses a temp dir and cleans up."""
    wav_path = os.path.join(UPLOAD_FOLDER, "out-" + secure_filename(filename) + ".wav")

    try:
        FluidSynth(sound_font="./soundfont/GeneralUser-GS.sf2").midi_to_audio(filename, wav_path)

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        b64 = base64.b64encode(wav_bytes).decode("ascii")

    except Exception as e:
        app.logger.warning(f"Audio render failed: {e}")
        os.remove(wav_path)
        os.remove(filename)
        return None

    try: 
        os.remove(wav_path)
        os.remove(filename)
    except Exception as e:
        app.logger.warning(f"Failed to clean up temp files: {e}")

    return f"data:audio/wav;base64,{b64}"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
    # tttt = "./tmp/e7d0a7df11384b138e88e7f19b3a457c_Contradance_n1.mid"
    # preprocess_midi_for_bilstm(tttt, scaler=scaler, program_freq=program_freq, chunk_size=200)