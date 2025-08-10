
import pandas as pd
import numpy as np
import pretty_midi
import joblib
import json
import base64
import tempfile
import os

from midi2audio import FluidSynth

VALID_COMPOSERS = ['Bach', 'Beethoven', 'Chopin', 'Mozart']

composer_list = VALID_COMPOSERS  # Already filtered
composer_to_id = {name: idx for idx, name in enumerate(composer_list)}

# Define attention layer
import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer

@keras.utils.register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        e = tf.math.tanh(tf.linalg.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = inputs * a
        return tf.reduce_sum(output, axis=1)
    

ACCENT_THRESHOLD = 80  # same as in training

FEATURE_NAMES = [
    'Pitch','Duration','Velocity','DeltaTime','Interval','Program_FE','Accented',
    'Tempo','PitchRange','NoteDensity','RepetitionRate','AvgInterval','RhythmicVariety','ChordDensity'
]

def preprocess_midi_for_bilstm(
    midi_path: str,
    scaler=None,                 # pass a fitted StandardScaler OR a path to scaler.pkl
    program_freq=None,           # pass a dict {program:int -> freq:float} OR a path to program_freq.json
    chunk_size: int = 200
):
    """
    Returns:
        X: np.ndarray with shape (num_chunks, 200, 14)
           Only full 200-note chunks are returned (no padding), matching training.
           If the piece has <200 usable notes, returns shape (0, 200, 14).
    """
    features = []
    # Load scaler if a path was provided
    if isinstance(scaler, str):
        with open(scaler, "rb") as f:
            scaler = joblib.load(f)

    # Load program frequency map if a path was provided
    if isinstance(program_freq, str):
        with open(program_freq, "r") as f:
            program_freq = json.load(f)
        # keys may be strings in JSON, cast to int
        program_freq = {int(k): float(v) for k,v in program_freq.items()}
    if program_freq is None:
        # default: empty map -> unseen becomes 0
        program_freq = {}

    # Parse MIDI
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Global tempo (fallback if estimate fails)
    try:
        tempo_bpm = float(midi.estimate_tempo())
    except Exception:
        tempo_bpm = 120.0

    # Gather non-drum notes and their programs
    # notes = []
    # programs = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        # prog = int(getattr(inst, "program", 0))
        # for n in inst.notes:
        #     notes.append(n)
        #     programs.append(prog)
        instrument_program = inst.program
        notes = sorted(inst.notes, key=lambda n: n.start)
        if len(notes) < chunk_size:
            continue

        for i in range(0, len(notes) - chunk_size + 1, chunk_size):
            chunk = notes[i:i+chunk_size]
            chunk_start = chunk[0].start
            chunk_end = chunk[-1].end
            chunk_duration = chunk_end - chunk_start

            # ---- Higher-level stats ----
            pitches = [n.pitch for n in chunk]
            durations = [n.end - n.start for n in chunk]
            intervals = [pitches[i] - pitches[i-1] for i in range(1, len(pitches))]

            pitch_range = max(pitches) - min(pitches)
            note_density = len(chunk) / (chunk_duration + 1e-6)  # notes/sec
            repetition_rate = sum([1 for i in range(1, len(pitches)) if pitches[i] == pitches[i-1]]) / len(pitches)
            avg_interval = np.mean(intervals) if intervals else 0.0
            rhythmic_variety = np.std(durations)
            chord_density = sum([1 for i in range(1, len(chunk)) if chunk[i].start < chunk[i-1].end]) / len(chunk)

            # ---- Per-note features ----
            prev_start = chunk[0].start
            prev_pitch = chunk[0].pitch

            for note in chunk:
                pitch = note.pitch
                start = note.start
                end = note.end
                duration = end - start
                delta_time = start - prev_start
                velocity = note.velocity
                interval = pitch - prev_pitch
                is_accented = 1 if velocity > 80 else 0

                features.append({
                    "Pitch": pitch,
                    "Duration": duration,
                    "Velocity": velocity,
                    "DeltaTime": delta_time,
                    "Start": start,
                    "End": end,
                    "Interval": interval,
                    "Program": instrument_program,
                    "Accented": is_accented,
                    "Tempo": tempo_bpm,
                    "PitchRange": pitch_range,
                    "NoteDensity": note_density,
                    "RepetitionRate": repetition_rate,
                    "AvgInterval": avg_interval,
                    "RhythmicVariety": rhythmic_variety,
                    "ChordDensity": chord_density
                })

                prev_start = start
                prev_pitch = pitch

    df = pd.DataFrame(features)
    df['Program_FE'] = df['Program'].map(program_freq)
    df['Program_FE'] = df['Program_FE'].fillna(0)

    df[FEATURE_NAMES] = scaler.transform(df[FEATURE_NAMES])

    X_chunks = []
    features = df[FEATURE_NAMES].values
    
    # Chunk features and assign label to each chunk
    for i in range(0, len(features) - chunk_size + 1, chunk_size):
        chunk = features[i:i + chunk_size]
        X_chunks.append(chunk)

    return np.array(X_chunks, dtype=np.float32)

custom_objects = {
    "AttentionLayer": AttentionLayer,
}

def midi_to_wav_data_url(app, midi_bytes: bytes) -> str | None:
    """Render MIDI bytes to WAV and return a data: URL. Uses a temp dir and cleans up."""
    try:
        with tempfile.TemporaryDirectory() as td:
            mid_path = os.path.join(td, "in.mid")
            wav_path = os.path.join(td, "out.wav")
            with open(mid_path, "wb") as f:
                f.write(midi_bytes)
            FluidSynth(sound_font="./soundfont/GeneralUser-GS.sf2").midi_to_audio(mid_path, wav_path)
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()
        b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/wav;base64,{b64}"
    except Exception as e:
        app.logger.warning(f"Audio render failed: {e}")
        return None
