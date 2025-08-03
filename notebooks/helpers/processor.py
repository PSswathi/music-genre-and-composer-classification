import pretty_midi
import mido
from  mido import MidiTrack
import os
import io


import pandas as pd 
import numpy as np 

from collections import Counter
from warnings import catch_warnings

def extract_chunk_features(notes, duration, num_instruments, tempo):
    pitches = []
    velocities = []
    durations = []


    for n in notes:
        pitches.append(n.pitch)
        velocities.append(n.velocity)
        durations.append(n.end - n.start)

    # Chord sizes
    chord_sizes = []
    i = 0
    while i < len(notes):
        chord = [notes[i]]
        j = i + 1
        while j < len(notes) and abs(notes[j].start - notes[i].start) < 0.05:
            chord.append(notes[j])
            j += 1
        chord_sizes.append(len(chord))
        i = j

    return {
        'tempo': tempo,
        'num_instruments': num_instruments,
        'duration': duration,
        'note_count': len(notes),
        'avg_pitch': np.mean(pitches) if pitches else 0,
        'pitch_range': max(pitches) - min(pitches) if pitches else 0,
        'std_pitch': np.std(pitches) if pitches else 0,
        'most_common_pitch': Counter(pitches).most_common(1)[0][0] if pitches else 0,
        'avg_duration': np.mean(durations) if durations else 0,
        'std_duration': np.std(durations) if durations else 0,
        'note_density': len(notes) / duration if duration > 0 else 0,
        'velocity_mean': np.mean(velocities) if velocities else 0,
        'velocity_std': np.std(velocities) if velocities else 0,
        'avg_chord_size': np.mean(chord_sizes) if chord_sizes else 0,
        'chord_density': len(chord_sizes) / duration if duration > 0 else 0
    }

def process_dir_to_chunks(dir_path, chunk_size=200):
    midi_data = []

    for subdir, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                file_path = os.path.join(subdir, file)
                composer = os.path.relpath(file_path, dir_path).split(os.sep)[0]

                with catch_warnings(record=True) as w:
                    midi = pretty_midi.PrettyMIDI(file_path)
                    if w:
                        midi = pretty_midi.PrettyMIDI(fix_midi_metadata(file_path))
                    
                    tempo = midi.estimate_tempo()
                    num_instruments = len(midi.instruments)

                    all_notes = []
                    for instrument in midi.instruments:
                        all_notes.extend(instrument.notes)

                    if len(all_notes) < chunk_size:
                        continue  # skip short files

                    # Sort notes chronologically
                    all_notes.sort(key=lambda n: n.start)

                    # Split into chunks of 200 notes
                    for i in range(0, len(all_notes), chunk_size):
                        chunk_notes = all_notes[i:i+chunk_size]
                        if len(chunk_notes) < chunk_size:
                            break  # discard short last chunk

                        chunk_duration = chunk_notes[-1].end - chunk_notes[0].start
                        features = extract_chunk_features(chunk_notes, chunk_duration, num_instruments, tempo)

                        midi_data.append({
                            'composer': composer,
                            'filename': file,
                            **features
                        })
                
                

    return pd.DataFrame(midi_data)

def fix_midi_metadata(midi_file):
    """
    Fixes the metadata of a MIDI file by removing any non-standard metadata.
    """
    midi = mido.MidiFile(midi_file)

    tracks = []
    tracks.append(midi.tracks[0])
    for track in midi.tracks[1:]: 
        newTrack = MidiTrack()
        newTrack.name = track.name
        for e in track:
            if  e.type not in ('set_tempo', 'key_signature', 'time_signature'):
                newTrack.append(e)

        tracks.append(newTrack)
    
    fixed = mido.MidiFile(
        type=midi.type,
        ticks_per_beat=midi.ticks_per_beat,
        charset=midi.charset if hasattr(midi, 'charset') else 'utf-8',
        debug=midi.debug if hasattr(midi, 'debug') else False,
        clip=midi.clip if hasattr(midi, 'clip') else False,
        tracks=tracks
    )

    buf = io.BytesIO()
    fixed.save(file=buf)
    buf.seek(0)
    
    return buf