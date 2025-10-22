import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

filename = './sounds/reggae-style-melody-piano-wet-ups_52bpm.wav'

y, sr = librosa.load(filename , sr=None)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

print(f"Tempo: {tempo}")


# === Feature extraction ===
# 1. Tempo and beats
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
tempo = float(tempo)
print("Tempo : ", tempo)
print("Beat frames dimensions : ", beat_frames.size)
print("Beat frames values : ", beat_frames)

# 2. Onsets (can capture percussive hits like snares)
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

# 3. Pitch estimation using YIN
f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

print("Pitch size : ", len(f0))


# === Time conversion ===
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
time_axis = np.linspace(0, len(y)/sr, num=len(f0))
print("beat timing : ", beat_times[0], beat_times[-1])


# === Visualization ===
plt.figure(figsize=(12, 6))

# Plot waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
plt.vlines(onset_times, -1, 1, color='g', linestyle=':', label='Onsets')
plt.title(f"Waveform with Beats & Onsets (Tempo = {tempo:.2f} BPM)")
plt.legend()

# Plot pitch contour
plt.subplot(2, 1, 2)
plt.plot(time_axis, f0, color='b', linewidth=1)
plt.title("Estimated Pitch (Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()
