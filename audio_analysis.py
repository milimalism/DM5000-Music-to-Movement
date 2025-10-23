import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# filename = './sounds/reggae-style-melody-piano-wet-ups_52bpm.wav'
filename = './sounds/pumped up kicks clipped.mov'

y, sr = librosa.load(filename , sr=None)

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
print("Pitch size : ", f0.shape)
mean_pitch = np.mean(f0)
std_pitch = np.std(f0)
print("Pitch mean : ", np.mean(f0), " Std : ", np.std(f0))


# 4. Rms (root mean square energy) extraction
rms = librosa.feature.rms(y=y)
print("Rms dimension : ", rms.shape)
print("Rms mean : ", np.mean(rms[0]), " Std : ", np.std(rms[0]))
frames = range(len(rms[0]))
times = librosa.frames_to_time(frames, sr=sr)


# === Time conversion ===
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)
time_axis = np.linspace(0, len(y)/sr, num=len(f0))
frames = range(len(rms[0]))
rms_times = librosa.frames_to_time(beat_frames, sr=sr)


# === Visualization ===
plt.figure(figsize=(12, 6))

# Plot waveform
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.vlines(beat_times, -1, 1, color='r', linestyle='--', label='Beats')
plt.vlines(onset_times, -1, 1, color='g', linestyle=':', label='Onsets')
plt.plot(times, rms[0], color='b', label='RMS Energy (Loudness)')
plt.title(f"Waveform with Beats & Onsets (Tempo = {tempo:.2f} BPM)")
plt.legend()

# Plot pitch contour
plt.subplot(2, 1, 2)
plt.plot(frames, f0, color='b', linewidth=1)
plt.axhline(mean_pitch, color='orange', linestyle='--', label='Mean')
plt.axhline(mean_pitch + std_pitch, color='green', linestyle='--', label='Mean + 1 STD')
plt.axhline(mean_pitch - std_pitch, color='green', linestyle='--', label='Mean - 1 STD')
plt.axhline(mean_pitch + 2*std_pitch, color='red', linestyle='--', label='Mean + 2 STD')
plt.axhline(mean_pitch - 2*std_pitch, color='red', linestyle='--', label='Mean - 2 STD')
plt.title("Estimated Pitch (Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()
plt.show()