import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (8,5)
file = "./data/blues00000.wav"

signal, sample_rate = librosa.load(file, sr=22050)
# print(signal.shape, sample_rate)

# WAVEFORM(amplitude, time)
#display waveform
# plt.figure(figsize=FIG_SIZE)
# librosa.display.waveshow(signal, sr=sample_rate, color="blue")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Waveform")


# FFT ()-> power spectrum

fft = np.fft.fft(signal)#perform fourier transform(magnitude, frequency)
spectrum = np.abs(fft)#get magnitude
# print(spectrum)
f = np.linspace(0, sample_rate, len(spectrum))
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

# plt.figure(figsize=FIG_SIZE)
# plt.plot(left_f, left_spectrum, alpha=0.4)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.title("Power spectrum")

# stt -> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)
# plt.figure(figsize=FIG_SIZE)
# librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.title("Spectrogram")

# MFCCs
mfccs = librosa.feature.mfcc(y=signal,n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(mfccs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()



plt.show()