import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.display
ms.use('seaborn-muted')


audio_path = librosa.util.example_audio_file()

y, sr = librosa.load(audio_path)


def mel(y, sr, n_mels=128):
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S


def harmonic_percussive(y, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    S_harmonic = librosa.feature.melspectrogram(y_harmonic, sr=sr)
    S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

    log_Sh = librosa.power_to_db(S_harmonic, ref=np.max)
    log_Sp = librosa.power_to_db(S_percussive, ref=np.max)
    return log_Sh, log_Sp


def chroma(y, sr):
    C = librosa.feature.chroma_cqt(y=y, sr=sr)
    return C


def beat_tracking(y, sr, S):
    plt.figure(figsize=(12, 6))
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats


def beat_sync(C, beats, aggregate=np.median):
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    return C_sync


def beat_sync_cqt(y, sr):
    C = librosa.core.cqt(y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    C_sync = librosa.util.sync(C, beats, aggregate=np.median)
    return C_sync


def beat_sync_stft(y, sr):
    S = librosa.core.stft(y)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    S_sync = librosa.util.sync(S, beats, aggregate=np.median)
    return S_sync
