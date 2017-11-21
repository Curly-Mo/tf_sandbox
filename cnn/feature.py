import numpy as np
import librosa
import fma


def one_hot(label_ids, label_dict):
    indices = [label_dict[id]['index'] for id in label_ids]
    y = np.zeros([1, len(label_dict)])
    y[0, indices] = 1
    return y


def training_features(x_files, y_labels, label_dict):
    win_size = 32
    hop_size = win_size*7//8
    y_train = np.vstack([one_hot(y, label_dict) for y in y_labels])

    x_train_spec = (mel_spec(x) for x in x_files)
    X = []
    Y = []
    for i, (x, y) in enumerate(zip(x_train_spec, y_train)):
        print(i)
        x = split_spec(x, win_size, hop_size)
        if x is not None:
            X.append(x)
            for i in range(len(x)):
                Y.append(y)
    X = np.vstack(X)
    Y = np.vstack(Y)

    labels = {label['index']: label for label in label_dict.values()}
    return X, Y, labels


def mel_spec(audio_path, n_fft=2048):
    print(audio_path)
    y, sr = librosa.load(audio_path, mono=True)
    melspec = librosa.feature.melspectrogram(y, n_fft=n_fft)
    return melspec.T


def split_spec(S, win_size, hop_size):
    X = []
    i = 0
    while i + win_size < len(S):
        x = S[i:i+win_size]
        X.append(x[np.newaxis, :, :, np.newaxis])
        i += hop_size
    if not X:
        return None
    return np.vstack(X)


def load_fma():
    X, Y, labels = fma.get_dataset()
    X, Y, labels = training_features(X, Y, labels)
    return X, Y, labels
