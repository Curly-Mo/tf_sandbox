import datetime
import librosa
import tflearn
import numpy as np
import argparse
import tensorflow as tf


def features_from_audio(y, sr, n_fft=512, seq_len=400):
    S = librosa.core.stft(y, n_fft=n_fft).astype(np.complex64)
    return features_from_spectrogram(S, sr, seq_len=seq_len)


def features_from_spectrogram(S, sr, seq_len=250, win_length=1):
    X, Y = [], []
    for i in range(len(S.T)-seq_len):
        x, y = S[:, i:i+seq_len], S[:, i+seq_len:i+seq_len+win_length]
        X.append(x.T.reshape([-1, x.T.shape[0], x.T.shape[1]]))
        Y.append(y.T)
    return np.vstack(X), np.vstack(Y)


def network(seq_len, n_features):
    init_state = (tf.ones([20, 3], dtype=tf.complex64), tf.ones([20, 3], dtype=tf.complex64))
    input = tf.placeholder(shape=(None, seq_len, n_features), dtype=tf.complex64)
    net = tflearn.input_data(placeholder=input, dtype=tf.complex64)
    net = tflearn.lstm(net, 3, initial_state=init_state)
    net = tflearn.dropout(net, 0.8)
    net = tflearn.fully_connected(net, n_features)
    net = tflearn.regression(net, optimizer='adam', loss='mean_square', dtype=tf.complex64)

    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


def generate(model, X, seq_len):
    Y = []
    for i in range(seq_len):
        y = model.predict([X])
        Y.append(y)
        X = np.vstack([X[1:], y])
    Y = np.vstack(Y)
    return Y.T


def main(audio_path=librosa.util.example_audio_file(), output='test.wav', load=None):
    y, sr = librosa.load(audio_path)
    X, Y = features_from_audio(y, sr)
    seq_len = X.shape[1]
    n_features = X.shape[2]
    model = network(seq_len, n_features)
    if load:
        model.load(load)
    else:
        model.fit(X, Y, n_epoch=5, batch_size=20)
        model.save(f"model/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.tflearn")
    Y_out = generate(model, X[3000], len(X))
    waveform = librosa.istft(Y_out)
    librosa.output.write_wav(output, waveform, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()
    main(args.input, args.output, args.load)
