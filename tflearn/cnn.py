import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, global_max_pool, global_avg_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import librosa
import numpy as np
import feature
import collections


def network(input_shape, output_shape, weight=1.0, activation='softmax', loss='categorical_crossentropy'):
    # Building convolutional net
    net = input_data(shape=[None, *input_shape], name='input')
    net = conv_2d(net, 48, [4, 128], activation='relu', regularizer="L2")
    net = max_pool_2d(net, 2)
    net = conv_2d(net, 48, [4, 128], activation='relu', regularizer="L2")
    net = max_pool_2d(net, 2)
    net = conv_2d(net, 48, [4, 128], activation='relu', regularizer="L2")
    global_max = global_max_pool(net)
    global_average = global_avg_pool(net)
    net = merge([global_max, global_average], 'concat')
    net = fully_connected(net, 2048, activation='relu')
    net = dropout(net, 0.8)
    net = fully_connected(net, 2048, activation='relu')
    net = dropout(net, 0.8)
    net = fully_connected(net, output_shape, activation='softmax')

    def wloss(y_pred, y_true):
        return tflearn.weighted_crossentropy(y_pred, y_true, weight=weight)
    net = regression(net,
                     optimizer='adam',
                     learning_rate=0.001,
                     loss=loss,
                     name='target')
    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


def network_simple(input_shape, output_shape, weight=1.0):
    # Building convolutional net
    net = input_data(shape=[None, *input_shape], name='input')
    net = conv_2d(net, 128, [4, 128], activation='relu', regularizer="L2")
    global_max = global_max_pool(net)
    global_average = global_avg_pool(net)
    net = merge([global_max, global_average], 'concat')
    net = fully_connected(net, 2048, activation='relu')
    net = dropout(net, 0.8)
    net = fully_connected(net, output_shape, activation='softmax')

    def wloss(y_pred, y_true):
        return tflearn.weighted_crossentropy(y_pred, y_true, weight=weight)
    net = regression(net, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


def train(model, X, Y, n_epoch=10, batch_size=10):
    print(X.shape)
    model.fit({'input': X}, {'target': Y}, n_epoch=n_epoch, batch_size=batch_size,
              # validation_set=({'input': testX}, {'target': testY}),
              show_metric=True)


def predict(model, audio_path, labels=None, sample=None, offset=0.0, duration=None):
    win_size = 64
    hop_size = win_size*15//16
    S = feature.mel_spec(audio_path, offset=offset, duration=duration)
    print(S.shape)
    X = feature.split_spec(S, win_size, hop_size)
    print(X.shape)
    del S
    X = X[..., np.newaxis]
    if sample and X.shape[0] > sample:
        X = X[np.random.choice(X.shape[0], sample, replace=False)]
    y = model.predict(X)
    Y = reversed(sorted([(i, float(val)) for i, val in enumerate((sum(y)/len(y)))], key=lambda x: x[1]))
    if labels:
        Y = [(labels[i], val) for i, val in Y]
    return y, Y


def main():
    win_size = 32
    audio_path = librosa.util.example_audio_file()
    S = feature.mel_spec(audio_path)
    X = feature.split_spec(S, win_size, win_size//4)
    Y = np.vstack([0, 1] for _ in X)
    model = network(X.shape[1:], Y.shape[-1])
    train(model, X, Y)
    return model


def balanced_sample(X_in, Y_in, count):
    X, Y = [], []
    counts = collections.defaultdict(int)
    for x, y in zip(X_in, Y_in):
        for indices in y.nonzero():
            if all(counts[i] < count for i in indices):
                X.append(x)
                Y.append(y)
                for i in indices:
                    counts[i] += 1
    return np.stack(X), np.stack(Y)


def network_tmp(input_shape, output_shape):
    # Building convolutional net
    net = input_data(shape=[None, *input_shape[1:]], name='input')
    net = fully_connected(net, 128, activation='tanh')
    net = dropout(net, 0.8)
    net = fully_connected(net, 128, activation='tanh')
    net = dropout(net, 0.8)
    net = fully_connected(net, output_shape, activation='softmax')

    net = regression(net, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')
    model = tflearn.DNN(net, tensorboard_verbose=3)
    return model


def train_tmp(model, X, Y, n_epoch=5, batch_size=20):
    X = X[:, 1, :, :]
    model.fit({'input': X}, {'target': Y}, n_epoch=n_epoch, batch_size=batch_size,
              # validation_set=({'input': testX}, {'target': testY}),
              show_metric=True)
