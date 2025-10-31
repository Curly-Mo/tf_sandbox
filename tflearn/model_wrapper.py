import os
import functools
import pathlib
import datetime
import distutils.dir_util

# import tensorflow as tf
import tflearn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import dill

MODEL_DIR = 'models'
NETWORK_DUMP = 'network.dill'
WEIGHTS_DUMP = 'weights.tflearn'

class Model(object):
    def __init__(self, network_func, params=[], name=None):
        tf.reset_default_graph()
        self.build_network = functools.partial(network_func, *params)
        self.model = self.build_network()
        if not name:
            name = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        self.name = name
        self.load_logs()

    def load_logs(self):
        try:
            pathlib.Path(self.model.trainer.tensorboard_dir).mkdir(parents=True, exist_ok=True)
            logs_dir = os.path.join(MODEL_DIR, self.name, 'logs')
            distutils.dir_util.copy_tree(logs_dir, self.model.trainer.tensorboard_dir)
        except:
            pass

    def train(self, X, Y, n_epoch=1000, batch_size=200, validation=None):
        for i in range(n_epoch):
            self.model.fit(
                {'input': X},
                {'target': Y},
                n_epoch=1,
                batch_size=batch_size,
                validation_set=validation,
                show_metric=True,
                run_id=self.name
            )
            self.save()

    def save(self, name=None):
        if not name:
            name = self.name
        loc = os.path.join(MODEL_DIR, name)
        pathlib.Path(loc).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(loc, NETWORK_DUMP), 'wb') as f:
            dill.dump(self.build_network, f)
        self.model.save(os.path.join(loc, WEIGHTS_DUMP))
        tflearn_logs = self.model.trainer.tensorboard_dir + self.name
        logs_dir = os.path.join(loc, 'logs', name)
        pathlib.Path(logs_dir).mkdir(parents=True, exist_ok=True)
        distutils.dir_util.copy_tree(tflearn_logs, logs_dir)

    @classmethod
    def load(cls, name):
        loc = os.path.join(MODEL_DIR, name)
        with open(os.path.join(loc, NETWORK_DUMP), 'rb') as f:
            network_func = dill.load(f)
        model = cls(network_func, name=name)
        model.model.load(os.path.join(loc, WEIGHTS_DUMP))
        return model
