import os
import pickle
import numpy as np


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def load_samples(folder):
        return [a for a in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, a))]

    @staticmethod
    def get_name(samples_dict, group, case, sz=2):
        labels = [key for key in samples_dict.keys() if samples_dict[key][case] == group]

        return ''.join(labels[0].split('_')[:sz])

    @staticmethod
    def load_bacteria(path: str, name: str, folder='capture'):
        sample_path = os.path.join(path, name)
        with open(os.path.join(sample_path, folder, name + '.pkl'), 'rb') as file:
            out = pickle.load(file)

        return out

    @staticmethod
    def hex2rgb(value):
        return tuple(int(value.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    @staticmethod
    def no_rep(names: list):
        return np.array(list(set(names)))

    @staticmethod
    def get_dict(samples: list):
        samples_dict = {}
        for sample, idx in zip(samples, np.arange(len(samples))):
            samples_dict[sample] = [idx + 1]

        return samples_dict

