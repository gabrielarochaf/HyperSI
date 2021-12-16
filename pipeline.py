import os
import numpy as np
import seaborn as sns
from utils import Utils
from sample import Sample
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from hsiroutine import HsiRoutine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix


from IPython.display import clear_output


class HsiPipeline:

    def __init__(self, data_folder: str, samples: dict):
        self.folder = data_folder
        self.samples = samples
        self.routine = HsiRoutine()

    def _remove_bg_sample(self, sample: Sample,
                          dark_prefix='DARKREF_',
                          white_prefix='WHITEREF_', ):
        pass

    def _signal_filter(self, sample: Sample):

        matrix = self.routine.hsi2matrix(sample.normalized)
        matrix = self.routine.normalize_mean(matrix)
        matrix = self.routine.sgolay(matrix=matrix, order=2, window=21, derivative=1, mode='constant')

        return self.routine.snv(matrix=matrix)

    def visualize_images(self):

        for idx, sample in enumerate(list(self.samples.keys())):
            bacteria = Utils.load_bacteria(path=self.folder, name=sample)

            clear_output()
            image = bacteria.normalized[50, :, :]
            out_i = self.routine.getCluster(image, bacteria.sample_cluster, 0, (1, 1, 1))
            plt.imshow(self.routine.rgbscale(out_i))
            plt.show()

    def process_images(self):

        for idx, sample in enumerate(list(self.samples.keys())):
            print(sample, idx)

            bacteria = Sample(self.folder, sample)
            darkref = Sample(self.folder, sample, sample_prefix='DARKREF_')
            whiteref = Sample(self.folder, sample, sample_prefix='WHITEREF_')

            bacteria.normalized = self.routine.raw2mat(image=bacteria, dark=darkref, white=whiteref, inplace=False)

            # Pre process block
            matrix = self._signal_filter(sample=bacteria)

            # Plot mean spectre
            # plot = self.routine.plot_mean_spectre(samples=[matrix])

            rows, cols = bacteria.normalized.shape[1:]
            cube = self.routine.matrix2hsi(matrix, rows, cols)

            idx = self.routine.removeBg(cube, 2) + 1

            image = cube[50, :, :]
            out_i = self.routine.getCluster(image, idx, 1, (1, 0, 0))
            out_i2 = self.routine.getCluster(image, idx, 2, (0, 1, 0))

            plt.imshow(self.routine.rgbscale(out_i))
            plt.show()
            plt.imshow(self.routine.rgbscale(out_i2))
            plt.show()

            cluster = input('red or green?')

            clear_output()
            ind, rm = self.routine.sum_idx_array(self.routine.realIdx(idx, int(cluster)))
            bacteria.sample_cluster = self.routine.rev_idx_array(ind, rm)

            out_i = self.routine.getCluster(image, bacteria.sample_cluster, 1, (1, 0, 0))
            plt.imshow(self.routine.rgbscale(out_i))
            plt.show()

            bacteria.image = None
            bacteria.save()

        clear_output()

    def get_Xy(self, case: int, spectral_range=(1, 241)):
        y_test = np.array([])
        y_train = np.array([])

        X_test = np.array([]).reshape(0, 240)
        X_train = np.array([]).reshape(0, 240)

        target_names = []

        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]

            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))
            idx_train, idx_test = train_test_split(ind, test_size=0.5, shuffle=False)
            X_test = np.concatenate((X_test, matrix[idx_test]))
            X_train = np.concatenate((X_train, matrix[idx_train]))

            y = np.ones(idx_test.shape) * self.samples[sample][case]
            y_test = np.concatenate((y_test, y))

            y = np.ones(idx_train.shape) * self.samples[sample][case]
            y_train = np.concatenate((y_train, y))

            if case != 0:
                target_names.append(Utils.get_name(self.samples,
                                                   self.samples[sample][case], case))
            else:
                target_names.append(sample)

        X_train, y_train = shuffle(X_train, y_train)

        return X_train, X_test, y_train, y_test, target_names

    @staticmethod
    def train_models(x_train: np.ndarray, x_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     models: list, target_names: list, models_file: str, work_dir='outputs'):

        for model in models:
            print(model.__class__.__name__)

            classifier = model.fit(x_train, y_train)
            predictions = classifier.predict(x_test)

            print(classification_report(y_test, predictions, target_names=list(target_names)))

            _, ax = plt.subplots(figsize=(13, 10))
            sns.set(font_scale=1.5)
            cf_matrix = confusion_matrix(y_test, predictions)
            sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,
                        xticklabels=target_names, yticklabels=target_names,
                        fmt='.2%', cmap='Blues', ax=ax, annot_kws={"size": 16})

        os.makedirs(os.path.join(os.getcwd(), work_dir), exist_ok=True)
        dump(models, os.path.join(os.getcwd(), work_dir, models_file))

