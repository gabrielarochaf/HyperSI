import os
import json
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.express as px
import matplotlib.patches as mpatches
from utils import Utils
from sample import Sample
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from hsiroutine import HsiRoutine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

from IPython.display import clear_output
from sklearn.metrics import confusion_matrix


colors = {
    '0': '#2001c4',
    '1': '#707e51',
    '2': '#df0100',
    '3': '#a96a77',
    '4': '#288bb0',
    '5': '#643a89',
    '6': '#715657',
    '7': '#533e55',
    '8': '#68be9e',
    '9': '#bafeac',
    '10': '#bf4f37',
    '11': '#0fd1c6',
    '12': '#22cbe3',
    '13': '#961e53',
    '14': '#1ad397',
    '15': '#811771',
    '16': '#404686',
    '17': '#4c42e2',
    '18': '#fbf899',
    '19': '#bdd387',
    '20': '#7774bd',
    '21': '#1b0f3f',
    '22': '#32e726',
    '23': '#b25e1c',
    '24': '#87508b',
    '25': '#fa38ff',
    '26': '#c0e33c',
    '27': '#6b8a93',
    '28': '#cec15f',
    '29': '#7cbca0',
    '30': '#692225',
    '31': '#4e7aee',
    '32': '#89f41d',
    '33': '#2a5ba7',
    '34': '#5cb70b',
    '35': '#c8f9a1',
    '36': '#cbc184',
    '37': '#253b85',
    '38': '#919b65',
    '39': '#76929b',
    '40': '#7e6943',
    '41': '#7b1170',
    '42': '#2785ca',
    '43': '#a16180',
    '44': '#45abc2',
    '45': '#eec78b',
    '46': '#f8310a',
    '47': '#1b8991',
    '48': '#a5c7a5',
    '49': '#d67f3a',
    '50': '#6dca0d',
    '51': '#139386',
    '52': '#3cf0ff',
    '53': '#4f8013',
    '54': '#4c1134',
    '55': '#c28f0d',
    '56': '#2ddfdb',
    '57': '#eaadda',
    '58': '#dcd64b',
    '59': '#c0a95c',
    '60': '#b375f6',
    '61': '#73b7df',
    '62': '#2a5b84',
    '63': '#d34e2d',
}


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

    def results(self, models: list,
                work_dir: str,
                fig_dest_dir=None,
                spectral_range=(1, 241), ):

        with open(os.path.join(os.getcwd(), work_dir, 'config.json'), 'r') as f_cfg:
            cfg = json.load(f_cfg)
            f_cfg.close()

        for sample in list(self.samples.keys()):
            for idx, model in enumerate(models):
                print(model.__class__.__name__, sample)

                bacteria = Utils.load_bacteria(path=self.folder, name=sample)
                matrix = self._signal_filter(sample=bacteria)
                matrix = matrix[:, spectral_range[0]:spectral_range[1]]

                ind, rem = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

                result = model.predict(matrix[ind])
                full_array = self.routine.rev_idx_array(ind, rem, tfill=result)

                targets = []
                cl_legends = []
                image = self.routine.matrix2hsi(matrix, *bacteria.normalized.shape[1:])[50, :, :]
                image = self.routine.getCluster(image, bacteria.sample_cluster, 0, (255, 255, 255))

                for classe in model.classes_:
                    #             print(hex2rgb(colors[str(int(classe))]), colors[str(int(classe))])
                    image = self.routine.getCluster(image, full_array, classe, Utils.hex2rgb(colors[str(int(classe))]))
                    cl_legends.append(colors[str(int(classe))])
                    targets.append(Utils.get_name(cfg['samples_training'], classe, 0))

                cl_legends = [colors[str(int(classe))] for classe in model.classes_]
                patches = [mpatches.Patch(color=cl_legends[i], label=targets[i])
                           for i in range(len(targets))]

                fig, ax = plt.subplots(**{'figsize': (14, 8), 'dpi': 300})

                ax.axis('off')
                ax.imshow(np.uint8(image))
                legend = fig.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                os.makedirs('{}/sample/{}'.format(fig_dest_dir, model.__class__.__name__ + '_' + str(idx)),
                            exist_ok=True)
                fig.savefig('{}/sample/{}/{}.jpg'
                            .format(fig_dest_dir, model.__class__.__name__ + '_' + str(idx), sample),
                            bbox_extra_artists=[legend], bbox_inches='tight')

                plt.show()

                del ind, bacteria, matrix

    def get_Xy(self, case: int, spectral_range=(1, 241), test_size=0.5):
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

            if case != 0:
                target_names.append(Utils.get_name(self.samples,
                                                   self.samples[sample][case], case))
            else:
                target_names.append(sample)

            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

            if test_size < 1.0:
                idx_train, idx_test = train_test_split(ind, test_size=test_size, shuffle=False)
                X_test = np.concatenate((X_test, matrix[idx_test]))
                X_train = np.concatenate((X_train, matrix[idx_train]))

                y = np.ones(idx_test.shape) * self.samples[sample][case]
                y_test = np.concatenate((y_test, y))

                y = np.ones(idx_train.shape) * self.samples[sample][case]
                y_train = np.concatenate((y_train, y))

            if test_size == 1.0:
                X_test = np.concatenate((X_test, matrix[ind]))
                y_test = np.concatenate((y_test, np.ones(ind.shape) * self.samples[sample][case]))

        if test_size < 1.0:
            X_train, y_train = shuffle(X_train, y_train)
            return X_train, X_test, y_train, y_test, target_names

        X_test, y_test = shuffle(X_test, y_test)

        return X_test, y_test, target_names

    def get_Xy_gram(self, case: int, spectral_range=(1, 241), test_size=0.5):
        y_test = np.array([])
        y_train = np.array([])

        X_test = np.array([]).reshape(0, 240)
        X_train = np.array([]).reshape(0, 240)

        target_names = []
        samples_dict_classes = ['Gram Positivo',
         'Gram Positivo', 
         'Gram Positivo',
         'Gram Positivo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
        ]
        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]

            if case != 0:
                target_names.append(Utils.get_name_gram(self.samples, case))
            else:
                target_names.append(Utils.get_name_gram(samples_dict_classes, idx))

            ind, _ = self.routine.sum_idx_array(self.routine.realIdx(bacteria.sample_cluster, 1))

            if test_size < 1.0:
                idx_train, idx_test = train_test_split(ind, test_size=test_size, shuffle=False)
                X_test = np.concatenate((X_test, matrix[idx_test]))
                X_train = np.concatenate((X_train, matrix[idx_train]))

                y = np.ones(idx_test.shape) * self.samples[sample][case]
                y_test = np.concatenate((y_test, y))

                y = np.ones(idx_train.shape) * self.samples[sample][case]
                y_train = np.concatenate((y_train, y))

            if test_size == 1.0:
                X_test = np.concatenate((X_test, matrix[ind]))
                y_test = np.concatenate((y_test, np.ones(ind.shape) * self.samples[sample][case]))

        if test_size < 1.0:
            X_train, y_train = shuffle(X_train, y_train)
            return X_train, X_test, y_train, y_test, target_names

        X_test, y_test = shuffle(X_test, y_test)

        return X_test, y_test, target_names


    @staticmethod
    def train_models(x_train: np.ndarray, x_test: np.ndarray,
                     y_train: np.ndarray, y_test: np.ndarray,
                     models: list, target_names: list, models_file: str,
                     samples_dict: dict, work_dir='outputs'):

        os.makedirs(os.path.join(os.getcwd(), work_dir), exist_ok=True)

        config = dict({'samples_training': samples_dict})
        for key in config.keys():
            for sample in config[key].keys():
                config[key][sample] = [int(case) for case in config[key][sample]]

        with open(os.path.join(os.getcwd(), work_dir, 'config.json'), 'w') as f_cfg:
            json.dump(config, f_cfg)
            f_cfg.close()

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

        dump(models, os.path.join(os.getcwd(), work_dir, models_file))


    def get_spectrals(self, case: int, spectral_range=(1, 241)):

        target_names = []

        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]
            ind, _ = HsiRoutine.sum_idx_array(HsiRoutine.realIdx(bacteria.sample_cluster, 1))
            self.samples[sample] = matrix[ind]
            
            if case != 0:
                target_names.append(Utils.get_name(self.samples,
                                                   self.samples[sample][case], case))
            else:
                target_names.append(sample)


        mean_matrix= HsiRoutine.mean_from_2d(matrix=self.samples[list(self.samples.keys())[1]], axis=0).reshape(1,-1)
        for key in list(self.samples.keys()):
            # if self.samples[key] == 9:
            #     continue
            matrix = self.samples[key]
            target_names.append(key)
            mean_matrix = np.concatenate((mean_matrix, HsiRoutine.mean_from_2d(matrix=matrix, axis=0).reshape(1,-1)))
        mean_matrix = mean_matrix[:9]
        plots = []
        fontproperties = {'family': 'sans-serif', 'weight': 'bold', 'size': 10}

        fig, axes = plt.subplots(figsize=(12,6), dpi=1000)

        plt.rcParams.update({'font.size': 10})

        for i, target in zip(range(mean_matrix.shape[0]), target_names):
            plotargs = {'label': target, 'linewidth': 2}
            axes.set_xlabel("Comprimento de onda")
            axes.set_ylabel("Pseudo Absorvancia")
            axes.set_xlabel(axes.get_xlabel(), fontproperties)
            axes.set_ylabel(axes.get_ylabel(), fontproperties)
            plots.append(axes.plot(np.arange(mean_matrix.shape[1]), mean_matrix[i,:], **plotargs))

        labels = [plot[0].get_label() for plot in plots]
        legend = fig.legend(
            plots,
            labels,
            # loc='down right',
            # bbox_to_anchor=(0.90, 2, 0.32, -.102),
            mode="expand",
            ncol="2",
            bbox_transform=fig.transFigure,
        )
        # plt.legend()
        fig.savefig('mean_espectre.jpeg', bbox_extra_artists=[], bbox_inches="tight")

        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=[legend_bbox.width+3, legend_bbox.height])

        legend_squared = legend_ax.legend(
            *axes.get_legend_handles_labels(),
            bbox_to_anchor=(0,0,1,1),
            bbox_transform=legend_fig.transFigure,
            frameon=False,
            fontsize=12,
            fancybox=None,
            shadow=False,
            ncol=1,
            mode="expand"
        )
        legend_ax.axis("off")

        legend_figpath = 'legend_mean_spectres.jpeg'
        print({'saving to: {legend_figpath} ' })
        legend_fig.savefig(
            legend_figpath,
            bbox_inches="tight",
            bbox_extra_artists=[legend_squared],
        )


    def get_pcas(self, case: int, spectral_range=(1, 241)):

        target_names = []

        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]
            ind, _ = HsiRoutine.sum_idx_array(HsiRoutine.realIdx(bacteria.sample_cluster, 1))
            self.samples[sample] = matrix[ind]
            
            if case != 0:
                target_names.append(Utils.get_name(self.samples,
                                                   self.samples[sample][case], case))
            else:
                target_names.append(sample)


        mean_matrix= HsiRoutine.mean_from_2d(matrix=self.samples[list(self.samples.keys())[1]], axis=0).reshape(1,-1)
        for key in list(self.samples.keys()):
            # if self.samples[key] == 9:
            #     continue
            matrix = self.samples[key]
            target_names.append(key)
            mean_matrix = np.concatenate((mean_matrix, HsiRoutine.mean_from_2d(matrix=matrix, axis=0).reshape(1,-1)))
        mean_matrix = mean_matrix[:9]
        
        df_temp=pd.DataFrame()
        df_temp['Amostras']=target_names

        pca  = PCA(n_components=2)
        components = pca.fit_transform(mean_matrix)
     
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(2),
        # color=1
        color=pd.core.series.Series(target_names[:9])  
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

    def get_spectrals_gram(self, case: int, spectral_range=(1, 241)):

        target_names = []
        samples_dict_classes = ['Gram Positivo',
         'Gram Positivo', 
         'Gram Positivo',
         'Gram Positivo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
        ]
        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]
            ind, _ = HsiRoutine.sum_idx_array(HsiRoutine.realIdx(bacteria.sample_cluster, 1))
            self.samples[sample] = matrix[ind]
            
            if case != 0:
                target_names.append(Utils.get_name_gram(self.samples, case))
            else:
                target_names.append(Utils.get_name_gram(samples_dict_classes, idx))


        mean_matrix= HsiRoutine.mean_from_2d(matrix=self.samples[list(self.samples.keys())[1]], axis=0).reshape(1,-1)
        for key in list(self.samples.keys()):
            # if self.samples[key] == 9:
            #     continue
            matrix = self.samples[key]
            target_names.append(key)
            mean_matrix = np.concatenate((mean_matrix, HsiRoutine.mean_from_2d(matrix=matrix, axis=0).reshape(1,-1)))
        mean_matrix = mean_matrix[:9]
        plots = []
        fontproperties = {'family': 'sans-serif', 'weight': 'bold', 'size': 10}

        fig, axes = plt.subplots(figsize=(12,6), dpi=1000)

        plt.rcParams.update({'font.size': 10})

        for i, target in zip(range(mean_matrix.shape[0]), target_names):
            plotargs = {'label': target, 'linewidth': 2}
            axes.set_xlabel("Comprimento de onda")
            axes.set_ylabel("Pseudo Absorvancia")
            axes.set_xlabel(axes.get_xlabel(), fontproperties)
            axes.set_ylabel(axes.get_ylabel(), fontproperties)
            plots.append(axes.plot(np.arange(mean_matrix.shape[1]), mean_matrix[i,:], **plotargs))

        labels = [plot[0].get_label() for plot in plots]
        legend = fig.legend(
            plots,
            labels,
            # loc='down right',
            # bbox_to_anchor=(0.90, 2, 0.32, -.102),
            mode="expand",
            ncol="2",
            bbox_transform=fig.transFigure,
        )
        # plt.legend()
        fig.savefig('mean_espectre.jpeg', bbox_extra_artists=[], bbox_inches="tight")

        fig.canvas.draw()
        legend_bbox = legend.get_tightbbox(fig.canvas.get_renderer())
        legend_bbox = legend_bbox.transformed(fig.dpi_scale_trans.inverted())
        legend_fig, legend_ax = plt.subplots(figsize=[legend_bbox.width+3, legend_bbox.height])

        legend_squared = legend_ax.legend(
            *axes.get_legend_handles_labels(),
            bbox_to_anchor=(0,0,1,1),
            bbox_transform=legend_fig.transFigure,
            frameon=False,
            fontsize=12,
            fancybox=None,
            shadow=False,
            ncol=1,
            mode="expand"
        )
        legend_ax.axis("off")

        legend_figpath = 'legend_mean_spectres.jpeg'
        print({'saving to: {legend_figpath} ' })
        legend_fig.savefig(
            legend_figpath,
            bbox_inches="tight",
            bbox_extra_artists=[legend_squared],
        )
    

    def get_pcas_gram(self, case: int, spectral_range=(1, 241)):

        target_names = []
        samples_dict_classes = ['Gram Positivo',
         'Gram Positivo', 
         'Gram Positivo',
         'Gram Positivo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
         'Gram Negativo',
        ]

        for idx, sample in enumerate(list(self.samples.keys())):
            if self.samples[sample][case] == -1:
                continue

            bacteria = Utils.load_bacteria(path=self.folder, name=sample)
            matrix = self._signal_filter(sample=bacteria)

            matrix = matrix[:, spectral_range[0]:spectral_range[1]]
            ind, _ = HsiRoutine.sum_idx_array(HsiRoutine.realIdx(bacteria.sample_cluster, 1))
            self.samples[sample] = matrix[ind]
            
            if case != 0:
                target_names.append(Utils.get_name_gram(self.samples, case))
            else:
                target_names.append(Utils.get_name_gram(samples_dict_classes, idx))


        mean_matrix= HsiRoutine.mean_from_2d(matrix=self.samples[list(self.samples.keys())[1]], axis=0).reshape(1,-1)
        for key in list(self.samples.keys()):
            # if self.samples[key] == 9:
            #     continue
            matrix = self.samples[key]
            target_names.append(key)
            mean_matrix = np.concatenate((mean_matrix, HsiRoutine.mean_from_2d(matrix=matrix, axis=0).reshape(1,-1)))
        mean_matrix = mean_matrix[:9]
        
        df_temp=pd.DataFrame()
        df_temp['Amostras']=target_names

        pca  = PCA(n_components=2)
        components = pca.fit_transform(mean_matrix)
     
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(2),
        # color=1
        color=pd.core.series.Series(target_names[:9])  
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()