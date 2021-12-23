import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

from sample import Sample


class HsiRoutine:

    def plot_mean_spectre(self, samples: list, **kwargs):
        """
            Plot the mean spectre, calculated from the 2D matrix, default axis is the rows
            samples: list of 2D matrix (x*y, z)
        """
        plot = []
        for matrix in samples:
            plot.append(plt.plot(np.arange(matrix.shape[1]),
                                 self.mean_from_2d(matrix=matrix), **kwargs)[0])

        return plot

    def raw2mat(self, image: Sample, white: Sample, dark: Sample,
                inplace=True):
        """
            Normalize the sample using the Dark (0% Reflectance) and
            White (100% Reflectance) references, using the equation:
            -log10((S - D)/(W - D))
        """

        def extract_lines(matrix, lines):
            rows = matrix.shape[1]
            return matrix[:, np.arange(0, rows, np.ceil(rows / lines)).astype(np.int), :]

        def replace_median(matrix):
            [_, rows, cols] = matrix.shape
            for z, x, y in zip(*np.where(matrix == 0)):
                if 0 < x < rows and 0 < y < cols:
                    window = matrix[z, x - 1:x + 2, y - 1:y + 2]

                    if len(np.where(window == 0)[0]) == 1:
                        matrix[z, x, y] = np.median(window[(window != 0)])

            return matrix

        extracted_dark = extract_lines(dark.sample, 25)
        extracted_white = extract_lines(white.sample, 25)

        raw_dark = self.mean_from_3d(matrix=extracted_dark, ndims=3, axis=1)
        raw_white = self.mean_from_3d(matrix=extracted_white, ndims=3, axis=1)
        raw_image = image.sample

        with np.errstate(divide='ignore', invalid='ignore'):
            pabs = np.nan_to_num(((raw_image - raw_dark) / (raw_white - raw_dark)), nan=0.0)

        normalized = replace_median(-np.log10((pabs * (pabs > 0)), where=(pabs > 0)))

        if inplace:
            image.normalized = normalized
            return

        return normalized

    @staticmethod
    def hsi2matrix(matrix: np.ndarray):
        """
            Rearrange the 3D matrix that so that each pixel became a
            row in the 2D returned matrix
        """
        return matrix.T.reshape((matrix.shape[1] * matrix.shape[2], matrix.shape[0]), order='F')

    @staticmethod
    def matrix2hsi(matrix: np.ndarray, rows: int, cols: int):
        """
            Rearrange the 2D matrix into a 3D matrix
            final shape (-1, rows, cols)
        """
        return matrix.T.reshape(-1, rows, cols)

    @staticmethod
    def mean_from_2d(matrix: np.ndarray, axis=0):
        """
            Return the mean spectre of the 2D matrix
            matrix: hypercube (x*y, z)
        """
        return np.mean(matrix, axis=axis)

    @staticmethod
    def mean_from_3d(matrix: np.ndarray, ndims=2, axis=1):
        """
            Return the mean spectre of the 3D sample
            matrix: hypercube (x, y, z)
        """
        mean = np.mean(matrix, axis=axis).astype(np.float64)
        if ndims == 3:
            return mean.reshape((mean.shape[0], 1, mean.shape[1]))

        return mean

    @staticmethod
    def snv(matrix: np.ndarray):
        """
            Standard Normal Variate
        """

        out = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            out[i, :] = (matrix[i, :] - np.mean(matrix[i, :])) / np.std(matrix[i, :])

        return out

    @staticmethod
    def normalize_mean(matrix: np.ndarray):
        """
            Center data in 0 with the mean
        """

        out = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            out[i, :] = (matrix[i, :] - np.mean(matrix[i, :]))

        return out

    @staticmethod
    def savitzky_golay(y, window_size, order, deriv=0, rate=1):
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2

        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)

        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

        y = np.concatenate((firstvals, y, lastvals))

        return np.convolve(m[::-1], y, mode='valid')

    @staticmethod
    def sgolay(matrix: np.ndarray, order=2, window=21, derivative=1, mode='wrap'):
        """
            Savitzky-Golay filter
        """
        return savgol_filter(matrix, window, order, deriv=derivative, mode=mode)

    def removeBg(self, matrix, pcs):
        """
            matrix 3d to remove the bg
            pcs: number of pcs to kmeans
        """
        scores = PCA(n_components=pcs).fit_transform(self.hsi2matrix(matrix))
        return KMeans(n_clusters=2, init='k-means++', n_init=5, max_iter=300).fit(scores).labels_

    @staticmethod
    def rgbscale(image):
        return (image * 255).astype(np.uint8)

    @staticmethod
    def realIdx(idx, c):
        out = np.arange(idx.shape[0])
        for idx, (rid, vec) in enumerate(zip(out, idx)):
            if vec != c:
                out[idx] = -1
        out[out == -1] = 0

        return out

    @staticmethod
    def sum_idx_array(idx):
        ind_r = []
        for i, (j, ind) in enumerate(zip(idx, np.arange(idx.shape[0]))):
            if j != ind:
                ind_r.append(i)

        return np.delete(idx, ind_r), np.array(ind_r)

    @staticmethod
    def rev_idx_array(idx, rmv, shape=None, tfill=None):
        """
            Create an array of idx according with
            idx and rmv, arrays of indexes
        """

        if shape is None:
            out = np.zeros(idx.shape[0] + rmv.shape[0])
        else:
            out = np.zeros(shape)

        out[rmv] = 0

        if tfill is not None:
            for i, row in enumerate(idx):
                out[row] = tfill[i]
        else:
            out[idx] = 1

        return out.astype(int)

    def getCluster(self, image, idx, c, rgb):
        """
            show the idx in the image
        """

        ind = self.realIdx(idx, c)
        out_i = np.concatenate((ind, ind, ind), axis=0).reshape((3, *(image.shape[:2])))

        if len(image.shape) == 2:
            image = MinMaxScaler(feature_range=(0, 1)).fit_transform(image)
            image = np.stack((image, image, image), axis=2)

        image[out_i[0] != 0, 0] = rgb[0]
        image[out_i[1] != 0, 1] = rgb[1]
        image[out_i[2] != 0, 2] = rgb[2]

        return image


if __name__ == '__main__':
    routine = HsiRoutine()
    print('so far so good')
