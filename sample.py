import os
import pickle
import spectral as sp

sp.settings.envi_support_nonlowercase_params = True


class Sample:
    def __init__(self, path, sample_name, inter='capture',
                 sample_prefix=None,
                 to_numpy_format=True):

        self.path = os.path.join(path, sample_name, (inter if inter else ''))
        self.sample_name = (sample_prefix if sample_prefix else '') + sample_name

        self.image = None
        self.sample = None
        self.processed = None
        self.normalized = None
        self.sample_cluster = None

        self._read_image(to_numpy_format)

    def _read_image(self, to_numpy_format):
        """
            image stores a class from Spectral package
            sample stores a numpy array with 3 dimensions (wavelength x rows x cols)
        """
        try:
            self.image = sp.open_image(os.path.join(self.path, self.sample_name + '.hdr'))
            self.sample = self.image.load()

            if to_numpy_format:
                self.sample = self.sample.transpose(2, 0, 1)

        except Exception as e:
            print(e)

    def save(self):
        """
            save class at self.name.txt
        """
        sample_path = os.path.join(self.path, self.sample_name)
        sample_file = sample_path + '.pkl'

        with open(sample_file, 'wb') as destination_dir:
            pickle.dump(self, destination_dir, -1)

    @property
    def image(self):
        return self.__image

    @image.setter
    def image(self, var):
        self.__image = var

    @property
    def sample(self):
        return self.__sample

    @sample.setter
    def sample(self, var):
        self.__sample = var

    @property
    def normalized(self):
        return self.__normalized

    @normalized.setter
    def normalized(self, var):
        self.__normalized = var

    @property
    def processed(self):
        return self.__processed

    @processed.setter
    def processed(self, var):
        self.__processed = var

    @property
    def sample_cluster(self):
        return self.__sample_cluster

    @sample_cluster.setter
    def sample_cluster(self, var):
        self.__sample_cluster = var


if __name__ == '__main__':
    sample = Sample('dir',
                    'Enterobacteaerogenes_13048_Plastico_A_Contaminado_180926-102646')

    print(sample.image.shape)
    print('done')
