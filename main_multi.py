'''Image similarity using deep features.

Recommendation: the threshold of the `DeepModel.cosine_distance` can be set as the following values.
    0.74 => for greater matches amount
    0.75 => for balance, default
    0.76 => for better accuracy
'''

from io import BytesIO
from multiprocessing import Pool

import datetime
import numpy as np
import requests
import h5py

from model_util import DeepModel, DataSequence


class ImageSimilarity():
    '''Image similarity.'''
    def __init__(self):
        self._batch_size = 64
        self._num_processes = 4
        self._model = None
        self._title = []

    @property
    def batch_size(self):
        '''Batch size of model prediction.'''
        return self._batch_size

    @property
    def num_processes(self):
        '''Number of processes using `Multiprocessing.Pool`.'''
        return self._num_processes

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @num_processes.setter
    def num_processes(self, num_processes):
        self._num_processes = num_processes

    def _data_generation(self, args):
        '''Generate input batches for predict generator.

        Args:
            args: parameters that pass to `sub_process`.
                - path: path of the image, online url by default.
                - fields: all other fields.

        Returns:
            batch_x: a batch of predict samples.
            batch_fields: a batch of fields that matches the samples.
        '''
        # Multiprocessing
        pool = Pool(self._num_processes)
        res = pool.map(self._sub_process, args)
        pool.close()
        pool.join()

        batch_x, batch_fields = [], []
        for x, fields in res:
            if x is not None:
                batch_x.append(x)
                batch_fields.append(fields)

        return batch_x, batch_fields

    def _predict_generator(self, paras):
        '''Build a predict generator.

        Args:
            paras: input parameters of all samples.
                - path: path of the image, online url by default.
                - fields: all other fields.

        Returns:
            The predict generator.
        '''
        return DataSequence(paras, self._data_generation, batch_size=self._batch_size)

    @staticmethod
    def _sub_process(para):
        '''A sub-process function of `multiprocessing`.

        Download image from url and process it into a numpy array.

        Args:
            para: input parameters of one image.
                - path: path of the image, online url by default.
                - fields: all other fields.

        Returns:
            feature: feature array of one image.
            fields: all other fields  of one image that passed from `para`.

        Note: If error happens, `None` will be returned.
        '''
        path, fields = para['path'], para['fields']
        try:
            res = requests.get(path)
            feature = DeepModel.preprocess_image(BytesIO(res.content))
            return feature, fields

        except Exception as e:
            print('Error downloading %s: %s' % (fields[0], e))

        return None, None

    @staticmethod
    def load_data_csv(fname, delimiter=None, include_header=True, cols=None):
        '''Load `.csv` file. Mostly it should be a file that list all fields to match.

        Args:
            fname: name or path to the file.
            delimiter: delimiter to split the content.
            include_header: whether the source file include header or not.
            cols: a list of columns to read. Pass `None` to read all columns.

        Returns:
            A list of data.
        '''
        assert delimiter is not None, 'Delimiter is required.'

        if include_header:
            usecols = None
            skip_header = 1
            if cols:
                with open(fname, 'r', encoding='utf-8') as f:
                    csv_head = f.readline().strip().split(delimiter)

                usecols = [csv_head.index(col) for col in cols]

        else:
            usecols = None
            skip_header = 0

        data = np.genfromtxt(
            fname,
            dtype=str,
            comments=None,
            delimiter=delimiter,
            encoding='utf-8',
            invalid_raise=False,
            usecols=usecols,
            skip_header=skip_header
        )

        return data if len(data.shape) > 1 else data.reshape(1, -1)

    @staticmethod
    def load_data_h5(fname):
        '''Load `.h5` file. Mostly it should be a file with features that extracted from the model.

        Args:
            fname: name or path to the file.

        Returns:
            A list of data.
        '''
        with h5py.File(fname, 'r') as h:
            data = np.array(h['data'])
        return data



    def save_data(self, fname, delimiter, cols=None):
        '''Load images from `.csv`, extract features and fields, save as `.h5` and `.csv` files.

        Args:
            fname: name or path to the file.
            delimiter: delimiter to split the content.
            cols: a list of columns to read. Pass `None` to read all columns.

        Returns:
            None. `.h5` and `.csv` files will be saved instead.
        '''
        # Load model
        if self._model is None:
            self._model = DeepModel()

        print('%s: download starts.' % fname)
        start = datetime.datetime.now()

        lines = self.load_data_csv(fname, delimiter=delimiter, include_header=True, cols=cols)

        args = [{'path': line[-1], 'fields': line} for line in lines]

        # Prediction
        generator = self._predict_generator(args)
        features = self._model.extract_feature(generator)

        # Save files
        title = fname.split('/')[-1].split('.')[0]

        if len(self._title) == 2:
            self._title = []
        self._title.append(title)

        fname_feature = '_' + title + '_feature.h5'
        with h5py.File(fname_feature, mode='w') as h:
            h.create_dataset('data', data=features)
        print('%s: feature saved to `%s`.' % (fname, fname_feature))

        fname_fields = '_' + title + '_fields.csv'
        np.savetxt(fname_fields, generator.list_of_label_fields, delimiter='\t', fmt='%s', encoding='utf-8')
        print('%s: fields saved to `%s`.' % (fname, fname_fields))

        print('%s: download succeeded.' % fname)
        print('Time consumed:', datetime.datetime.now()-start)
        print()

    def iteration(self, save_header, thresh=0.75, title1=None, title2=None):
        '''Calculate the cosine distance of two inputs, save the matched fields to `.csv` file.

        Args:
            save_header: header of the result `.csv` file.
            thresh: threshold of the similarity.
            title1, title2: Optional. If `save_data()` is not invoked, titles of two inputs should be passed.

        Returns:
            A matrix of element-wise cosine distance.

        Note:
            1. The threshold can be set as the following values.
                0.74 = greater matches amount
                0.75 = balance, default
                0.76 = better accuracy

            2. If the generated files are exist, set `title1` or `title2` as same as the title of their source files.
                For example, pass `test.csv` to `save_data()` will generate `_test_feature.h5` and `_test_fields.csv` files,
                so set `title1` or `title2` to `test`, and `save_data()` will not be required to invoke.
        '''
        print('Iteration starts.')
        start = datetime.datetime.now()

        if title1 and title2:
            self._title = [title1, title2]

        assert len(self._title) == 2, 'Two inputs are required.'

        feature1 = self.load_data_h5('_' + self._title[0] + '_feature.h5')
        feature2 = self.load_data_h5('_' + self._title[1] + '_feature.h5')

        fields1 = self.load_data_csv('_' + self._title[0] + '_fields.csv', delimiter='\t', include_header=False)
        fields2 = self.load_data_csv('_' + self._title[1] + '_fields.csv', delimiter='\t', include_header=False)

        result = [save_header]

        distances = DeepModel.cosine_distance(feature1, feature2)
        indexes = np.argmax(distances, axis=1)

        for x, y in enumerate(indexes):
            if distances[x][y] >= thresh:
                result.append(np.concatenate((fields1[x], fields2[y]), axis=0))

        if len(result) > 0:
            np.savetxt('result_similarity.csv', result, fmt='%s', delimiter='\t', encoding='utf-8')

        print('Iteration finished: results saved to `result_similarity.csv`.')
        print('Time consumed:', datetime.datetime.now()-start)
        print()

        return distances


if __name__ == '__main__':
    similarity = ImageSimilarity()

    '''Setup'''
    similarity.batch_size = 16
    similarity.num_processes = 2

    '''Save features and fields'''
    similarity.save_data('./demo/test1.csv', ',')
    similarity.save_data('./demo/test2.csv', ',', cols=['id', 'url'])

    '''Calculate similarities'''
    result = similarity.iteration(['test1_id', 'test1_url', 'test2_id', 'test2_url'], thresh=0.745)
    print('Row for source file 1, and column for source file 2.')
    print(result)
