import gc
import sys
import pickle
import logging
from time import time
import numpy as np
from glob import glob
import pandas as pd
from pprint import pprint
import pyarrow.parquet as pq
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
logger = logging.getLogger('dnnlab')


class DataLoader(object):
    """Data loader to load data in a memory-friendly way, which allows to
    use unlimited data to train neural network models with appropriate
    (default) arguments.
    """

    def __init__(self, args, dataframe_process=None, sep_cols=None):
        self.args = args
        self.label = args.label
        self.no_cache = args.prod or args.no_cache
        self.load_cache = args.load_cache
        self.data_format = args.data_format
        self.data_config = args.data_config
        self.eval_trained = args.eval_trained
        self.resample = args.imb_learn if args.imb_learn[:2] in ['os', 'us'
                                                                 ] else None
        self.gen_load_threshold = args.gen_load_threshold
        self.dataframe_process = dataframe_process or (lambda _, x: x)
        self.trains = []
        self.tests = []
        self.predictions = []
        self.nb_train = 1e-15
        self.nb_test = 1e-15
        self.nb_prediction = 1e-15
        self.nb_positive_train = 0
        self.nb_positive_test = 0
        self.train_pos_frac = 0
        self.test_pos_frac = 0
        self.column_emb = args.column_emb
        self.online_learning = args.online_learning
        self.csv_sep = args.csv_sep
        self.sep_cols = sep_cols or (lambda ce, x: x)
        self.data_label = 'None'
        self._data_config()
        args.no_info or self._get_data_stats()
        self.load_data(_get_features=True)

    @staticmethod
    def _resample_data(resample, x, y):
        """Resample data to balance positive rate. 

        # Arguments
            resample: should be a decimal prefixed with a resample method 'os'
              (oversampling) or 'us' (undersampling) such as 'us.35' or
              'os.5', where the third to last characters will be converted to
              resample ratio (minority over majority).
            x: features.
            y: targets.

        # Returns
            Resampled x and y
        """

        if resample:
            if resample[:2] == 'os':
                rs = RandomOverSampler(float(resample[2:]))
            if resample[:2] == 'us':
                rs = RandomUnderSampler(float(resample[2:]))
            x, y = rs.fit_sample(x, y)
        return x, y

    def _ls_data_paths(self, ds):
        """Get list of data file paths. 

        # Arguments
            ds: directories where data files locate.

        # Returns
            List of file paths. 
        """

        lst = []
        for d in ds:
            if not d.endswith('/'):
                d += '/'

            if self.data_format == 'parquet':
                lst += glob('%s*.snappy.parquet' % d)

            if self.data_format == 'csv':
                lst += glob('%s*' % d)
        return lst

    def _data_config(self):
        """Parse the keyword argument '--data-config' to get data file paths
        for training, test and prediction.
        """

        d_cfg = self.data_config.split('-')

        if len(d_cfg) == 4:
            self.data_label = d_cfg[0].split('/')[-2] + self.data_config.split(
                '/')[-1]
            tr_ds = [
                '%s*%d/' % (d_cfg[0], d)
                for d in range(int(d_cfg[1]) - int(d_cfg[2]), int(d_cfg[1]))
            ]
            te_ds = [
                '%s*%d/' % (d_cfg[0], d)
                for d in range(int(d_cfg[1]), int(d_cfg[1]) + int(d_cfg[3]))
            ]

            print('\nTrain directories:')
            pprint(tr_ds or 'None')
            print('Test directories:')
            pprint(te_ds or 'None')
            print('Prediction data path:')
            pprint(self.args.predict_path or 'None')

            self.trains = self._ls_data_paths(tr_ds)
            self.tests = self._ls_data_paths(te_ds)
            self.predictions = [self.args.predict_path]

            if len(self.trains) == 0 and not self.eval_trained:
                raise ValueError('Invalid dataset config: ' + str(d_cfg))

        if self.args.prod:
            self.trains = self._ls_data_paths(d_cfg)
            logger.info('Found %d data files for training' % len(self.trains))

    def _check_label(self, df):
        assert df.columns[-1] == self.label, (
            'Last column should be "' + self.label + '" but found "' +
            df.columns[-1] + '", please move '
            'your target column to the last column in "dataframe_process" '
            'function defined by yourself.')

    def _read_all_data(self, ps):

        df = pd.DataFrame()
        for i, p in enumerate(ps):
            print('Processed %d/%d' % ((i + 1), len(ps)), end='\r')
            if self.data_format == 'parquet':
                df_tmp = pq.read_table(p).to_pandas()
            if self.data_format == 'csv':
                df_tmp = pd.read_csv(p)

            df = df.append(df_tmp, ignore_index=True)
        sys.stdout.write('\033[K')
        return df

    def _parquet_read_generator(self,
                                ps,
                                feed_mode,
                                batch_size,
                                resample=None,
                                pred=False):
        """Parquet read generator to generate data by small batch.

        # Arguments
            ps: paths of parquet meta files.
            feed_mode: if 'batch', read all files once, if 'generator',
              recurrently read all files.
            batch_size: number of rows to read each time.           
            resample: should be a decimal prefixed with a resample method 'os'
              (oversampling) or 'us' (undersampling) such as 'us.35' or
              'os.5', where the third to last characters will be converted to
              resample ratio (minority over majority).
            pred: prediction mode, where the label 'y' doesn't exist.

        # Returns
            List of parquet meta files. 
        """

        df = pd.DataFrame()
        mark = True
        while mark:
            logger.info("Generating data from parquet files")
            if not self.online_learning:
                np.random.shuffle(ps)
            for p in ps:
                tmp = pq.read_table(p).to_pandas()
                if pred:
                    tmp[self.label] = 0
                df = df.append(tmp, ignore_index=True)
                if len(df) >= self.gen_load_threshold or p == ps[-1]:
                    df = self.dataframe_process(self.args, df)
                    self._check_label(df)
                    for x_b, y_b in self._shuffle_resample_and_yield(
                            df, batch_size, resample):
                        x_b = self.sep_cols(self.column_emb, x_b)
                        yield x_b, y_b

                    df = pd.DataFrame()
                    gc.collect()
            if feed_mode == 'batch':
                mark = False
            logger.info("Loaded all data from parquet files")

    def _csv_read_generator(self,
                            ps,
                            feed_mode,
                            batch_size,
                            resample=None,
                            nb_b=128,
                            pred=False):
        """CSV read generator to generate data by small batch.

        # Arguments
            ps: paths of CSV files.
            feed_mode: if 'batch', read all files once, if 'generator',
              recurrently read all files.
            batch_size: number of rows to read each time.           
            resample: should be a decimal prefixed with a resample method 'os'
              (oversampling) or 'us' (undersampling) such as 'us.35' or
              'os.5', where the third to last characters will be converted to
              resample ratio (minority over majority).
            nb_b: number of batches to read as a chunk.
            pred: prediction mode, where the label 'y' doesn't exist.

        # Returns
            List of parquet meta files. 
        """

        mark = True
        while mark:
            for p in ps:
                for df in pd.read_csv(
                        p, chunksize=batch_size * nb_b, sep=self.csv_sep):
                    df = self.dataframe_process(self.args, df)
                    if pred:
                        df[self.label] = 0
                    self._check_label(df)
                    for x_b, y_b in self._shuffle_resample_and_yield(
                            df, batch_size, resample):
                        x_b = self.sep_cols(self.column_emb, x_b)
                        yield x_b, y_b

                    gc.collect()
            if feed_mode == 'batch':
                mark = False

    def _shuffle_resample_and_yield(self, df, batch_size, resample):
        nb_batch = np.ceil(len(df) / batch_size)
        data = df.values
        # if using online_learning, should keep the order of original data
        if not self.online_learning:
            np.random.shuffle(data)
        x, y = data[:, :-1], data[:, -1]
        x, y = DataLoader._resample_data(resample, x, y)

        for x_b, y_b in zip(
                np.array_split(x, nb_batch), np.array_split(y, nb_batch)):
            yield x_b, y_b

    def _cache_data(self, dn, data=None):
        cache_path = '.cache/%s_cache_narray.bin' % dn
        if data is None:
            print('Loading cached %s data' % dn)
            return pickle.load(open(cache_path, 'rb'))
        else:
            if not self.no_cache:
                print('Dumping %s data' % dn, end='\r')
                pickle.dump(data, open(cache_path, 'wb'))
                print('Dumped %s data' % dn)
            return data

    def _count(self, paths, data_name):
        print('Getting %s data statistics...' % data_name, end='\r')
        nb = 0
        nb_positive = 0
        if self.data_format == 'parquet':
            for p in paths:
                p_pd = pq.read_table(p).to_pandas()
                nb += len(p_pd.index)
                nb_positive += p_pd[self.label].sum()

        if self.data_format == 'csv':
            for p in paths:
                for tmp_df in pd.read_csv(
                        p, chunksize=1 << 20, sep=self.csv_sep):
                    nb += len(tmp_df.index)
                    nb_positive += tmp_df[self.label].sum()

        if data_name == 'train':
            self.nb_train += nb
            self.nb_positive_train += nb_positive

        if data_name == 'test':
            self.nb_test = nb
            self.nb_positive_test += nb_positive

    def _get_data_stats(self):
        """Get positive rate for training data and test data."""

        from queue import Queue
        from threading import Thread

        ts = []
        if not self.eval_trained and len(self.trains) > 0:
            ts.append(Thread(target=self._count, args=(self.trains, 'train')))
        if len(self.tests) > 0:
            ts.append(Thread(target=self._count, args=(self.tests, 'test')))

        [(t.start(), t.join()) for t in ts]

        if not self.eval_trained:
            self.train_pos_frac = self.nb_positive_train / self.nb_train
            print('Number train: %d, Positive ratio: %.5f' %
                  (self.nb_train, self.train_pos_frac))
            logger.info('Number train: %d, Positive ratio: %.5f' %
                        (self.nb_train, self.train_pos_frac))
        self.test_pos_frac = self.nb_positive_test / self.nb_test
        print('Number test: %d, Positive ratio: %.5f' % (self.nb_test,
                                                         self.test_pos_frac))

    def set_data_paths(self, trains=None, tests=None, predictions=None):
        """Interface to manually set the files paths for training, test and
        prediction.
        """

        self.trains = trains if isinstance(trains, list) else [trains]
        self.tests = tests if isinstance(tests, list) else [tests]
        self.predictions = predictions if isinstance(predictions,
                                                     list) else [predictions]

    def load_data(self,
                  data_name='train',
                  batch_size=4096,
                  _get_features=False,
                  feed_mode='batch'):
        """Interface to load training, test or prediction data, called from
        model_manager.

        # Arguments
            data_name: phase of model to feed data, can be 'train', 'test' or
              'validation', or 'prediction'. Using two initials of them also
              works, such as 'tr' for 'train'.
            batch_size: number of data in each batch, only important for
              training phase.
            _get_features: get number of features.
            feed_mode: can be 'batch' which load all the data only once and
              read them by batch to avoid RAM insufficiency, 'generator' which
              load data by samll batch and repeat infinitely over the whole
              data, or 'all' which load all the data into memory.

        # Return
            Compiled Keras model
        """

        self.batch_size = batch_size

        if self.data_format == 'parquet':

            if _get_features:
                if len(self.trains) > 0:
                    p = self.trains[0]
                else:
                    p = self.tests[0]
                tmp = pq.read_table(p).to_pandas()
                df_sample = self.dataframe_process(self.args, tmp)
                self.features = df_sample.columns.values[:-1]
                self.nb_features = self.features.shape[0]
                return

            # train dataset
            if 'tr' in data_name:
                if self.online_learning:
                    self.trains.sort()
                ps = self.trains
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('train')
                    print('Loading train data from %d parquet files' % len(ps))
                    df = self._read_all_data(ps)
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x, y = data[:, :-1], data[:, -1:]
                    x, y = DataLoader._resample_data(self.resample, x, y)
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('train', (x, y))
                else:
                    return self._parquet_read_generator(
                        ps, feed_mode, batch_size, self.resample)

            # validation dataset
            if 'va' in data_name or 'te' in data_name:
                ps = self.tests
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('validation')
                    df = pd.DataFrame()
                    print('Loading val data from %d parquet files' % len(ps))
                    df = self._read_all_data(ps)
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x, y = data[:, :-1], data[:, -1:]
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('validation', (x, y))
                else:
                    return self._parquet_read_generator(
                        ps, feed_mode, batch_size)

            # prediction dataset
            if 'pr' in data_name:
                ps = self.predictions
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('prediction')
                    df = pd.DataFrame()
                    print('Loading pred data from %d parquet files' % len(ps))
                    df = self._read_all_data(ps)
                    df[self.label] = 0  # add a null column as 'label'
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x = data[:, :-1]
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('prediction', x)
                else:
                    return self._parquet_read_generator(
                        ps, feed_mode, batch_size, True)

            raise ValueError('Invalid `data_name`: ' + data_name)

        if self.data_format == 'csv':
            if _get_features:
                if len(self.trains) > 0:
                    p = self.trains[0]
                else:
                    p = self.tests[0]
                tmp = pd.read_csv(p, chunksize=2)
                df_sample = self.dataframe_process(self.args, next(tmp))
                self.features = df_sample.columns.values[:-1]
                self.nb_features = self.features.shape[0]
                return

            # train dataset
            if 'tr' in data_name:
                if self.online_learning:
                    self.trains.sort()
                ps = self.trains
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('train')
                    print('Loading train data from %d csv files' % len(ps))
                    df = self._read_all_data(ps)
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x, y = data[:, :-1], data[:, -1:]
                    x, y = DataLoader._resample_data(self.resample, x, y)
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('train', (x, y))
                else:
                    return self._csv_read_generator(ps, feed_mode, batch_size,
                                                    self.resample)

            # validation dataset
            if 'va' in data_name or 'te' in data_name:
                ps = self.tests
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('validation')
                    print('Loading val data from %d csv files' % len(ps))
                    df = self._read_all_data(ps)
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x, y = data[:, :-1], data[:, -1:]
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('validation', (x, y))
                else:
                    return self._csv_read_generator(ps, feed_mode, batch_size)

            # prediction dataset
            if 'pred' in data_name:
                ps = self.predictions
                if len(ps) == 0:
                    return
                if feed_mode == 'all':
                    if self.load_cache:
                        return self._cache_data('prediction')
                    print('Loading pred data from %d csv files' % len(ps))
                    df = self._read_all_data(ps)
                    df[self.label] = 0
                    df = self.dataframe_process(self.args, df)
                    data = df.values
                    self._check_label(df)
                    x = data[:, :-1]
                    x = self.sep_cols(self.column_emb, x)
                    return self._cache_data('prediction', x)
                else:
                    return self._csv_read_generator(ps, feed_mode, batch_size,
                                                    True)

            raise ValueError('Invalid `data_name`: ' + data_name)
        raise ValueError('Invalid `data_format`: ' + data_format)
