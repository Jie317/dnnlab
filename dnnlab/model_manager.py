import gc
import os
import shutil
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time, strftime, sleep
from datetime import timedelta
from dnnlab.tensorboard import tensorBoard
from sklearn.metrics import auc, roc_auc_score, log_loss
from sklearn.metrics import precision_recall_curve, classification_report
from keras.callbacks import ProgbarLogger, CallbackList, History, BaseLogger
from keras.models import load_model
# from keras.utils import plot_model
import pandas as pd
from pprint import pprint
import json
import logging
logger = logging.getLogger('dnnlab')




class ModelManager(object):
    """Manage and control models during training and test (TODO: prediction)"""

    def __init__(self, args, data_loader, build_dnn_model, send_metric):
        super(ModelManager, self).__init__()
        self.args = args
        self.send_metric = send_metric or (lambda _, __: None)
        self.data = data_loader
        self.max_feature = int(eval(args.max_feature))
        self.model_name = args.model_name
        self.epochs = args.epochs
        self.input_length = self.data.nb_features
        self.batch_size = args.batch_size
        self.sess_id = '%s_%s_%s' % (args.id_prefix, self.data.data_label,
                                     args.model_name or args.machine_learning
                                     ) + ('_ol'
                                          if args.online_learning else '')
        self.steps_per_epoch = args.steps_per_epoch
        self.class_weight = {
            0: 1,
            1: eval(args.imb_learn[2:]) if 'cw' in args.imb_learn else 1
        }
        self.threshold = eval(
            args.imb_learn[2:]) if 'th' in args.imb_learn else .5
        self.build_dnn_model = build_dnn_model
        self.trained_path = args.trained_path or 'results/trained_models/last_model.h5'
        self.continue_train = args.continue_train
        self.eval_trained = args.eval_trained
        self.feed_mode = args.feed_mode
        self.no_save = args.prod or args.no_save
        self.export_option = args.prod or args.export_option
        self.summary = args.summary
        self.max_q_size = 256
        self.workers = 1
        self.verbose = args.verbose
        self.no_eval = args.no_eval
        self.online_learning = args.online_learning
        self.prod = args.prod
        self.val_feed_mode = args.val_feed_mode
        self.machine_learning = args.machine_learning
        self.load_data = self.data.load_data
        self.train_xy = (None, None)  # avoid reloading data
        self.validation_xy = [None, None]
        self.predict_x = None
        self.lst_thresholds = [
            0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.1, 0.2, 0.5
        ]
        self._make_dirs()
        if self.prod:
            self.callbacks = []
        else:
            tbCB = tensorBoard(
                'results/tensorboard/tbGraph_%s/' % self.sess_id,
                track_by_samples=True)
            self.callbacks = [tbCB]

    def _make_dirs(self):
        if self.prod:
            dirs = ['results/exported_model_cc/']
        else:
            dirs = [
                'results/trained_models/', 'results/tensorboard/',
                'results/backups/', 'results/figures_prc/',
                'results/different_thresholds/', '.cache/',
                'results/exported_model_cc/'
            ]
        [os.path.exists(d) or os.makedirs(d) for d in dirs]

    def _start_dnn(self):
        self.start_time = time()

        # build or load Keras model
        if self.eval_trained or self.continue_train:
            print('Loading model from: ', self.trained_path)
            self.model = load_model(self.trained_path)
        else:
            self.model = self.build_dnn_model(
                self.model_name, self.max_feature, self.input_length)

        # print model summar
        if self.summary:
            self.model.summary()
        # save graph
        # plot_model(self.model, to_file='results/model.png', show_shapes=True) # TMP

        # start training model
        if not self.eval_trained:
            print('\n', strftime('%c'))

            if self.feed_mode == 'all':
                history = self._train_on_all()

            elif self.feed_mode == 'batch':
                history = self._train_by_batch()

            elif self.feed_mode == 'generator':
                history = self._train_on_generator()

            else:
                raise ValueError('Invalid `feed_mode`: ' + self.feed_mode)

            final_train_loss = history.history['loss'][-1]

            logger.info('Finished training model, final training loss: %.4f' %
                        final_train_loss)
            loss_path = 'results/loss.txt'
            loss_dict = {'training loss': str(final_train_loss)}
            json.dump(loss_dict, open(loss_path, 'w'))

            self.send_metric('model_training_loss', final_train_loss)
            time_used = str(timedelta(seconds=int(time() - self.start_time)))
            print('Training runtime:', time_used)

        # store model and backup config
        if not (self.no_save or self.eval_trained):
            self._save_and_backup()

        # evaluate the model
        if not (self.prod or self.no_eval):
            print('Evaluation')
            if self.val_feed_mode == 'all':
                self.validation_xy = self.validation_xy or self.load_data(
                    'val', feed_mode='all')
                probs = self.model.predict(
                    self.validation_xy[0],
                    batch_size=self.batch_size * 64,
                    verbose=1)

            if self.val_feed_mode == 'batch':
                self.validation_xyb = self.load_data('val', feed_mode='batch')
                raise NotImplementedError

            self._get_metric_scores(self.validation_xy[1], probs,
                                    self.model_name)

        # export model for tensorflow serving
        if self.export_option:
            # self._export_model_for_tfserving(self.model)
            self._export_model_for_tfcc()

        # predict new data
        if self.args.predict_path:
            self.predict_x = self.load_data('pred', feed_mode='all')
            probs = self.model.predict(
                self.predict_x, batch_size=self.batch_size * 64, verbose=1)
            np.savetxt('results/predicted_probabilites.csv', probs, fmt='%.8f')
            print('Done prediction for data in %s\n' % self.args.predict_path)

    
    def _export_model_for_tfcc(self):
        import tensorflow as tf
        from keras import backend as K
        from os import path as osp


        K.set_learning_phase(0)
        with K.get_session() as sess:
            # Alias the outputs in the model - this sometimes makes them easier to access in TF
            pred = []
            # add another node to copy the output node
            new_output = tf.identity(self.model.output[0], name='click_proba')
            print('Output node name: ', 'click_proba')

            outdir = 'results/exported_model_cc/'
            name = 'graph.pb'

            # Write the graph in binary .pb file
            from tensorflow.python.framework import graph_util
            from tensorflow.python.framework import graph_io
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['click_proba'])
            graph_io.write_graph(constant_graph, outdir, name, as_text=False)
            print('Saved the constant graph (ready for inference) at: ', osp.join(outdir, name))


    def _export_model_for_tfserving(self, model):
        """Export model for tensorflow serving (not tested if they work with
        tensorflow c++)
        """
        if self.prod:
            do_export = 'y'
        else:
            do_export = input(
                'Export model for tensorflow serving? [Y/n]') or 'y'

        if 'y' in do_export.lower():
            import tensorflow as tf
            from keras import backend as K
            from tensorflow.python.saved_model import builder as saved_model_builder
            from tensorflow.python.saved_model import tag_constants
            from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
            if self.prod:
                version_num = 1
            else:
                version_num = input('Version number (int): (default: 1) ') or 1

            export_path = 'results/exported_model/%d/' % version_num
            if os.path.exists(export_path):
                shutil.rmtree(export_path)

            builder = saved_model_builder.SavedModelBuilder(export_path)

            signature = predict_signature_def(
                inputs={'request': model.input},
                outputs={'click_probability': model.output})

            K.set_learning_phase(0)
            with K.get_session() as sess:
                builder.add_meta_graph_and_variables(
                    sess=sess,
                    tags=[tag_constants.SERVING],
                    signature_def_map={'predict': signature})
                builder.save()

            print('Done exporting!\nYou can pass the exported model to'
                  ' tensorflow serving or reload it'
                  ' with tensorflow c++ API\n')
            logger.info('Exported model to path: %s' % export_path)

            return export_path

    def _start_xgboost(self, tr_xy, va_xy):
        from xgboost import XGBClassifier
        self.start_time = time()

        print('\nRunning Xgboost\n')
        self.model = XGBClassifier(
            max_depth=7,
            max_delta_step=1,
            silent=False,
            n_estimators=178,
            learning_rate=0.1,
            objective='binary:logistic',
            min_child_weight=1,
            scale_pos_weight=1)
        self.model.fit(
            *tr_xy, eval_set=[va_xy], eval_metric='logloss', verbose=True)

        train_time = str(timedelta(seconds=int(time() - self.start_time)))
        print('Training runtime:', train_time)
        probs = self.model.predict_proba(va_xy[0])[:, 1:]
        self._get_metric_scores(va_xy[1], probs, 'xgboost')

    def _start_randomforest(self, tr_xy, va_xy):
        from sklearn.ensemble import RandomForestClassifier
        self.start_time = time()

        print('\nRunning Random Forest\n')
        self.model = RandomForestClassifier(
            n_estimators=200,
            criterion='gini',
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_split=1e-07,
            bootstrap=True,
            oob_score=False,
            n_jobs=4,
            random_state=None,
            verbose=1,
            warm_start=False,
            class_weight=None)
        self.model.fit(*tr_xy)

        train_time = str(timedelta(seconds=int(time() - self.start_time)))
        print('Training runtime:', train_time)
        probs = self.model.predict_proba(va_xy[0])[:, 1:]
        self._get_metric_scores(va_xy[1], probs, 'random_forest')

    def _start_adaboost(self, tr_xy, va_xy):
        from sklearn.ensemble import AdaBoostClassifier
        self.start_time = time()

        print('\nRunning Adaboost\n')
        self.model = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=.3,
            algorithm='SAMME.R',
            random_state=None)
        self.model.fit(*tr_xy)

        train_time = str(timedelta(seconds=int(time() - self.start_time)))
        print('Training runtime:', train_time)
        probs = self.model.predict_proba(va_xy[0])[:, 1:]
        self._get_metric_scores(va_xy[1], probs, 'adaboost',
                                feature_importance)

    def _get_metric_scores(self, y_real, y_proba, model_name):
        """Calculate metric scores. Input shape must be (n, 1)

        # Arguments
            y_real: 1D array-like ground truth (correct) target values.
            y_proba: 1D array-like estimated probabilities as returned by a
              classifier (model).
            model_name: name of the model to evaluate.

        # Returns
            A set of evaluation results stored in the generated folder
              'results', where the file 'results.csv' appends scalar values,
              the folder 'different_thresholds' stores a table of different
              decision thresholds and their corresponding scores of
              precision, recall, true positives, etc. The precision-recall
              curve is registered in the folder 'pics_prc'.
        """

        def metrics_prf(y_real, y_pred):
            """Compute precision, recall and f-measure"""
            TP = np.sum(y_pred * y_real).astype('int')
            real_pos = np.sum(y_real).astype('int')
            pred_pos = np.sum(y_pred).astype('int')
            P = TP / (pred_pos + 1e-15)
            R = TP / (real_pos + 1e-15)
            Fm = 2 * P * R / (P + R + 1e-15)
            FP = pred_pos - TP
            FN = real_pos - TP
            TN = len(y_real) - real_pos - FP
            return P, R, Fm, TP, FP, FN, TN, real_pos, pred_pos

        def get_prf_for_diff_thresholds(y_real, y_proba, threshold):
            pred_classes = (y_proba > threshold).astype('int8')
            P, R, Fm, TP, FP, FN, TN, _, _ = metrics_prf(y_real, pred_classes)
            return Fm, P, R, TP, FP, FN, TN

        # 1 logloss
        logloss = log_loss(y_real, y_proba)

        # 2 ROC AUC score
        aucRoc = roc_auc_score(y_real, y_proba)

        # 3 precision-recall curve and PR AUC score
        precision, recall, thresholds = precision_recall_curve(y_real, y_proba)
        aucPrc = auc(recall, precision)

        # plt.clf()
        plt.plot(
            recall,
            precision,
            label='%s (aucPR=%.4f)' % (self.sess_id, aucPrc))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.05])
        # plt.title('%s - PRCurve ' % self.sess_id)
        plt.legend(loc="upper right")
        # plt.show()

        # 4 confusion matrix
        pred_classes = (y_proba > self.threshold).astype('int8')
        report = classification_report(y_real, pred_classes)
        P, R, Fm, TP, FP, FN, TN, RP, PP = metrics_prf(y_real, pred_classes)
        print('\n', report)
        print('\nUsing threshold %f' % self.threshold)
        print(
            ' - Precision: %.4f (%d/%d)\n - Recall: %.4f (%d/%d)\n - TP: %d\t'
            '- FP: %d\n - FN: %d\t- TN: %d\n - F1: %.4f\t- Logloss: %.4f\n'
            ' - aucRoc: %.4f - aucPR: %.4f\n' % (P, TP, PP, R, TP, RP, TP, FP,
                                                 FN, TN, Fm, logloss, aucRoc,
                                                 aucPrc))

        # 5 different thresholds
        lst_pr_segs = [
            get_prf_for_diff_thresholds(y_real, y_proba, th)
            for th in self.lst_thresholds
        ]

        # 6 feature importance
        has_f_imp = hasattr(self.model, 'feature_importances_')
        if has_f_imp:
            f_imp = pd.DataFrame({
                'Feature': self.data.features,
                'Score': self.model.feature_importances_
            })
            f_imp.sort_values(by='Score', ascending=False, inplace=True)
            f_imp = f_imp.round(8)
            print(f_imp)

        print('Train pos ratio: %.8f' % (self.data.train_pos_frac),
              'Test pos ratio: %.8f' % self.data.test_pos_frac)

        # write to results.csv
        if not self.no_save:
            np.savetxt(
                'results/different_thresholds/%s' % self.sess_id,
                np.hstack((list(zip(self.lst_thresholds)), lst_pr_segs)),
                fmt='%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d',
                header='Threshold\tF\tP\tR\tTP\tFP\tFN\tTN')
            plt.savefig('results/figures_prc/%s_%.4f.png' % (self.sess_id,
                                                             aucPrc))
            path_results_csv = 'results/results.csv'
            exists_csv = os.path.exists(path_results_csv)
            with open(path_results_csv, 'a+') as res:
                if not exists_csv:
                    res.write(
                        'Data\taucRoc\tlogLoss\taucPrc\tF\tP\tR\tTP\tFP\tFN'
                        '\tTN\tnbTrain\tratioTrain\tnbTest\tratioTest\t'
                        'epochs\trunTime\tDate\n\n')
                res.write(
                    '%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t'
                    '%d\t%.4f\t%d\t%.4f\t%d\t%s\t%s\n' %
                    (self.sess_id, aucRoc, logloss, aucPrc, Fm, P, R, TP, FP,
                     FN, TN, self.data.nb_train, self.data.train_pos_frac,
                     self.data.nb_test, self.data.test_pos_frac, self.epochs,
                     timedelta(seconds=int(time() - self.start_time)),
                     strftime('%c')))

            if has_f_imp:
                f_imp.to_csv(
                    'results/feature_importances_%s' % model_name,
                    sep='\t',
                    index=None,
                    float_format='%.8f')

            print('\nWritten results to files.\n\n')

    def _train_by_batch(self):
        # batch finite generator should be loaded within epoch loop
        logger.info('Start training by batch')
        self.validation_xy = self.load_data('val', feed_mode='all')
        do_validation = bool(self.validation_xy)

        # prepare display labels in tensorboard
        out_labels = self.model._get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.model.history = History()
        callbacks = [BaseLogger()] + (self.callbacks or []) + [self.model.history]
        # callbacks = (self.callbacks or []) + [self.model.history]
        if self.verbose:
            callbacks += [ProgbarLogger(count_mode='samples')]
        callbacks = CallbackList(callbacks)

        # it's possible to callback a different model than this model
        if hasattr(self.model, 'callback_model') and self.model.callback_model:
            callback_model = self.model.callback_model
        else:
            callback_model = self.model
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': self.epochs,
            'samples': self.data.nb_train,
            'verbose': self.verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        for epoch in range(self.epochs):
            start_e = time()
            callbacks.on_epoch_begin(epoch)
            xy_gen = self.load_data('train', feed_mode='batch')
            logger.info('New training epoch')
            for batch_index, (x, y) in enumerate(xy_gen):
                # build batch logs
                batch_logs = {}
                if isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = self.model.train_on_batch(x, y)

                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o
                callbacks.on_batch_end(batch_index, batch_logs)

                if (batch_index + 1) % 1000 == 0 and do_validation:
                    val_outs = self.model.evaluate(
                        *self.validation_xy, batch_size=81920, verbose=0)
                    batch_logs = {}
                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    for l, o in zip(out_labels, val_outs):
                        batch_logs['val_' + l] = o
                    print(' - Eval inside: %.6f' % val_outs[0])
                    for cb in self.callbacks:
                        if cb.__class__ == tensorBoard:
                            cb.on_batch_end(
                                batch_index, batch_logs, count=False)

            epoch_logs = {}
            if do_validation:
                val_outs = self.model.evaluate(
                    *self.validation_xy, batch_size=81920, verbose=0)
                if not isinstance(val_outs, list):
                    val_outs = [val_outs]
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o

            callbacks.on_batch_end(epoch, epoch_logs)
            callbacks.on_epoch_end(epoch, epoch_logs)

            elapsed_e = timedelta(seconds=int(time() - start_e))
            self.send_metric('elapsed_per_epoch', elapsed_e)

            if not self.no_save and do_validation and (epoch !=
                                                       self.epochs - 1):
                self.model.save(
                    'results/trained_models/%s_ctr_model_%.4f_epoch_%d.h5' %
                    (self.sess_id, val_outs[0], epoch))

        callbacks.on_train_end()
        return self.model.history

    def _train_on_all(self):
        self.train_xy = self.load_data('train', feed_mode='all')
        self.validation_xy = self.load_data('val', feed_mode='all')

        return self.model.fit(
            *self.train_xy,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=self.validation_xy,
            shuffle=True,
            callbacks=self.callbacks)

    def _train_on_generator(self):
        tr_xy = self.load_data('train', feed_mode='generator')
        self.validation_xy = self.load_data('val', feed_mode='all')

        return self.model.fit_generator(
            tr_xy,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            validation_data=self.validation_xy,
            class_weight=self.class_weight,
            max_q_size=self.max_q_size,
            workers=self.workers,
            pickle_safe=False,
            initial_epoch=0,
            verbose=self.verbose,
            callbacks=self.callbacks)

    def _save_and_backup(self):
        self.model.save('results/trained_models/%s_model.h5' % (self.sess_id))
        self.model.save(self.trained_path)
        file_name = None  # TODO: get name of the file to backup
        # con_p = 'results/backups/%s_%s.py' % (self.sess_id, file_name)
        # shutil.copyfile(file_name, con_p)
        # with open(con_p, 'a+') as cout:
        #     cout.write('\n# Arguments applied to this run:\n# ' +
        #                str(self.args))
        print('\nSaved model to %s' % self.trained_path)

    def start(self):
        """Start. """

        if self.model_name:
            self._start_dnn()

        ml = self.machine_learning
        if ml:
            tr_xy = self.train_xy or self.load_data('train', feed_mode='all')
            va_xy = self.validation_xy or self.load_data(
                'test', feed_mode='all')
            if 'xgb' in ml:
                self._start_xgboost(tr_xy, va_xy)
            if 'rf' in ml:
                self._start_randomforest(tr_xy, va_xy)
            if 'adab' in ml:
                self._start_adaboost(tr_xy, va_xy)
