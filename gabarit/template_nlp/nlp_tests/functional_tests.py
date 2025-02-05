#!/usr/bin/env python3
# Starts all functional tests
# Copyright (C) <2018-2022>  <Agence Data Services, DSI Pôle Emploi>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Libs unittest
import unittest

# utils libs
import os
import json
import shutil
import tempfile
import subprocess
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
from datetime import datetime

from test_template_nlp import utils
from test_template_nlp.models_training import model_huggingface, model_aggregation, model_class
from test_template_nlp.models_training.models_sklearn import (model_tfidf_svm, model_tfidf_gbt, model_tfidf_lgbm,
                                                              model_tfidf_sgdc)
from test_template_nlp.models_training.models_tensorflow import (model_tfidf_dense, model_embedding_lstm, model_embedding_lstm_attention,
                                                                 model_embedding_lstm_structured_attention, model_embedding_lstm_gru,
                                                                 model_embedding_cnn)

def remove_dir(path):
    if os.path.isdir(path): shutil.rmtree(path)


def get_last_model_created(path_to_folder):
    list_models = list(os.walk(path_to_folder))[0][1]
    list_models.sort()
    return list_models[-1]


def test_reload_model(test_class, model_type, arguments, change_file=None):
    if model_type == model_aggregation.ModelAggregation:
        sub_model_1 = model_tfidf_svm.ModelTfidfSvm()
        sub_model_2 = model_tfidf_gbt.ModelTfidfGbt()
        arguments['list_models'] = [sub_model_1, sub_model_2]
    model = model_type(**arguments)
    x_train = ['technique triage logistique fabricants', 'technique', 'triage', 'logistique fabricants', 'technique', 'technique logistique fabricants', 
               'triage logistique', 'technique triage fabricants', 'logistique', 'triage']
    y_train = ['class_1', 'class_2', 'class_3', 'class_1', 'class_1', 'class_2', 'class_3', 'class_1', 'class_2', 'class_3']
    x_valid = ['fabricants', 'logistique fabricants', 'triage', 'technique fabricants', 'technique technique', 'logistique logistique']
    y_valid = ['class_2', 'class_2', 'class_3', 'class_1', 'class_2', 'class_1']
    x_test = x_train + x_valid + ['logistique technique', 'triage fabricants', 'fabricants fabricants']
    model.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)
    model.save()
    model_name = os.path.split(model.model_dir)[1]

    dict_change_file = {'sklearn_pipeline_file': ('sklearn_pipeline_standalone.pkl', 'sklearn_pipeline_standalone_2.pkl'),
                        'weights_file': ('best.hdf5', 'best_2.hdf5'),
                        'tokenizer_file': ('embedding_tokenizer.pkl', 'embedding_tokenizer_2.pkl'),
                        'tfidf_file': ('tfidf_standalone.pkl', 'tfidf_standalone_2.pkl'),
                        'aggregation_function_file': ('aggregation_function.pkl', 'aggregation_function_2.pkl'), 
                        'hf_model_dir': ('hf_model', 'hf_model_2'),
                        'hf_tokenizer_dir': ('hf_tokenizer', 'hf_tokenizer_2')}

    if change_file is None:
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_reload_model.py -m {model_name}"
    else:
        old_file_name, new_file_name = dict_change_file[change_file]
        shutil.move(os.path.join(model.model_dir, old_file_name), os.path.join(model.model_dir, new_file_name))
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_reload_model.py -m {model_name} --{change_file} {new_file_name}"
    test_class.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

    path_to_model = os.path.split(model.model_dir)[0]
    new_model_name = get_last_model_created(path_to_model)
    new_model_dir = os.path.join(path_to_model, new_model_name)
    new_model, new_conf = model_class.ModelClass.load_model(model_dir=new_model_dir)
    test_same_model_predictions(test_class, model, new_model, x_test)
    test_class.assertNotEqual(model.model_dir, new_model.model_dir)
    return model, new_model


def test_same_model_tfidf(test_class, model, new_model, name_sub_model, model_equal_attributes, model_almost_equal_attributes):
    tfidf = model.tfidf
    new_tfidf = new_model.tfidf
    for attribute in ['max_df', 'min_df']:
        test_class.assertAlmostEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
    for attribute in ['ngram_range', 'norm']:
        test_class.assertEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
    sub_model = getattr(model, name_sub_model)
    new_sub_model = getattr(new_model, name_sub_model)
    for attribute in model_equal_attributes:
        test_class.assertEqual(getattr(sub_model, attribute), getattr(new_sub_model, attribute))
    for attribute in model_almost_equal_attributes:
        test_class.assertAlmostEqual(getattr(sub_model, attribute), getattr(new_sub_model, attribute))


def test_same_model_not_tfidf(test_class, model, new_model, equal_attributes, almost_equal_attributes):
    for attribute in equal_attributes:
        test_class.assertEqual(getattr(model, attribute), getattr(new_model, attribute))
    for attribute in almost_equal_attributes:
        test_class.assertAlmostEqual(getattr(model, attribute), getattr(new_model, attribute))


def test_same_model_predictions(test_class, model, new_model, x_test):
    test_class.assertEqual(type(model), type(new_model))
    list_classes = model.list_classes
    list_classes.sort()
    new_list_classes = new_model.list_classes
    new_list_classes.sort()
    test_class.assertEqual(list_classes, new_list_classes)
    model_list_predict = list(model.predict(x_test))
    new_model_list_predict = list(new_model.predict(x_test))
    test_class.assertEqual(model_list_predict, new_model_list_predict)


class Case1_e2e_pipeline(unittest.TestCase):
    '''Class to test the project end to end'''

    def test01_CreateSamples(self):
        '''Test of the file utils/0_create_samples.py'''
        print("Test of the file utils/0_create_samples.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py --overwrite -f mono_class_mono_label.csv -n 15"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_15_samples.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_15_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df.shape[0], 15)
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py -f mono_class_mono_label.csv -n 15"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py --overwrite -f mono_class_mono_label.csv -n 15"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

        # Double files
        double_files_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_create_samples.py --overwrite -f mono_class_mono_label.csv multi_class_mono_label.csv -n 2000"
        self.assertEqual(subprocess.run(double_files_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_2000_samples.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'multi_class_mono_label_2000_samples.csv')))
        df1 = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        df2 = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/multi_class_mono_label_2000_samples.csv", sep=';', encoding='utf-8')
        self.assertEqual(df1.shape[0], 200)
        self.assertEqual(df2.shape[0], 200)  # 200 row max

    def test02_GetEmbeddingDict(self):
        '''Test of the file utils/0_get_embedding_dict.py'''
        print("Test of the file utils/0_get_embedding_dict.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_get_embedding_dict.py --overwrite -f custom.300.vec"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'custom.300.pkl')))
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_get_embedding_dict.py -f custom.300.vec"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_get_embedding_dict.py --overwrite -f custom.300.vec"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

    def test03_MergeFiles(self):
        '''Test of the file utils/0_merge_files.py'''
        print("Test of the file utils/0_merge_files.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_merge_files.py --overwrite -f mono_class_mono_label.csv multi_class_mono_label.csv -c x_col y_col -o merged_file.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'merged_file.csv')))
        df = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/merged_file.csv", sep=';', encoding='utf-8')
        self.assertGreater(df.shape[0], 200)  # We check that there are more than 200 elements (ie. the size of one of the two files)
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_merge_files.py -f mono_class_mono_label.csv multi_class_mono_label.csv -c x_col y_col -o merged_file.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_merge_files.py --overwrite -f mono_class_mono_label.csv multi_class_mono_label.csv -c x_col y_col -o merged_file.csv"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

    def test04_ReloadModel(self):
        '''Test of the file utils/0_reload_model.py'''
        print("Test of the file utils/0_reload_model.py")

        # ------------------------------------
        # Sklearn Models
        # ------------------------------------

        # ModelTfidfSvm
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        svc_params = {'penalty':'l2', 'loss':'hinge', 'C':0.9}
        model, new_model = test_reload_model(self, model_tfidf_svm.ModelTfidfSvm, {'tfidf_params': tfidf_params, 'svc_params': svc_params})
        test_same_model_tfidf(self, model, new_model, 'svc', ['penalty', 'loss', 'fit_intercept'], ['C'])
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelTfidfGbt
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        gbt_params = {'learning_rate':0.11, 'n_estimators':90, 'min_samples_split':3}
        model, new_model = test_reload_model(self, model_tfidf_gbt.ModelTfidfGbt, {'tfidf_params': tfidf_params, 'gbt_params': gbt_params})
        test_same_model_tfidf(self, model, new_model, 'gbt', ['n_estimators', 'min_samples_split'], ['learning_rate'])
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelTfidfLgbm
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        lgbm_params = {'num_leaves': 29, 'max_depth': 30, 'learning_rate': 0.11, 'n_estimators': 98, 'min_split_gain': 0.01}
        model, new_model = test_reload_model(self, model_tfidf_lgbm.ModelTfidfLgbm, {'tfidf_params': tfidf_params, 'lgbm_params': lgbm_params})
        test_same_model_tfidf(self, model, new_model, 'lgbm', ['num_leaves', 'max_depth', 'n_estimators'], ['learning_rate', 'min_split_gain'])
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelTfidfSgdc
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        sgdc_params = {'penalty': 'l1', 'alpha': 0.0002, 'l1_ratio': 0.09}
        model, new_model = test_reload_model(self, model_tfidf_sgdc.ModelTfidfSgdc, {'tfidf_params': tfidf_params, 'sgdc_params': sgdc_params})
        test_same_model_tfidf(self, model, new_model, 'sgdc', ['loss', 'penalty'], ['alpha', 'l1_ratio'])
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ------------------------------------
        # Keras Models
        # ------------------------------------

        # Attributes for ModelKeras
        equal_attributes_keras = ['batch_size', 'epochs', 'patience', 'embedding_name']
        almost_equal_attributes_keras = ['validation_split']
        
        # ModelEmbeddingCnn
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_cnn.ModelEmbeddingCnn, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelEmbeddingLstmAttention
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_lstm_attention.ModelEmbeddingLstmAttention, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelEmbeddingLstmGru
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_lstm_gru.ModelEmbeddingLstmGru, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelEmbeddingLstmStructuredAttention
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelEmbeddingLstm
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_lstm.ModelEmbeddingLstm, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelTfidfDense
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        equal_attributes = []
        model, new_model = test_reload_model(self, model_tfidf_dense.ModelTfidfDense, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'tfidf_params':tfidf_params})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        tfidf = model.tfidf
        new_tfidf = new_model.tfidf
        for attribute in ['max_df', 'min_df']:
            self.assertAlmostEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
        for attribute in ['ngram_range', 'norm']:
            self.assertEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ------------------------------------
        # Other Models
        # ------------------------------------

        # ModelHuggingFace
        equal_attributes = ['transformer_name', 'batch_size', 'epochs', 'patience']
        almost_equal_attributes = ['validation_split']
        model, new_model = test_reload_model(self, model_huggingface.ModelHuggingFace, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 4, 'validation_split':0.1, 'patience': 4})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes, almost_equal_attributes)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ModelAggregation
        equal_attributes = ['using_proba']
        almost_equal_attributes = []
        model, new_model = test_reload_model(self, model_aggregation.ModelAggregation, {'aggregation_function': "proba_argmax", 'using_proba': True})
        test_same_model_not_tfidf(self, model, new_model, equal_attributes, almost_equal_attributes)
        self.assertEqual(model.aggregation_function.__name__, new_model.aggregation_function.__name__)
        for sub_model, new_sub_model in zip(model.sub_models, new_model.sub_models):
            sub_model = sub_model['model']
            new_sub_model = new_sub_model['model']
            if hasattr(sub_model, 'svc'):
                test_same_model_tfidf(self, sub_model, new_sub_model, 'svc', ['penalty', 'loss', 'fit_intercept'], ['C'])
            if hasattr(sub_model, 'gbt'):
                test_same_model_tfidf(self, sub_model, new_sub_model, 'gbt', ['n_estimators', 'min_samples_split'], ['learning_rate'])
        for sub_model in model.sub_models:
            remove_dir(sub_model['model'].model_dir)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # ------------------------------------
        # Check file paths
        # ------------------------------------

        # sklearn_pipeline_file
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        svc_params = {'penalty':'l2', 'loss':'hinge', 'C':0.9}
        model, new_model = test_reload_model(self, model_tfidf_svm.ModelTfidfSvm, {'tfidf_params': tfidf_params, 'svc_params': svc_params},
                                             change_file='sklearn_pipeline_file')
        test_same_model_tfidf(self, model, new_model, 'svc', ['penalty', 'loss', 'fit_intercept'], ['C'])
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # weights_file
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_cnn.ModelEmbeddingCnn, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999},
                                            change_file='weights_file')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # tokenizer_file
        equal_attributes = ['max_sequence_length', 'max_words', 'padding', 'truncating', 'tokenizer_filters']
        model, new_model = test_reload_model(self, model_embedding_cnn.ModelEmbeddingCnn, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'max_sequence_length':199, 'max_words': 99999},
                                            change_file='tokenizer_file')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # tfidf_file
        tfidf_params = {'min_df': 2, 'max_df': 0.9, 'norm':'l1', 'ngram_range':(1, 2)}
        equal_attributes = []
        model, new_model = test_reload_model(self, model_tfidf_dense.ModelTfidfDense, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 16, 'validation_split':0.1, 'patience': 4,
                                                                                           'tfidf_params':tfidf_params}, change_file='tfidf_file')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes_keras+equal_attributes, almost_equal_attributes_keras)
        tfidf = model.tfidf
        new_tfidf = new_model.tfidf
        for attribute in ['max_df', 'min_df']:
            self.assertAlmostEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
        for attribute in ['ngram_range', 'norm']:
            self.assertEqual(getattr(tfidf, attribute), getattr(new_tfidf, attribute))
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # hf_model_dir
        equal_attributes = ['transformer_name', 'batch_size', 'epochs', 'patience']
        almost_equal_attributes = ['validation_split']
        model, new_model = test_reload_model(self, model_huggingface.ModelHuggingFace, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 4, 'validation_split':0.1, 'patience': 4},
                                             change_file='hf_model_dir')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes, almost_equal_attributes)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # hf_tokenizer_dir
        equal_attributes = ['transformer_name', 'batch_size', 'epochs', 'patience']
        almost_equal_attributes = ['validation_split']
        model, new_model = test_reload_model(self, model_huggingface.ModelHuggingFace, {'embedding_name': "custom.300.pkl", 'epochs': 3,
                                                                                           'batch_size': 4, 'validation_split':0.1, 'patience': 4},
                                             change_file='hf_tokenizer_dir')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes, almost_equal_attributes)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

        # aggregation_function_file
        equal_attributes = ['using_proba']
        almost_equal_attributes = []
        model, new_model = test_reload_model(self, model_aggregation.ModelAggregation, {'aggregation_function': "proba_argmax", 'using_proba': True},
                                             change_file='aggregation_function_file')
        test_same_model_not_tfidf(self, model, new_model, equal_attributes, almost_equal_attributes)
        self.assertEqual(model.aggregation_function.__name__, new_model.aggregation_function.__name__)
        for sub_model, new_sub_model in zip(model.sub_models, new_model.sub_models):
            sub_model = sub_model['model']
            new_sub_model = new_sub_model['model']
            if hasattr(sub_model, 'svc'):
                test_same_model_tfidf(self, sub_model, new_sub_model, 'svc', ['penalty', 'loss', 'fit_intercept'], ['C'])
            if hasattr(sub_model, 'gbt'):
                test_same_model_tfidf(self, sub_model, new_sub_model, 'gbt', ['n_estimators', 'min_samples_split'], ['learning_rate'])
        for sub_model in model.sub_models:
            remove_dir(sub_model['model'].model_dir)
        remove_dir(model.model_dir)
        remove_dir(new_model.model_dir)

    def test05_SplitTrainValidTest(self):
        '''Test of the file utils/0_split_train_valid_test.py'''
        print("Test of the file utils/0_split_train_valid_test.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_test.csv')))
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 120)
        self.assertEqual(df_valid.shape[0], 40)
        self.assertEqual(df_test.shape[0], 40)
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

        # Test of perc_x arguments
        test_perc = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.3 --perc_valid 0.6 --perc_test 0.1 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_perc, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        self.assertEqual(df_train.shape[0], 60)
        self.assertEqual(df_valid.shape[0], 120)
        self.assertEqual(df_test.shape[0], 20)

        # Test split_type stratified
        test_stratified = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type stratified --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_stratified, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        # Check number of elements
        self.assertGreater(df_train.shape[0], 114)
        self.assertLess(df_train.shape[0], 126)
        self.assertGreater(df_valid.shape[0], 34)
        self.assertLess(df_valid.shape[0], 46)
        self.assertGreater(df_test.shape[0], 34)
        self.assertLess(df_test.shape[0], 46)
        # Check stratified
        self.assertGreater(sum(df_train.y_col == "oui")/df_train.shape[0], 0.47)
        self.assertLess(sum(df_train.y_col == "oui")/df_train.shape[0], 0.53)
        self.assertGreater(sum(df_valid.y_col == "oui")/df_valid.shape[0], 0.47)
        self.assertLess(sum(df_valid.y_col == "oui")/df_valid.shape[0], 0.53)
        self.assertGreater(sum(df_test.y_col == "oui")/df_test.shape[0], 0.47)
        self.assertLess(sum(df_test.y_col == "oui")/df_test.shape[0], 0.53)

        # Test split_type hierarchical
        test_hierarchical = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type hierarchical --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        self.assertEqual(subprocess.run(test_hierarchical, shell=True).returncode, 0)
        df_train = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_train.csv", sep=';', encoding='utf-8')
        df_valid = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_valid.csv", sep=';', encoding='utf-8')
        df_test = pd.read_csv(f"{full_path_lib}/test_template_nlp-data/mono_class_mono_label_test.csv", sep=';', encoding='utf-8')
        # Check number of elements
        self.assertGreater(df_train.shape[0], 114)
        self.assertLess(df_train.shape[0], 126)
        self.assertGreater(df_valid.shape[0], 34)
        self.assertLess(df_valid.shape[0], 46)
        self.assertGreater(df_test.shape[0], 34)
        self.assertLess(df_test.shape[0], 46)
        # Check hierarchical
        self.assertFalse(any([_ in df_valid.x_col.values for _ in df_train.x_col.values]))
        self.assertFalse(any([_ in df_test.x_col.values for _ in df_train.x_col.values]))
        self.assertFalse(any([_ in df_valid.x_col.values for _ in df_test.x_col.values]))

    def test06_sweetviz_report(self):
        '''Test of the file utils/0_sweetviz_report.py'''
        print("Test of the file utils/0_sweetviz_report.py")

        # We first create a sweetviz configuration file
        config_path = os.path.join(full_path_lib, "test_config.json")
        if os.path.exists(config_path):
            os.remove(config_path)
        with open(config_path, 'w') as f:
            json.dump({"open_browser": False}, f)

        report_path = os.path.join(full_path_lib, "test_template_nlp-data", "reports", "sweetviz")
        remove_dir(report_path)

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        list_filenames = list(os.walk(report_path))[0][2]
        self.assertTrue(len([filename for filename in list_filenames if "report_source" in filename and "report_source_w" not in filename]) == 1)
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_sweetviz_report.py -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label.csv --source_names source --config {config_path} --mlflow_experiment sweetviz_experiment_1"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

        # Compare datasets
        test_compare = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_sweetviz_report.py --overwrite -s mono_class_mono_label_train.csv --source_names train -c mono_class_mono_label_valid.csv mono_class_mono_label_test.csv --compare_names valid test --config {config_path} --mlflow_experiment sweetviz_experiment_2"
        self.assertEqual(subprocess.run(test_compare, shell=True).returncode, 0)
        list_filenames = list(os.walk(report_path))[0][2]
        self.assertTrue(len([filename for filename in list_filenames if "report_train_valid" in filename]) == 1)
        self.assertTrue(len([filename for filename in list_filenames if "report_train_test" in filename]) == 1)

        # With target
        # Sweetviz does not with categorical target. Hence, we'll create a temporary dataframe with a binary target.
        data_path = os.path.join(full_path_lib, 'test_template_nlp-data')
        original_dataset_path = os.path.join(data_path, 'mono_class_mono_label.csv')
        with tempfile.NamedTemporaryFile(dir=data_path) as tmp_file:
            # Read dataset, add a tmp target as binary class & save it in the tmp file
            df = pd.read_csv(original_dataset_path, sep=';', encoding='utf-8')
            df['tmp_target'] = df['y_col'].apply(lambda x: 1. if x == 'oui' else 0.)
            df.to_csv(tmp_file.name, sep=';', encoding='utf-8', index=None)
            test_target = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_sweetviz_report.py --overwrite -s {tmp_file.name} --source_names source_with_target -t tmp_target --config {config_path} --mlflow_experiment sweetviz_experiment_3"
            self.assertEqual(subprocess.run(test_target, shell=True).returncode, 0)
            list_filenames = list(os.walk(report_path))[0][2]
            self.assertTrue(len([filename for filename in list_filenames if "report_source_with_target" in filename]) == 1)

        # Clean up sweetviz config path (useful ?)
        os.remove(config_path)

    def test07_PreProcessData(self):
        '''Test of the file 1_preprocess_data.py'''
        print("Test of the file 1_preprocess_data.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py --overwrite -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check if exists
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train_preprocess_P1.csv')))
        self.assertTrue(os.path.exists(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid_preprocess_P1.csv')))
        df_train = pd.read_csv(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_train_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        df_valid = pd.read_csv(os.path.join(full_path_lib, 'test_template_nlp-data', 'mono_class_mono_label_valid_preprocess_P1.csv'), sep=';', encoding='utf-8', skiprows=1)
        # Check col preprocessed_text
        self.assertTrue('preprocessed_text' in df_train.columns)
        self.assertTrue('preprocessed_text' in df_valid.columns)
        # Check preprocess (at least lower)
        self.assertEqual(list(df_train.preprocessed_text.str.lower().values), list(df_train.preprocessed_text.values))
        self.assertEqual(list(df_valid.preprocessed_text.str.lower().values), list(df_valid.preprocessed_text.values))
        # retry without overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # retry with overwrite
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py --overwrite -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)

    def test08_TrainingE2E(self):
        '''Test of the file 2_training.py'''
        print("Test of the file 2_training.py")

        # "Basic" case
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/2_training.py -f mono_class_mono_label_train_preprocess_P1.csv -x preprocessed_text -y y_col --filename_valid mono_class_mono_label_valid_preprocess_P1.csv --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm')  # tfidf svm by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 1)

        # Multilabel - no preprocess - no valid
        multilabel_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/2_training.py -f mono_class_multi_label.csv -x x_col -y y_col_1 y_col_2 --mlflow_experiment gabarit_ci/mlflow_test"
        self.assertEqual(subprocess.run(multilabel_run, shell=True).returncode, 0)
        # Check model saved
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm')  # tfidf svm by default
        self.assertTrue(os.path.exists(save_model_dir))
        listdir = os.listdir(os.path.join(save_model_dir))
        self.assertEqual(len(listdir), 2)

    def test09_PredictE2E(self):
        '''Test of the file 3_predict.py'''
        print("Test of the file 3_predict.py")

        # "Basic" case
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm')  # tfidf svm by default
        listdir = sorted(os.listdir(os.path.join(save_model_dir)))
        model_name = listdir[0]  # First one trained (ordered by date)
        basic_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_mono_label_test.csv -x x_col -m {model_name}"
        self.assertEqual(subprocess.run(basic_run, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_nlp-data', 'predictions', 'mono_class_mono_label_test')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # With y_col
        run_with_y = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_mono_label_test.csv -x x_col -y y_col -m {model_name}"
        self.assertEqual(subprocess.run(run_with_y, shell=True).returncode, 0)
        # Check predictions
        listdir = sorted(os.listdir(os.path.join(save_predictions_dir)))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[-1], 'predictions_with_y_true.csv')))  # last folder

        # Multilabel
        save_model_dir = os.path.join(full_path_lib, 'test_template_nlp-models', 'model_tfidf_svm')  # tfidf svm by default
        listdir = sorted(os.listdir(os.path.join(save_model_dir)))
        model_name = listdir[1]  # Second one trained (ordered by date)
        multilabel_run = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_multi_label.csv -x x_col -m {model_name}"
        self.assertEqual(subprocess.run(multilabel_run, shell=True).returncode, 0)
        # Check predictions
        save_predictions_dir = os.path.join(full_path_lib, 'test_template_nlp-data', 'predictions', 'mono_class_multi_label')
        self.assertTrue(os.path.exists(save_predictions_dir))
        listdir = os.listdir(os.path.join(save_predictions_dir))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[0], 'predictions.csv')))

        # Multilabel - with y_col
        multilabel_run_with_y = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/3_predict.py -f mono_class_multi_label.csv -x x_col -y y_col_1 y_col_2 -m {model_name}"
        self.assertEqual(subprocess.run(multilabel_run_with_y, shell=True).returncode, 0)
        # Check predictions
        listdir = sorted(os.listdir(os.path.join(save_predictions_dir)))
        self.assertTrue(os.path.exists(os.path.join(save_predictions_dir, listdir[-1], 'predictions_with_y_true.csv')))  # last folder


def test_model_mono_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Verify gabarit version
    with open(os.path.join(test_model.model_dir, 'configurations.json'), 'r') as f:
        configurations = json.load(f)
        test_class.assertTrue("gabarit_version" in configurations)
    # Try some functions
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertEqual(list(preds), ['non', 'oui'])
    # predict_proba
    index_non = test_model.list_classes.index('non')
    index_oui = test_model.list_classes.index('oui')
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertGreater(probas[0][index_non], 0.5)
    test_class.assertLess(probas[0][index_oui], 0.5)
    test_class.assertGreater(probas[1][index_oui], 0.5)
    test_class.assertLess(probas[1][index_non], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis'], return_proba=True)
    test_class.assertGreater(probas2[0][index_non], 0.5)
    test_class.assertLess(probas2[0][index_oui], 0.5)
    test_class.assertGreater(probas2[1][index_oui], 0.5)
    test_class.assertLess(probas2[1][index_non], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis'])
    test_class.assertEqual(list(pred_proba[0]), ['non', 'oui'])
    test_class.assertGreater(pred_proba[1][0][index_non], 0.5)
    test_class.assertLess(pred_proba[1][0][index_oui], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_oui], 0.5)
    test_class.assertLess(pred_proba[1][1][index_non], 0.5)
    # get_predict_position
    # position start at 1
    test_class.assertEqual(list(test_model.get_predict_position(['cdi à temps complet', 'vous disposez du permis', 'titi'], ['oui', 'oui', 'toto'])), [2, 1, -1])
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['non', 'oui'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5)  # Only 2 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=2)
    test_class.assertEqual([list(_) for _ in top_n], [['non', 'oui'], ['oui', 'non']])
    test_class.assertEqual([list(_) for _ in top_n_proba], [[probas[0][index_non], probas[0][index_oui]], [probas[1][index_oui], probas[1][index_non]]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['non', 'oui'])

    # Remove dir
    remove_dir(test_model.model_dir)


class Case2_MonoClassMonoLabel(unittest.TestCase):
    '''Class to test the mono-class / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for mono-class / mono-label case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py --overwrite -f mono_class_mono_label_train.csv mono_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                            tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                            gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                            multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                            multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #          filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')

    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                   filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=20,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGru(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru.ModelEmbeddingLstmGru(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGru failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Mono-class & Mono-label")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_mono_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test012_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=16, epochs=40, patience=20,
                                                                                                         max_sequence_length=60, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('ModelEmbeddingLstmStructuredAttention failed')

    def test13_Model_HuggingFace(self):
        '''Test of the model HuggingFace'''
        print('            ------------------ >     Test of the model HuggingFace     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'huggingface_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_huggingface.ModelHuggingFace(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            batch_size=16, epochs=2, patience=5,
                                                            transformer_name='Geotrend/distilbert-base-fr-cased',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_HuggingFace failed')

    def test14_Model_Aggregation(self):
        '''Test of the model Aggregation'''
        print('            ------------------ >     Test of the model Aggregation     /   Mono-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model with function majority_vote and list_models=[model, model, model]
            model_name = 'aggregation_mono_class_mono_label'
            model_name_svm1 = 'aggregation_mono_class_mono_label_svm1'
            model_name_svm2 = 'aggregation_mono_class_mono_label_svm2'
            model_name_gbt = 'aggregation_mono_class_mono_label_gbt'
            model_dir_svm1 = os.path.join(utils.get_models_path(), model_name_svm1, datetime.now().strftime(f"{model_name_svm1}_%Y_%m_%d-%H_%M_%S"))
            model_dir_svm2 = os.path.join(utils.get_models_path(), model_name_svm2, datetime.now().strftime(f"{model_name_svm2}_%Y_%m_%d-%H_%M_%S"))
            model_dir_gbt = os.path.join(utils.get_models_path(), model_name_gbt, datetime.now().strftime(f"{model_name_gbt}_%Y_%m_%d-%H_%M_%S"))
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            test_model = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_mono_label(self, test_model)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function majority_vote and list_models=[model_name, model_name, model_name]
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1)
            svm1.save()
            svm2 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2)
            svm2.save()
            gbt = model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)
            gbt.save()

            list_models = [os.path.split(model_dir_svm1)[-1], os.path.split(model_dir_svm2)[-1], os.path.split(model_dir_gbt)[-1]]
            model_name = 'aggregation_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))

            test_model_2 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_mono_label(self, test_model_2)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function majority_vote and list_models=[model_name, model, model]
            model_name = 'aggregation_mono_class_mono_label'
            model_name_svm1 = 'aggregation_mono_class_mono_label_svm1'
            model_dir_svm1 = os.path.join(utils.get_models_path(), model_name_svm1, datetime.now().strftime(f"{model_name_svm1}_%Y_%m_%d-%H_%M_%S"))
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1)
            svm1.save()
            list_models = [os.path.split(model_dir_svm1)[-1], model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_3 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            test_model_mono_class_mono_label(self, test_model_3)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function proba_argmax
            model_name = 'aggregation_mono_class_mono_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_4 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=True, aggregation_function='proba_argmax',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_4)
            test_model_mono_class_mono_label(self, test_model_4)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function given
            model_name = 'aggregation_mono_class_mono_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))

            # This function is a copy of majority_vote function
            def function_test(predictions: pd.Series, **kwargs) -> list:
                labels, counts = np.unique(predictions, return_counts=True)
                votes = [(label, count) for label, count in zip(labels, counts)]
                votes = sorted(votes, key=lambda x: x[1], reverse=True)
                possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
                return [prediction for prediction in predictions if prediction in possible_classes][0]

            test_model_5 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function=function_test,
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='mono_class_mono_label_train_preprocess_P1.csv', model=test_model_5)
            test_model_mono_class_mono_label(self, test_model_5)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

        except Exception:
            self.fail('testModel_Aggregation failed')


def test_model_mono_class_multi_label(test_class, test_model):
    '''Generic fonction to test a given model for mono-class/multi-labels'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Verify gabarit version
    with open(os.path.join(test_model.model_dir, 'configurations.json'), 'r') as f:
        configurations = json.load(f)
        test_class.assertTrue("gabarit_version" in configurations)
    # Try some functions
    index_col_1 = test_model.list_classes.index('y_col_1')
    index_col_2 = test_model.list_classes.index('y_col_2')
    pred_none = [0, 0]
    pred_col_1 = [0, 0]
    pred_col_1[index_col_1] = 1
    pred_col_2 = [0, 0]
    pred_col_2[index_col_2] = 1
    pred_all = [1, 1]
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual([list(_) for _ in preds], [pred_none, pred_col_1, pred_col_2, pred_all])
    # predict_proba
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertLess(probas[0][index_col_1], 0.5)
    test_class.assertLess(probas[0][index_col_2], 0.5)
    test_class.assertGreater(probas[1][index_col_1], 0.5)
    test_class.assertLess(probas[1][index_col_2], 0.5)
    test_class.assertLess(probas[2][index_col_1], 0.5)
    test_class.assertGreater(probas[2][index_col_2], 0.5)
    test_class.assertGreater(probas[3][index_col_1], 0.5)
    test_class.assertGreater(probas[3][index_col_2], 0.5)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'], return_proba=True)
    test_class.assertLess(probas2[0][index_col_1], 0.5)
    test_class.assertLess(probas2[0][index_col_2], 0.5)
    test_class.assertGreater(probas2[1][index_col_1], 0.5)
    test_class.assertLess(probas2[1][index_col_2], 0.5)
    test_class.assertLess(probas2[2][index_col_1], 0.5)
    test_class.assertGreater(probas2[2][index_col_2], 0.5)
    test_class.assertGreater(probas2[3][index_col_1], 0.5)
    test_class.assertGreater(probas2[3][index_col_2], 0.5)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual([list(_) for _ in pred_proba[0]], [pred_none, pred_col_1, pred_col_2, pred_all])
    test_class.assertLess(pred_proba[1][0][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][0][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][1][index_col_1], 0.5)
    test_class.assertLess(pred_proba[1][1][index_col_2], 0.5)
    test_class.assertLess(pred_proba[1][2][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][2][index_col_2], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_1], 0.5)
    test_class.assertGreater(pred_proba[1][3][index_col_2], 0.5)
    # get_predict_position
    # position start at 1
    with test_class.assertRaises(ValueError):
        test_model.get_predict_position(['toto', 'tata', 'toto', 'titi'], ['toto', 'tata', 'toto', 'titi'])  # Does not work with multi-labels
    # get_classes_from_proba
    test_class.assertEqual([list(_) for _ in test_model.get_classes_from_proba(probas)], [pred_none, pred_col_1, pred_col_2, pred_all])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=2)  # Does not work with multi-labels
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), [(), ('y_col_1',), ('y_col_2',), ('y_col_1', 'y_col_2')])

    # Remove dir
    remove_dir(test_model.model_dir)


class Case3_MonoClassMultiLabel(unittest.TestCase):
    '''Class to test the mono-class / multi-labels case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the mono-class / multi-labels case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f mono_class_multi_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py --overwrite -f mono_class_multi_label_train.csv mono_class_multi_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=True, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                            tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                            gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                            multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                            multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                              multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #          filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')

    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                     filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    # TODO : Make sure that this test passes everytime by making it more stable
    @unittest.skip("The test of the LSTM model with attention + Mono-class & Multi-labels is unstable, for now, we skip it !")
    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=40,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGru(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru.ModelEmbeddingLstmGru(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGru failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Mono-class & multi-labels")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=16, epochs=40, patience=40,
                                                                                    max_sequence_length=60, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=True, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=True, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=True, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_mono_class_multi_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=True, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            #           filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_mono_class_multi_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test12_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=16, epochs=40, patience=40,
                                                                                                         max_sequence_length=60, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
            filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmStructuredAttention failed')

    def test13_Model_HuggingFace(self):
        '''Test of the model HuggingFace'''
        print('            ------------------ >     Test of the model HuggingFace     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'huggingface_mono_class_multi_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_huggingface.ModelHuggingFace(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            batch_size=16, epochs=2, patience=5,
                                                            transformer_name='Geotrend/distilbert-base-fr-cased',
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
        except Exception:
            self.fail('testModel_HuggingFace failed')

    def test14_Model_Aggregation(self):
        '''Test of the model Aggregation'''
        print('            ------------------ >     Test of the model Aggregation     /   Mono-class & Multi-labels')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model function all_predictions and list_models=[model, model, model]
            model_name = 'aggregation_mono_class_multi_label'
            model_name_svm1 = 'aggregation_mono_class_multi_label_svm1'
            model_name_svm2 = 'aggregation_mono_class_multi_label_svm2'
            model_name_gbt = 'aggregation_mono_class_multi_label_gbt'
            model_dir_svm1 = os.path.join(utils.get_models_path(), model_name_svm1, datetime.now().strftime(f"{model_name_svm1}_%Y_%m_%d-%H_%M_%S"))
            model_dir_svm2 = os.path.join(utils.get_models_path(), model_name_svm2, datetime.now().strftime(f"{model_name_svm2}_%Y_%m_%d-%H_%M_%S"))
            model_dir_gbt = os.path.join(utils.get_models_path(), model_name_gbt, datetime.now().strftime(f"{model_name_gbt}_%Y_%m_%d-%H_%M_%S"))
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            list_models = [model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(multi_label=True, model_dir=model_dir_gbt)]
            test_model = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='all_predictions',
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model)
            test_model_mono_class_multi_label(self, test_model)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function all_predictions and list_models=[model_name, model_name, model_name]
            model_name = 'aggregation_mono_class_multi_label'
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1, multi_label=True)
            svm1.save()
            svm2 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2, multi_label=True)
            svm2.save()
            gbt = model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt, multi_label=True)
            gbt.save()

            list_models = [os.path.split(model_dir_svm1)[-1], os.path.split(model_dir_svm2)[-1], os.path.split(model_dir_gbt)[-1]]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_2 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='all_predictions',
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_mono_class_multi_label(self, test_model_2)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function all_predictions and list_models=[model_name, model, model]
            model_name = 'aggregation_mono_class_multi_label'
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1, multi_label=True)
            svm1.save()
            list_models = [os.path.split(model_dir_svm1)[-1], model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(multi_label=True, model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_3 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='all_predictions',
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_3)
            test_model_mono_class_multi_label(self, test_model_3)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function vote_labels
            model_name = 'aggregation_mono_class_multi_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(multi_label=True, model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_4 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='vote_labels',
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_4)
            test_model_mono_class_multi_label(self, test_model_4)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function given
            model_name = 'aggregation_mono_class_multi_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(multi_label=True, model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(multi_label=True, model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))

            # This function is a copy of all_predictions function
            def function_test(predictions: pd.Series, **kwargs) -> list:
                return np.sum(predictions, axis=0, dtype=bool).astype(int)

            test_model_5 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function=function_test,
                                                            multi_label=True, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='mono_class_multi_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col_1', 'y_col_2'],
                      filename_valid='mono_class_multi_label_train_preprocess_P1.csv', model=test_model_5)
            test_model_mono_class_multi_label(self, test_model_5)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

        except Exception:
            self.fail('testModel_Aggregation failed')


def test_model_multi_class_mono_label(test_class, test_model):
    '''Generic fonction to test a given model for multi-classes/mono-label'''

    # Check files exists
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, 'configurations.json')))
    test_class.assertTrue(os.path.exists(os.path.join(test_model.model_dir, f'{test_model.model_name}.pkl')))
    # Verify gabarit version
    with open(os.path.join(test_model.model_dir, 'configurations.json'), 'r') as f:
        configurations = json.load(f)
        test_class.assertTrue("gabarit_version" in configurations)
    # Try some functions
    index_none = test_model.list_classes.index('none')
    index_a = test_model.list_classes.index('a')
    index_b = test_model.list_classes.index('b')
    index_both = test_model.list_classes.index('both')
    pred_none = [0, 0, 0, 0]
    pred_none[index_none] = 1
    pred_a = [0, 0, 0, 0]
    pred_a[index_a] = 1
    pred_b = [0, 0, 0, 0]
    pred_b[index_b] = 1
    pred_both = [0, 0, 0, 0]
    pred_both[index_both] = 1
    # predict
    preds = test_model.predict(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(list(preds), ['none', 'a', 'b', 'both'])
    # predict_proba
    probas = test_model.predict_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(round(probas.sum(), 3), 4.)  # We round for deep learning models
    test_class.assertGreater(probas[0][index_none], 1/4)
    test_class.assertLess(probas[0][index_a], probas[0][index_none])
    test_class.assertLess(probas[0][index_b], probas[0][index_none])
    test_class.assertLess(probas[0][index_both], probas[0][index_none])
    test_class.assertLess(probas[1][index_none], probas[1][index_a])
    test_class.assertGreater(probas[1][index_a], 1/4)
    test_class.assertLess(probas[1][index_b], probas[1][index_a])
    test_class.assertLess(probas[1][index_both], probas[1][index_a])
    test_class.assertLess(probas[2][index_none], probas[2][index_b])
    test_class.assertLess(probas[2][index_a], probas[2][index_b])
    test_class.assertGreater(probas[2][index_b], 1/4)
    test_class.assertLess(probas[2][index_both], probas[2][index_b])
    test_class.assertLess(probas[3][index_none], probas[3][index_both])
    test_class.assertLess(probas[3][index_a], probas[3][index_both])
    test_class.assertLess(probas[3][index_b], probas[3][index_both])
    test_class.assertGreater(probas[3][index_both], 1/4)
    # predict w/ return_proba=True
    probas2 = test_model.predict(['cdi à temps complet', 'vous disposez du permis',
                                  'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'],
                                  return_proba=True)
    test_class.assertEqual(round(probas2.sum(), 3), 4.)  # We round for deep learning models
    test_class.assertGreater(probas2[0][index_none], 1/4)
    test_class.assertLess(probas2[0][index_a], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_b], probas2[0][index_none])
    test_class.assertLess(probas2[0][index_both], probas2[0][index_none])
    test_class.assertLess(probas2[1][index_none], probas2[1][index_a])
    test_class.assertGreater(probas2[1][index_a], 1/4)
    test_class.assertLess(probas2[1][index_b], probas2[1][index_a])
    test_class.assertLess(probas2[1][index_both], probas2[1][index_a])
    test_class.assertLess(probas2[2][index_none], probas2[2][index_b])
    test_class.assertLess(probas2[2][index_a], probas2[2][index_b])
    test_class.assertGreater(probas2[2][index_b], 1/4)
    test_class.assertLess(probas2[2][index_both], probas2[2][index_b])
    test_class.assertLess(probas2[3][index_none], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_a], probas2[3][index_both])
    test_class.assertLess(probas2[3][index_b], probas2[3][index_both])
    test_class.assertGreater(probas2[3][index_both], 1/4)
    # predict_with_proba
    pred_proba = test_model.predict_with_proba(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire'])
    test_class.assertEqual(list(pred_proba[0]), ['none', 'a', 'b', 'both'])
    test_class.assertEqual(round(pred_proba[1].sum(), 3), 4.)  # We round for deep learning models
    test_class.assertGreater(pred_proba[1][0][index_none], 1/4)
    test_class.assertLess(pred_proba[1][0][index_a], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_b], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][0][index_both], pred_proba[1][0][index_none])
    test_class.assertLess(pred_proba[1][1][index_none], pred_proba[1][1][index_a])
    test_class.assertGreater(pred_proba[1][1][index_a], 1/4)
    test_class.assertLess(pred_proba[1][1][index_b], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][1][index_both], pred_proba[1][1][index_a])
    test_class.assertLess(pred_proba[1][2][index_none], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][2][index_a], pred_proba[1][2][index_b])
    test_class.assertGreater(pred_proba[1][2][index_b], 1/4)
    test_class.assertLess(pred_proba[1][2][index_both], pred_proba[1][2][index_b])
    test_class.assertLess(pred_proba[1][3][index_none], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_a], pred_proba[1][3][index_both])
    test_class.assertLess(pred_proba[1][3][index_b], pred_proba[1][3][index_both])
    test_class.assertGreater(pred_proba[1][3][index_both], 1/4)
    # get_predict_position
    # position start at 1
    predict_pos = test_model.get_predict_position(['cdi à temps complet', 'vous disposez du permis', 'le véhicule est nécessaire', 'vous disposez du permis et le véhicule est nécessaire', 'titi'], ['none', 'a', 'a', 'both', 'toto'])
    test_class.assertEqual(list(predict_pos[[0, 1, 3, 4]]), [1, 1, 1, -1])
    test_class.assertGreater(predict_pos[2], 1)
    # get_classes_from_proba
    test_class.assertEqual(list(test_model.get_classes_from_proba(probas)), ['none', 'a', 'b', 'both'])
    # get_top_n_from_proba
    with test_class.assertRaises(ValueError):
        test_model.get_top_n_from_proba(probas, n=5)  # Only 4 classes in our model
    top_n, top_n_proba = test_model.get_top_n_from_proba(probas, n=4)
    test_class.assertEqual([_[0] for _ in top_n], ['none', 'a', 'b', 'both'])
    test_class.assertEqual(sorted(top_n[0]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[1]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[2]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual(sorted(top_n[3]), sorted(['none', 'a', 'b', 'both']))
    test_class.assertEqual([_[0] for _ in top_n_proba], [probas[0][index_none], probas[1][index_a], probas[2][index_b], probas[3][index_both]])
    # inverse_transform
    test_class.assertEqual(list(test_model.inverse_transform(preds)), ['none', 'a', 'b', 'both'])

    # Remove dir
    remove_dir(test_model.model_dir)


class Case4_MultiClassMonoLabel(unittest.TestCase):
    '''Class to test the multi-classes / mono-label case'''

    def test01_PrepareDatasets(self):
        '''Prepares the datasets'''
        print("Prepares the datasets for the multi-classes / mono-label case")

        # Gen. datasets
        split_train_valid_test = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/utils/0_split_train_valid_test.py --overwrite -f multi_class_mono_label.csv --split_type random --perc_train 0.6 --perc_valid 0.2 --perc_test 0.2 --x_col x_col --y_col y_col --seed 42"
        preprocessing = f"{activate_venv}python {full_path_lib}/test_template_nlp-scripts/1_preprocess_data.py --overwrite -f multi_class_mono_label_train.csv multi_class_mono_label_valid.csv -p preprocess_P1 --input_col x_col"
        self.assertEqual(subprocess.run(split_train_valid_test, shell=True).returncode, 0)
        self.assertEqual(subprocess.run(preprocessing, shell=True).returncode, 0)

    def test02_Model_TfidfSvm(self):
        '''Test of the model TF-IDF/SVM'''
        print('            ------------------ >     Test of the model TF-IDF/SVM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_svm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       svc_params={'C': 1.0, 'max_iter': 10000},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_svm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         svc_params={'C': 1.0, 'max_iter': 10000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_svm_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_svm.ModelTfidfSvm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              svc_params={'C': 1.0, 'max_iter': 10000},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSvm failed')

    def test03_Model_TfidfGbt(self):
        '''Test of the model TF-IDF/GBT'''
        print('            ------------------ >     Test of the model TF-IDF/GBT     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_gbt_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                       tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                       gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                       multi_label=False, model_name=model_name, model_dir=model_dir,
                                                       multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_gbt_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_gbt_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_gbt.ModelTfidfGbt(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                              tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                              gbt_params={'learning_rate': 0.1, 'n_estimators': 5, 'max_depth': 5, 'subsample': 1.0, 'max_features': 'auto'},
            #                                              multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                              multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfGbt failed')

    def test04_Model_TfidfLgbm(self):
        '''Test of the model TF-IDF/LGBM'''
        print('            ------------------ >     Test of the model TF-IDF/LGBM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_lgbm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_lgbm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            # We overfit on purpose !
            test_model_2 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_lgbm_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # # We overfit on purpose !
            # test_model_3 = model_tfidf_lgbm.ModelTfidfLgbm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                lgbm_params={'learning_rate': 0.1, 'n_estimators': 50, 'max_depth': 2000, 'subsample': 1.0, 'num_leaves': 12070, 'min_data_in_leaf': 2},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #          filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfLgbm failed')

    def test05_Model_TfidfDense(self):
        '''Test of the model TF-IDF/Dense'''
        print('            ------------------ >     Test of the model TF-IDF/Dense     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'tfidf_dense_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_dense.ModelTfidfDense(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           batch_size=16, epochs=20, patience=20,
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                     filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_TfidfDense failed')

    def test06_Model_EmbeddingLstm(self):
        '''Test of the model Embedding/LSTM'''
        print('            ------------------ >     Test of the model Embedding/LSTM     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                   filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstm failed')

    def test07_Model_EmbeddingLstmAttention(self):
        '''Test of the model Embedding/LSTM/Attention'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_attention.ModelEmbeddingLstmAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                    batch_size=64, epochs=40, patience=20,
                                                                                    max_sequence_length=10, max_words=100000,
                                                                                    embedding_name="custom.300.pkl",
                                                                                    multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmAttention failed')

    def test08_Model_EmbeddingLstmGru(self):
        '''Test of the model Embedding/LSTM/GRU'''
        print('            ------------------ >     Test of the model Embedding/LSTM/GRU     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_gru_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_gru.ModelEmbeddingLstmGru(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                               batch_size=16, epochs=20, patience=20,
                                                                               max_sequence_length=60, max_words=100000,
                                                                               embedding_name="custom.300.pkl",
                                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmGru failed')

    def test09_Model_EmbeddingCnn(self):
        '''Test of the model Embedding/CNN'''
        print('            ------------------ >     Test of the model Embedding/CNN     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_cnn_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_cnn.ModelEmbeddingCnn(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                               batch_size=16, epochs=20, patience=20,
                                                               max_sequence_length=60, max_words=100000,
                                                               embedding_name="custom.300.pkl",
                                                               multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingCnn failed')

    def test10_Model_Keras_continue_training(self):
        '''Test continuing a training for a keras model'''
        print("            ------------------ >     Test continuing a training for a keras model     /   Multi-class & Mono-label")

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_mono_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm.ModelEmbeddingLstm(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                 batch_size=16, epochs=20, patience=20,
                                                                 max_sequence_length=60, max_words=100000,
                                                                 embedding_name="custom.300.pkl",
                                                                 multi_label=False, model_name=model_name, model_dir=model_dir)
            # Run a first training
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertEqual(model_dir, test_model.model_dir)

            # Retrieve model & run a second training
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            self.assertNotEqual(model_dir, test_model.model_dir)

            # Test second trained model
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_Keras_continue_training failed')

    def test11_Model_TfidfSgdc(self):
        '''Test of the model TF-IDF/SGDClassifier'''
        print('            ------------------ >     Test of the model TF-IDF/SGDClassifier     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model
            model_name = 'tfidf_sgdc_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                         tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                         sgdc_params={'loss': 'hinge', 'max_iter': 1000},
                                                         multi_label=False, model_name=model_name, model_dir=model_dir,
                                                         multiclass_strategy=None)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)

            # Set second model
            model_name = 'tfidf_sgdc_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model_2 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                           tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
                                                           sgdc_params={'loss': 'log', 'max_iter': 1000},
                                                           multi_label=False, model_name=model_name, model_dir=model_dir,
                                                           multiclass_strategy='ovr')
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)

            # Set third model
            # 'ovo' non stable
            # model_name = 'tfidf_sgdc_multi_class_mono_label'
            # model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            # os.makedirs(model_dir)
            # test_model_3 = model_tfidf_sgdc.ModelTfidfSgdc(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
            #                                                tfidf_params={'analyzer': 'word', 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.25, 'max_features': 100000},
            #                                                sgdc_params={'loss': 'log', 'max_iter': 1000},
            #                                                multi_label=False, model_name=model_name, model_dir=model_dir,
            #                                                multiclass_strategy='ovo')
            # # Test it
            # test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            #           filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            # test_model_multi_class_mono_label(self, test_model_3)
        except Exception:
            self.fail('testModel_TfidfSgdc failed')

    def test12_Model_EmbeddingLstmStructuredAttention(self):
        '''Test of the model Embedding/LSTM/Attention + explainable'''
        print('            ------------------ >     Test of the model Embedding/LSTM/Attention + explainable     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'embedding_lstm_attention_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_embedding_lstm_structured_attention.ModelEmbeddingLstmStructuredAttention(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                                                                         batch_size=64, epochs=40, patience=20,
                                                                                                         max_sequence_length=10, max_words=100000,
                                                                                                         embedding_name="custom.300.pkl",
                                                                                                         multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
            filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_EmbeddingLstmStructuredAttention failed')

    def test13_Model_HuggingFace(self):
        '''Test of the model HuggingFace'''
        print('            ------------------ >     Test of the model HuggingFace     /   Multi-classes & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)
            # Set model
            model_name = 'huggingface_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            os.makedirs(model_dir)
            test_model = model_huggingface.ModelHuggingFace(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            batch_size=16, epochs=2, patience=5,
                                                            transformer_name='Geotrend/distilbert-base-fr-cased',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
        except Exception:
            self.fail('testModel_HuggingFace failed')

    def test14_Model_Aggregation(self):
        '''Test of the model Aggregation'''
        print('            ------------------ >     Test of the model Aggregation     /   Multi-class & Mono-label')

        try:
            # Load training file
            spec = importlib.util.spec_from_file_location("test", f'{full_path_lib}/test_template_nlp-scripts/2_training.py')
            test = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test)

            # Set model with function majority_vote and list_models=[model, model, model]
            model_name = 'aggregation_multi_class_mono_label'
            model_name_svm1 = 'aggregation_multi_class_mono_label_svm1'
            model_name_svm2 = 'aggregation_multi_class_mono_label_svm2'
            model_name_gbt = 'aggregation_multi_class_mono_label_gbt'
            model_dir_svm1 = os.path.join(utils.get_models_path(), model_name_svm1, datetime.now().strftime(f"{model_name_svm1}_%Y_%m_%d-%H_%M_%S"))
            model_dir_svm2 = os.path.join(utils.get_models_path(), model_name_svm2, datetime.now().strftime(f"{model_name_svm2}_%Y_%m_%d-%H_%M_%S"))
            model_dir_gbt = os.path.join(utils.get_models_path(), model_name_gbt, datetime.now().strftime(f"{model_name_gbt}_%Y_%m_%d-%H_%M_%S"))
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            test_model = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model)
            test_model_multi_class_mono_label(self, test_model)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function majority_vote and list_models=[model_name, model_name, model_name]
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1)
            svm1.save()
            svm2 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2)
            svm2.save()
            gbt = model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)
            gbt.save()

            list_models = [os.path.split(model_dir_svm1)[-1], os.path.split(model_dir_svm2)[-1], os.path.split(model_dir_gbt)[-1]]
            model_name = 'aggregation_multi_class_mono_label'
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_2 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_2)
            test_model_multi_class_mono_label(self, test_model_2)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function majority_vote and list_models=[model_name, model, model]
            model_name = 'aggregation_multi_class_mono_label'
            model_name_svm1 = 'aggregation_multi_class_mono_label_svm1'
            model_dir_svm1 = os.path.join(utils.get_models_path(), model_name_svm1, datetime.now().strftime(f"{model_name_svm1}_%Y_%m_%d-%H_%M_%S"))
            svm1 = model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1)
            svm1.save()
            list_models = [os.path.split(model_dir_svm1)[-1], model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_3 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function='majority_vote',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_3)
            test_model_multi_class_mono_label(self, test_model_3)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function proba_argmax
            model_name = 'aggregation_multi_class_mono_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))
            test_model_4 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=True, aggregation_function='proba_argmax',
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_4)
            test_model_multi_class_mono_label(self, test_model_4)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

            # Set model with function given
            model_name = 'aggregation_multi_class_mono_label'
            list_models = [model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm1), model_tfidf_svm.ModelTfidfSvm(model_dir=model_dir_svm2), model_tfidf_gbt.ModelTfidfGbt(model_dir=model_dir_gbt)]
            model_dir = os.path.join(utils.get_models_path(), model_name, datetime.now().strftime(f"{model_name}_%Y_%m_%d-%H_%M_%S"))

            # This function is a copy of majority_vote function
            def function_test(predictions: pd.Series, **kwargs) -> list:
                labels, counts = np.unique(predictions, return_counts=True)
                votes = [(label, count) for label, count in zip(labels, counts)]
                votes = sorted(votes, key=lambda x: x[1], reverse=True)
                possible_classes = {vote[0] for vote in votes if vote[1]==votes[0][1]}
                return [prediction for prediction in predictions if prediction in possible_classes][0]

            test_model_5 = model_aggregation.ModelAggregation(x_col='preprocessed_text', y_col='y_col', level_save="HIGH",
                                                            list_models=list_models, using_proba=False, aggregation_function=function_test,
                                                            multi_label=False, model_name=model_name, model_dir=model_dir)
            # Test it
            test.main(filename='multi_class_mono_label_train_preprocess_P1.csv', x_col='preprocessed_text', y_col=['y_col'],
                      filename_valid='multi_class_mono_label_train_preprocess_P1.csv', model=test_model_5)
            test_model_multi_class_mono_label(self, test_model_5)
            remove_dir(model_dir)
            remove_dir(model_dir_svm1)
            remove_dir(model_dir_svm2)
            remove_dir(model_dir_gbt)

        except Exception:
            self.fail('testModel_Aggregation failed')


if __name__ == '__main__':
    # Change directory to script directory parent
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    parentname = str(Path(dname).parent)
    os.chdir(parentname)
    # Manage venv
    full_path_lib = os.path.abspath(os.path.join(os.getcwd(), 'test_template_nlp'))
    if os.name == 'nt':
        is_windows = True
        # Windows: activate the virtual environment & continue with the other processes
        activate_venv = f"cd {full_path_lib}/venv_test_template_nlp/Scripts & activate & "
    else:
        is_windows = False
        # UNIX : We can't use "source" so we directly call python/pip from the bin of the virtual environment
        activate_venv = f"{full_path_lib}/venv_test_template_nlp/bin/"
    # Start tests
    unittest.main()
