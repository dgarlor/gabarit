#!/usr/bin/env python3

## Generates fairness metrics
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
#


import os
import json
import logging
import argparse
import pandas as pd
import fairlens as fl
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from typing import List, Union, Tuple

from {{package_name}} import utils

# Get logger
logger = logging.getLogger("{{package_name}}.0_fairness_metrics.py")

def find_bias(distribution_score:pd.DataFrame, 
              min_proportion:float, 
              min_distance:float, 
              max_p_value:float) -> pd.DataFrame:
    '''Gets the biased groups when given a distribution_score dataframe. Actually just filters it 
    on the Proportion, Distance and P-Value columns. Also adds a column number_of_attributes containing
    the number of attributes defining the group.

    Args:
        distribution_score (pd.DataFrame) : A dataframe obtained by the method distribution_score of a fl.FairnessScorer
        min_proportion (float) : The minimal proportion of a subgroup to be considered as biased
        min_distance (float) : The minimal distance (Kolmogorov-Smirnov) of a subgroup to be considered as biased
        max_p_value (float) : The maximal p-value (Kolmogorov-Smirnov) of a subgroup to be considered as biased
    Returns:
        pd.DataFrame : the biased groups
    
    '''
    biased_groups = distribution_score.copy()
    biased_groups = biased_groups[biased_groups['Proportion']>=min_proportion]
    biased_groups = biased_groups[abs(biased_groups['Distance'])>=min_distance]
    biased_groups = biased_groups[biased_groups['P-Value']<=max_p_value]
    biased_groups = biased_groups.sort_values('Distance', ascending=False)
    biased_groups['number_of_attributes'] = biased_groups['Group'].apply(lambda x:x.count(',')+1)
    return biased_groups


def get_fairlens_metrics(data:pd.DataFrame, col_target: str, sensitive_cols: List[str], output_path: str, sep: str, encoding:str):
    '''Instanciates a fl.FairnessScorer and then writes three files in output_path:
        data_distributions.png : The distribution with respect to the target for each sensitive attribute's subgroup
        data_distribution_score.csv : A table containing the Kolmogorov-Smirnov statistics for each subgroups
        data_biased_groups.csv : A sub-table of the one above containing the biased groups only

    Args:
        data (pd.DataFrame) : The data we want to explore
        col_target (str) : The name of the target column in data
        sensitive_cols (List[str]) : The list of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)
        output_path (str) : The path to the folder where the files will be saved
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    '''
    # Instanciates the fl.FairnessScorer
    fl_scorer = fl.FairnessScorer(data[sensitive_cols+[col_target]], col_target, sensitive_attrs = sensitive_cols)
    # Plots and saves the distributions
    logger.info(f"Calculates distributions graphs")
    fl_scorer.plot_distributions(normalize=True)
    plt.savefig(os.path.join(output_path, 'data_distributions.png'))
    # Calculates and saves the Kolmogorov-Smirnov distances
    logger.info(f"Calculates Kolmogorov-Smirnov distances for each subgroup")
    distribution_score = fl_scorer.distribution_score(p_value=True)
    distribution_score.to_csv(os.path.join(output_path, 'data_distribution_score.csv'), sep=sep)
    # Filters the distribution_score to keep only biased groups and saves them
    biased_groups = find_bias(distribution_score=distribution_score, 
                              min_proportion=0.01, 
                              min_distance=0.05,
                              max_p_value=0.0001)
    biased_groups.to_csv(os.path.join(output_path, 'data_biased_groups.csv'), sep=sep, encoding=encoding)


def get_fairlearn_metrics(data:pd.DataFrame, col_target: str, sensitive_cols: List[str], output_path: str, sep: str, encoding:str):
    pass


def bin_continuous_col(data, col):
    pass


def bin_datetime_col(data, col):
    pass



def bin_continuous_sensitive_cols(data:pd.DataFrame, col_target:str, sensitive_cols:List[str]) -> pd.DataFrame:
    fl_scorer = fl.FairnessScorer(data[sensitive_cols+[col_target]], col_target, sensitive_attrs = sensitive_cols)
    for attr, attr_dist_type in zip([fl_scorer.sensitive_attrs, fl_scorer.sensitive_distr_types]):
        if attr_dist_type.value=='continuous':
            bin_continuous_col(data, attr)
        if attr_dist_type.value=='datetime':
            bin_datetime_col(data, attr)

def main(filename:str, col_target:str, sensitive_cols:List[str], output_folder:str, col_pred:Union[None, str]=None, 
         sep: str = '{{default_sep}}', encoding: str = '{{default_encoding}}'):
    '''

    Args:
        
    Kwargs:
        col_pred
        sep (str): Separator to use with the .csv files
        encoding (str): Encoding to use with the .csv files
    '''
    logger.info(f"Loading data")
    data_path = utils.get_data_path()
    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    data, metadata = utils.read_csv(os.path.join(data_path, filename), sep=sep, encoding=encoding)
    # Bins continuous sensitive attributes
    bin_continuous_sensitive_cols(data, col_target, sensitive_cols)
    # Gets fairlens metrics ie metrics on fairness of subgroups with respect to the target
    logger.info(f"Gets fairlens metrics")
    get_fairlens_metrics(data=data, col_target=col_target, sensitive_cols=sensitive_cols, output_path=output_path, sep=sep, encoding=encoding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('fairness_metrics', description=(
            "Calculates various metrics for fairness."
        ),
    )
    parser.add_argument('-f', '--filename', required=True, help="Path to the dataset (actually paths relative to {{package_name}}-data)")
    parser.add_argument('-t', '--target', required=True, help="The name of the column containing the target")
    parser.add_argument('-s', '--sensitive_cols', required=True, nargs='+', help="The names of the columns containing sensitive attributes (eg. sex, age, ethnicity,...)")
    parser.add_argument('-o', '--output_folder', required=True, help="The name of the output folder")
    parser.add_argument('-p', '--col_pred', default=None, help="The column containing the predictions of a model")
    parser.add_argument('--sep', default='{{default_sep}}', help="Separator to use with the .csv files.")
    parser.add_argument('--encoding', default='{{default_encoding}}', help="Encoding to use with the .csv files.")
    args = parser.parse_args()
    main(filename=args.filename, col_target=args.target, sensitive_cols=args.sensitive_cols, 
         col_pred=args.col_pred, output_folder=args.output_folder, sep=args.sep, encoding=args.encoding)
