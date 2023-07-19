
import os
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import dask.array as da
from dask_ml.wrappers import ParallelPostFit

from oc_cidds_run import DaskOCSVM  # dont remove, needed for joblib model loading
from xai.util import tabular_reference_points
from data.cidds.util import get_cols_and_dtypes, get_column_mapping, get_summed_columns


def xai_to_categorical(df, cat_encoding='onehot', num_encoding='quantized'):
    # sum all encoded scores to single categorical values for each column
    categories = get_column_mapping(cat_encoding=cat_encoding, num_encoding=num_encoding, as_int=True)
    category_names = get_summed_columns()

    data = df.values
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    data_cat = pd.DataFrame(data_cat, columns=category_names, index=df.index)
    return data_cat


def get_expl_scores(explanation, gold_standard, score_type='auc_roc'):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    # Explanation values for each feature treated as likelihood of anomalous feature
    #  -aggregated to feature-scores over all feature assignments
    #  -flattened to match shape of y_true
    #  -inverted, so higher score means more anomalous
    explanation = xai_to_categorical(explanation)
    scores = []
    for i, row in explanation.iterrows():
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.iloc[i], y_score=row))
        elif score_type == 'auc_pr':
            scores.append(average_precision_score(y_true=gold_standard.iloc[i], y_score=row))
        elif score_type == 'pearson_corr':
            scores.append(pearsonr(x=gold_standard.iloc[i], y=row))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.iloc[i].values.reshape(1, -1), row.values.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   model,
                   gold_standard_path,
                   expl_path,
                   xai_type,
                   out_path):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(expl_path, header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    if model in ['IF', 'OCSVM']:
        expl = -1 * expl
    # Load gold standard explanations and convert to pd.Series containing
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
    gold_expl = (gold_expl == 'X')

    assert expl.shape[0] == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {expl.shape[0]}"

    # watch out for naming inconsistency! The dataset=data that get_expl_scores gets is an ERPDataset instance!
    roc_mean, roc_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='auc_roc')
    cos_mean, cos_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='cosine_sim')
    pearson_mean, pearson_std = get_expl_scores(explanation=expl,
                                                gold_standard=gold_expl,
                                                score_type='pearson_corr')

    out_dict = {'xai': xai_type,
                'variant': background,
                f'ROC': roc_mean,
                f'ROC-std': roc_std,
                f'Cos': cos_mean,
                f'Cos-std': cos_std,
                f'Pearson': pearson_mean,
                f'Pearson-std': pearson_std}
    [print(key + ':', val) for key, val in out_dict.items()]

    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()
        out_df = out_df.append(out_dict, ignore_index=True)
        out_df.to_csv(out_path, index=False)
    return out_dict


def explain_anomalies(compare_with_gold_standard,
                      expl_folder,
                      xai_type='shap',
                      model='AE',
                      background='zeros',
                      out_path=None,
                      **kwargs):
    """
    :param train_path:      Str path to train dataset
    :param test_path:       Str path to test dataset
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load, one of ['AE', 'OCSVM', 'IF']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """

    print('Loading data...')
    cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized')
    X_expl = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / f'anomalies_rand.csv',
                         index_col=None, usecols=cols + ['attackType'], header=0, dtype={'attackType': str, **dtypes})
    y_test = X_expl.pop('attackType')
    if background in ['mean', 'kmeans', 'NN']:
        X_train = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / 'train.csv.gz',
                              index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')
        X_train = X_train.sample(frac=0.001, random_state=42)  # sample normal data for kmeans and NN background
    else:
        X_train = pd.DataFrame(np.empty(X_expl.shape), columns=X_expl.columns, index=X_expl.index)

    print('Loading detector...')
    if model == 'AE':
        from anomaly_detection.autoencoder_torch import Autoencoder
        params = {'cpus': 8, 'n_layers': 3, 'n_bottleneck': 32, 'epochs': 10, 'batch_size': 2048, 'verbose': 2,
                  'learning_rate': 0.01, 'n_inputs': 146}
        detector = Autoencoder(**params)
        detector = detector.load('./outputs/models/cidds/cidds-ae-16_best.pt')
        detector.to('cpu')
    elif model == 'IF':
        import joblib
        detector = joblib.load('./outputs/models/cidds/cidds-if-41_best.pkl')
    elif model == 'OCSVM':
        import joblib
        detector = joblib.load('./outputs/models/cidds/cidds-oc-12_best.pkl')
    else:
        raise ValueError(f"Model {model} not supported!")

    # Generating explanations
    if not os.path.exists(os.path.join(expl_folder, f'{model}_shap_{background}.csv')):
        print("Generating explanations...")
        out_template = os.path.join(expl_folder, f'{model}_{{}}_{background}.csv')

        if xai_type == 'shap':
            import xai.xai_shap

            if isinstance(detector, ParallelPostFit):  # trick for multiprocessing single core algorithms with dask
                def predict_fn(X):
                    data = da.from_array(X, chunks=(100, -1))
                    return detector.predict(data).compute()
            else:
                predict_fn = detector.score_samples

            if background in ['zeros', 'mean', 'NN', 'kmeans']:
                if model == 'AE':
                    ref_predict_fn = functools.partial(detector.score_samples, output_to_numpy=False)
                else:
                    ref_predict_fn = predict_fn

                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            predict_fn=ref_predict_fn)
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_template.format(xai_type))

        elif xai_type == 'reconstruction':
            recon = detector.reconstruct(x=X_expl)
            error = (recon - X_expl)**2
            expl = pd.DataFrame(error, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_template.format(xai_type))

        elif xai_type == 'uniform_noise':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_template.format(xai_type))

        elif xai_type == 'uniform_noise_times_input':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl = expl * X_expl
            expl.to_csv(out_template.format(xai_type))

        else:
            raise ValueError(f'Unknown xai_type: {xai_type}')

    if compare_with_gold_standard:
        print('Evaluating explanations...')
        out_dict = evaluate_expls(background=background,
                                  expl_path=f'./outputs/explanation/cidds/{model}_{xai_type}_{background}.csv',
                                  gold_standard_path=f'data/cidds/data_raw/anomalies_rand_expl.csv',
                                  xai_type=xai_type,
                                  model=model,
                                  out_path=out_path)
        return out_dict


if __name__ == '__main__':

    backgrounds = ['kmeans']  # ['zeros', 'mean', 'kmeans', 'NN', 'optimized']
    model = 'AE'  # ['AE', 'IF', 'OCSVM']
    xai_type = 'shap'  # ['shap', 'uniform_noise', 'uniform_noise_times_input', 'reconstruction']

    compare_with_gold_standard = True
    add_to_summary = False

    expl_folder = './outputs/explanation/cidds/asdf'
    out_path = './outputs/explanation/cidds_summary.csv' if add_to_summary else None

    for background in backgrounds:
        explain_anomalies(compare_with_gold_standard=compare_with_gold_standard,
                          expl_folder=expl_folder,
                          xai_type=xai_type,
                          model=model,
                          background=background,
                          out_path=out_path)
