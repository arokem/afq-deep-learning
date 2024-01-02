import os
import os.path as op
from tempfile import mkdtemp
import glob
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from afqinsight.pipeline import make_base_afq_pipeline

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from afqinsight.datasets import AFQDataset
from afqinsight.augmentation import jitter, time_warp, scaling, magnitude_warp, window_warp
import tempfile
from sklearn.impute import SimpleImputer
from neurocombat_sklearn import CombatModel
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.utils import shuffle, resample
import pandas as pd
import pydra

import afqinsight.nn.tf_models as nn
import matplotlib.pyplot as plt
from afqinsight.nn.tf_models import cnn_lenet, mlp4, cnn_vgg, lstm1v0, lstm1, lstm2, blstm1, blstm2, lstm_fcn, cnn_resnet

import pickle


AUG_SCALING = 1/10
data_path = "/gscratch/escience/arokem/hbn/"


def load_data_notf():
    afq_dataset = AFQDataset.from_files(
        fn_nodes=f"{data_path}combined_tract_profiles.csv",
        fn_subjects=f"{data_path}participants_updated_id.csv",
        dwi_metrics=["dki_fa", "dki_md", "dki_mk"],
        index_col="subject_id",
        target_cols=["age", "dl_qc_score", "scan_site_id"],
        label_encode_cols=["scan_site_id"]
    )
    afq_dataset.drop_target_na()
    qc = afq_dataset.y[:, 1]
    y = afq_dataset.y[:, 0][qc>0]
    site = afq_dataset.y[:, 2][qc>0]
    X = afq_dataset.X[qc>0]
    return X, y, site


def load_data():
    afq_dataset = AFQDataset.from_files(
        fn_nodes=f"{data_path}combined_tract_profiles.csv",
        fn_subjects=f"{data_path}participants_updated_id.csv",
        dwi_metrics=["dki_fa", "dki_md", "dki_mk"],
        index_col="subject_id",
        target_cols=["age", "dl_qc_score", "scan_site_id"],
        label_encode_cols=["scan_site_id"]
    )
    afq_dataset.drop_target_na()
    full_dataset = list(afq_dataset.as_tensorflow_dataset().as_numpy_iterator())
    X = np.concatenate([xx[0][None] for xx in full_dataset], 0)
    y = np.array([yy[1][0] for yy in full_dataset])
    qc = np.array([yy[1][1] for yy in full_dataset])
    site = np.array([yy[1][2] for yy in full_dataset])
    X = X[qc>0]
    y = y[qc>0]
    site = site[qc>0]
    return X, y, site


def tf_aug(X_in, scaler=AUG_SCALING):
    if AUG_SCALING is None:
        raise ValueError("Need to over-write the value of AUG_SCALER")
    X_out = np.zeros_like(X_in)
    for channel in range(X_in.shape[-1]):
        this_X = X_in[..., channel][np.newaxis, ..., np.newaxis]
        scale = np.abs(np.max(this_X) - np.min(this_X)) * scaler
        this_X = jitter(this_X, sigma=scale)
        this_X = scaling(this_X, sigma=scale)
        this_X = time_warp(this_X, sigma=scale)
        # this_X = window_warp(this_X, window_ratio=scale)
        X_out[..., channel] = this_X[0, ..., 0]
    return X_out


def augment_this(X_in, y_in):
    X_out = tf.numpy_function(tf_aug, [X_in], tf.float32)
    return X_out, y_in


def model_fit(model_func, X_train, y_train, lr, batch_size=32, n_epochs=1000,
              augment=None):
    # Split into train and validation:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train.astype(np.float32), y_train.astype(np.float32)))
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val.astype(np.float32), y_val.astype(np.float32)))

    model = model_func(input_shape=X_train.shape[1:],
                       n_classes=1,
                       output_activation=None,
                       verbose=True)

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['mean_squared_error',
                           tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                           'mean_absolute_error'],
                 )

    # ModelCheckpoint
    ckpt_filepath = tempfile.NamedTemporaryFile().name + '.h5'
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath = ckpt_filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        )

    # EarlyStopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        mode="min",
        patience=100)

    # ReduceLROnPlateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=20,
        verbose=1)

    callbacks = [early_stopping, ckpt, reduce_lr]
    if augment is not None:
        train_dataset = train_dataset.map(augment)
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    history = model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset,
                        callbacks=callbacks, use_multiprocessing=True, verbose=2)
    model.load_weights(ckpt_filepath)
    return model


def fit_and_eval(model, model_dict, X, y, site, random_state, batch_size=32,
                 n_epochs=1000, augment=None, train_size=None):

    model_func = model_dict[model]["model"]
    lr = model_dict[model]["lr"]
    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(X, y, site, test_size=0.2, random_state=random_state)
    imputer = SimpleImputer(strategy="median")
    # If train_size is set, select train_size subjects to be the training data:
    if train_size is not None:
        X_train, y_train, site_train = resample(X_train, y_train, site_train, replace=False, n_samples=train_size, random_state=random_state)

    # Impute train and test separately:
    X_train = np.concatenate([imputer.fit_transform(X_train[..., ii])[:, :, None] for ii in range(X_train.shape[-1])], -1)
    X_test = np.concatenate([imputer.fit_transform(X_test[..., ii])[:, :, None] for ii in range(X_test.shape[-1])], -1)
    # Combat
    X_train = np.concatenate([CombatModel().fit_transform(X_train[..., ii], site_train[:, None], None, None)[:, :, None] for ii in range(X_train.shape[-1])], -1)
    X_test = np.concatenate([CombatModel().fit_transform(X_test[..., ii], site_test[:, None], None, None)[:, :, None] for ii in range(X_test.shape[-1])], -1)

    trained = model_fit(model_func, X_train, y_train, lr,
                        batch_size=batch_size, n_epochs=n_epochs,
                        augment=augment)
    metric = []
    value = []

    y_pred = trained.predict(X_test)
    metric.append("mae")
    value.append(mean_absolute_error(y_test, y_pred))
    metric.append("mad")
    value.append(median_absolute_error(y_test, y_pred))
    metric.append("r2")
    value.append(r2_score(y_test, y_pred))

    result = {'Model': [model] * len(metric),
              'Metric': metric,
              'Value': value}

    return pd.DataFrame(result), pd.DataFrame(dict(y_pred=y_pred.squeeze(), y_test=y_test))


def fit_and_eval_notf(model, X, y, site, random_state, train_size=None):

    X_train, X_test, y_train, y_test, site_train, site_test = train_test_split(
        X, y, site, test_size=0.2, random_state=random_state)
    imputer = SimpleImputer(strategy="median")
    # If train_size is set, select train_size subjects to be the training data:
    if train_size is not None:
        X_train, y_train, site_train = resample(X_train, y_train, site_train,
                                                replace=False,
                                                n_samples=train_size,
                                                random_state=random_state)

    # Impute train and test separately:
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.fit_transform(X_test)
    # Combat
    X_train = CombatModel().fit_transform(X_train, site_train.reshape(-1, 1))
    X_test = CombatModel().fit_transform(X_test, site_test.reshape(-1, 1))

    trained = model.fit(X_train, y_train)
    metric = []
    value = []

    y_pred = trained.predict(X_test)
    metric.append("mae")
    value.append(mean_absolute_error(y_test, y_pred))
    metric.append("mad")
    value.append(median_absolute_error(y_test, y_pred))
    metric.append("r2")
    value.append(r2_score(y_test, y_pred))

    result = {'Model': [model] * len(metric),
              'Metric': metric,
              'Value': value}

    return pd.DataFrame(result), pd.DataFrame(dict(y_pred=y_pred.squeeze(), y_test=y_test))


def learning_curve(x, max_acc, min_acc, k):
    """
    A model of change in R2 as a function of sample size
    """
    return max_r2 - (max_r2 - min_r2) * np.exp(-1 * (x - np.min(x)) / k)


model_dict = {
#   "cnn_lenet": {"model": cnn_lenet, "lr": 0.001},
#   "mlp4": {"model": mlp4, "lr": 0.001},
#   "cnn_vgg": {"model": cnn_vgg, "lr": 0.001},
#   "lstm1v0": {"model": lstm1v0, "lr": 0.01},
#   "lstm1": {"model": lstm1, "lr": 0.01},
#   "lstm2": {"model": lstm2, "lr": 0.01},
#   "blstm1": {"model": blstm1, "lr": 0.01},
#   "blstm2": {"model": blstm1, "lr": 0.01},
#   "lstm_fcn": {"model": lstm_fcn, "lr": 0.01},
#   "cnn_resnet": {"model": cnn_resnet, "lr": 0.01},
  "pclasso": {"model": None, "lr": None}
             }

metric_to_slice = {"fa": slice(0, 24),
                   "md": slice(24, 48),
                   "mk": slice(48, 72)}

seeds = np.array([484, 645, 714, 244, 215, 1503, 1334, 1576, 469, 1795])

train_sizes = [100, 175, 350, 700, 1000, None]


@pydra.mark.task
def train_cnn_on_hyak(model, run, train_size=None, metric=None):

    # Create local filesystem:
    out_path = op.join("/gscratch/escience/arokem/afqdl-data/")
    os.makedirs(out_path, exist_ok=True)
    print(f"Output path is {out_path}")
    seed = seeds[run]

    if model == "pclasso":
        X, y, site = load_data_notf()
        if metric is not None:
            start = metric_to_slice[metric].start * 100
            stop = metric_to_slice[metric].stop * 100
            X = X[:, start:stop]

        model_obj = make_base_afq_pipeline(
            feature_transformer=PCA,
            scaler="standard",
            estimator=LassoCV,
            estimator_kwargs={
                    "verbose": 0,
                    "n_alphas": 50,
                    "cv": 3,
                    "n_jobs": 28,
                    "max_iter": 500}
                    )
        eval, pred = fit_and_eval_notf(
            model_obj,
            X,
            y,
            site,
            random_state=seed,
            train_size=train_size)
    else:
        X, y, site = load_data()
        if metric is not None:
            X = X[:, :, metric_to_slice[metric]]

        eval, pred = fit_and_eval(
            model,
            model_dict,
            X,
            y,
            site,
            random_state=seed,
            train_size=train_size)

    eval["run"] = run
    pred["run"] = run

    if train_size is None:
        train_size = "all"

    if metric is None:
        metric = "all"

    eval.to_csv(f"{out_path}/{model}_run-{run}_train-{train_size}_metric-{metric}_eval.csv")
    pred.to_csv(f"{out_path}/{model}_run-{run}_train-{train_size}_metric-{metric}_pred.csv")

sbatch_args = "-J afqdl -p gpu-a40 -A escience --mem=58G --time=18:00:00 -o /gscratch/scrubbed/arokem/logs/afqdl.out -e /gscratch/scrubbed/arokem/logs/afqdl.err --mail-user=arokem@uw.edu --mail-type=ALL --partition=gpu-a40 --gpus=1"


if __name__ == "__main__":
    scratch_dir = "/gscratch/scrubbed/arokem/"
    scratch_dir_tmp = op.join(scratch_dir, "tmp_")
    cache_dir_tmp = mkdtemp(prefix=scratch_dir_tmp)
    today = datetime.today().strftime('%Y-%m-%d')
    today = "2023-03-21"
    for model in list(model_dict.keys()):
        for run in range(10):
            for train_size in train_sizes:
                for metric in [None, "fa", "md", "mk"]:
                    task = train_cnn_on_hyak(
                        model=model,
                        run=run,
                        metric=metric,
                        train_size=train_size,
                        cache_dir=cache_dir_tmp)
                    try:
                        with pydra.Submitter(
                            plugin="slurm",
                            sbatch_args=sbatch_args) as sub:
                                sub(runnable=task)
                    except Exception as e:
                        print(e)
                        continue
