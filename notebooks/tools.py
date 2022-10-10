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

AUG_SCALING = 1/5


def load_data():
    afq_dataset = AFQDataset.from_files(
        fn_nodes="../data/raw/combined_tract_profiles.csv",
        fn_subjects="../data/raw/participants_updated_id.csv",
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
        this_X = window_warp(this_X, window_ratio=scale)
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
                        callbacks=callbacks, use_multiprocessing=True)
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
