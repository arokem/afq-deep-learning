import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from afqinsight.augmentation import jitter, time_warp, scaling, magnitude_warp, window_warp
import tempfile

def load_data():
    afq_dataset.drop_target_na()


def tf_aug(X_in, scaler=1/20):
    X_out = np.zeros_like(X_in)
    for channel in range(X_in.shape[-1]):
        this_X = X_in[..., channel][np.newaxis, ..., np.newaxis]
        scale = np.abs(np.max(this_X) - np.min(this_X)) * scaler
        this_X = jitter(this_X, sigma=scale)
        this_X = scaling(this_X, sigma=scale)
        this_X = time_warp(this_X, sigma=scale)
        this_X = window_warp(this_X)
        X_out[..., channel] = this_X[0, ..., 0]
    return X_out


def augment_this(X_in, y_in):
    X_out = tf.numpy_function(tf_aug, [X_in], tf.float32)    
    return X_out, y_in


def model_fit(model_func, X_train, y_train, lr, batch_size=128, n_epochs=1000):
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
    train_dataset = train_dataset.map(augment_this) 
    train_dataset = train_dataset.batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    history = model.fit(train_dataset, epochs=n_epochs, validation_data=val_dataset,
                        callbacks=callbacks, use_multiprocessing=True)
    model.load_weights(ckpt_filepath)
    return model
