import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf

from DamDataGenerator import DataGenerator


def train(
        model: tf.keras.Model,
        data_generator: DataGenerator,
        model_type: str,
        training_params: dict,
        output_path: Path = Path("./output"),
        save_first_vals: bool = False
):
    loss = training_params["loss"]
    learning_rate = training_params["learning_rate"]
    optimizer_kind = training_params["optimizer"]
    epochs = training_params["epochs"]

    val_images, val_labels = data_generator.get_valbatch()

    batch_per_epoch = data_generator.batch_per_epoch
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[10 * batch_per_epoch, 20 * batch_per_epoch],
        values=[learning_rate, learning_rate / 10, learning_rate / 100]
    )

    optimizer = get_optimizer(optimizer_kind, learning_rate_fn)
    model = model.compile(optimizer=optimizer, loss=loss)

    net_name = model_type + str(uuid.uuid1())
    print('Training Supervised Network: ')
    print(net_name)

    model_output_path = output_path / net_name
    model_output_path.mkdir(parents=True)

    if save_first_vals:
        first_val = model.evaluate(val_images, val_labels, verbose=1)
        npy_path = str((model_output_path / "firstval.npy").absolute())
        np.save(npy_path, first_val)

    log_output_path = model_output_path / "losses.csv"
    csv_logger = tf.keras.callbacks.CSVLogger(log_output_path, append=True, separator=';')
    model.fit(data_generator, verbose=2, epochs=epochs, steps_per_epoch=batch_per_epoch,
              validation_data=(val_images, val_labels), validation_steps=None, validation_freq=1,
              callbacks=[csv_logger])
    model.save(model_output_path / "model")
    return model


def get_optimizer(opt, learning_rate_fn):
    if opt == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate_fn)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    elif opt == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_fn)
    elif opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn)
    elif opt == 'adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_fn)
    return optimizer


if __name__ == '__main__':
    print("HelloWorld")
