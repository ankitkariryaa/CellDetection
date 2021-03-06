import os

from keras.callbacks import ModelCheckpoint, TensorBoard


def start_training(checkpoint_dir, model, dataset_generator, samples_per_epoc,
                   nb_epoc, callbacks=[], validation_generator=None, nb_val_samples=None):
    checkpoint_file = os.path.join(checkpoint_dir, 'model_checkpoints', 'model.hdf5')
    model_dir = os.path.join(checkpoint_dir, 'model_checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tf_logdir = os.path.join(checkpoint_dir, 'tf_logs')

    checkpoint_cb = ModelCheckpoint(checkpoint_file, verbose=1, save_best_only=False)
    tensorboard_cb = TensorBoard(tf_logdir, write_images=True)
    all_callbacks = callbacks + [checkpoint_cb, tensorboard_cb]

    # model.summary()
    model.fit_generator(dataset_generator, samples_per_epoch=samples_per_epoc, nb_epoch=nb_epoc,
                        callbacks=all_callbacks, validation_data=validation_generator, nb_val_samples=nb_val_samples)
