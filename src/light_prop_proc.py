import os
import typing
from pathlib import Path
import numpy as np
import sklearn
import sklearn.metrics
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Conv2D, Flatten, Reshape
from tensorflow.keras.models import Model, load_model


def light_prop_proc():
    base_dir = Path('/opt', 'data', 'gen_data', 'tfr')
    data_img_shape = (32, 100, 100)

    vocab = np.asarray([0, 1, 2]).astype('uint64')
    vocab_str = np.asarray(['zero', 'one', 'two'])
    visualize_ds_flag = False
    long_exposure_flag = True
    model_type = 'cnn'  # values: rnn, cnn, or crnn

    batch_size = 100
    num_shuffle_batches = 3
    epochs = 200
    validation_split = 0.75
    model_filename = 'model'
    img_file = './model_arch'

    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    metrics = 'categorical_accuracy'

    epoch_steps = 10
    val_steps = 10
    test_steps = 1

    # get files for use in dataset
    train_files = [base_dir / 'train' / file_name for file_name in os.listdir(base_dir / 'train')]
    test_files = [base_dir / 'test' / file_name for file_name in os.listdir(base_dir / 'test')]

    # split the training files into training and validation
    split = int(len(train_files) * validation_split)
    valid_files = train_files[split:]
    train_files = train_files[:split]

    # get datasets from files
    train_dataset: tf.data.Dataset = get_dataset(train_files, img_shape=data_img_shape)
    valid_dataset: tf.data.Dataset = get_dataset(valid_files, img_shape=data_img_shape)
    test_dataset: tf.data.Dataset = get_dataset(test_files, img_shape=data_img_shape)

    def map_1h(image_inside_batched, label):
        # scale input data
        image_inside_batched = image_inside_batched / 128.  # scale from 0 to 2
        image_inside_batched = image_inside_batched - 1  # Zero-center
        # add dimension to note monochromatic light
        image_inside_batched = tf.expand_dims(image_inside_batched, 3)
        # one-hot encode output when flag is up
        label = tf.one_hot(int(label), depth=len(vocab))
        return image_inside_batched, label

    def map_1hls(image_inside_batched, label):
        # scale input data
        image_inside_batched = image_inside_batched / 128.  # scale from 0 to 2
        image_inside_batched = image_inside_batched - 1  # Zero-center
        # create 'long-exposure' images from short-exposure images
        image_inside_batched = tf.reduce_sum(image_inside_batched, axis=0, keepdims=False) / 32.
        # add dimension to note monochromatic light
        image_inside_batched = tf.expand_dims(image_inside_batched, 2)
        # one-hot encode output when flag is up
        label = tf.one_hot(int(label), depth=len(vocab))
        return image_inside_batched, label

    # repeat, shuffle, scale, batch, and prefetch the datasets in a function
    def repeat_shuffle_scale_batch_prefetch_dataset(dataset: tf.data.Dataset):
        # repeat
        dataset = dataset.repeat()
        # shuffle
        dataset = dataset.shuffle(buffer_size=batch_size * num_shuffle_batches)
        # map (based on given flags)
        if long_exposure_flag:
            dataset = dataset.map(map_func=map_1hls, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.map(map_func=map_1h, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # chain together batch and prefetch
        dataset = dataset.batch(batch_size=batch_size).prefetch(1)
        return dataset

    # process our datasets in a pipeline
    train_dataset = repeat_shuffle_scale_batch_prefetch_dataset(train_dataset)
    valid_dataset = repeat_shuffle_scale_batch_prefetch_dataset(valid_dataset)
    test_dataset = repeat_shuffle_scale_batch_prefetch_dataset(test_dataset)

    # check our dataset by visualizing it. Note this can go forever if you let it
    if visualize_ds_flag:
        for img_batch, label_batch in train_dataset.as_numpy_iterator():
            # display 1 long exposure frame or multiple short exposure frames
            if long_exposure_flag:
                fig, ax = plt.subplots(1, 1)
                plt.suptitle(f'Label: {label_batch[0]}\n'
                             f'X Shape: {np.shape(img_batch)}, Y Shape: {np.shape(label_batch)}')
                ax.imshow(img_batch[0, :, :, 0])
                ax.axis('off')
                plt.tight_layout()
                plt.show()
                print('break point 2')
            else:
                fig, ax = plt.subplots(4, 8)  # assumes at least 32 frames per sample
                plt.suptitle(f'Label: {label_batch[0]}\n'
                             f'X Shape: {np.shape(img_batch)}, Y Shape: {np.shape(label_batch)}')
                for rr in range(4):
                    for cc in range(8):
                        ax[rr, cc].imshow(img_batch[0, (rr * 4) + cc, :, :])
                        ax[rr, cc].axis('off')
                plt.tight_layout()
                plt.show()
                print('break point 2')

    # create model

    if model_type == 'crnn':
        print('USING CRNN')
        model = create_crnn_model(data_img_shape, vocab)
    elif model_type == 'cnn':
        print('USING CNN')
        model = create_cnn_model(data_img_shape, vocab)
    elif model_type == 'rnn':
        print('USING RNN')
        model = create_rnn_model(data_img_shape, vocab)
    else:
        model = None

    # use specific optimizer, loss, and accuracy based on the output type
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    # disply summary of model and save an image of the used model
    model.summary()
    tf.keras.utils.plot_model(model, to_file=(img_file+'_'+model_type+'.png'), show_shapes=True)

    # fit model to dataset
    model.fit(train_dataset, batch_size=batch_size, epochs=epochs, steps_per_epoch=epoch_steps,
              validation_data=valid_dataset, validation_steps=val_steps, verbose=1)

    # save model
    model.save(model_filename+'_'+model_type+'.h5')

    # evaluate on test dataset
    test_results = model.evaluate(test_dataset, steps=test_steps)
    for metric, name in zip(test_results, model.metrics_names):
        print(f"{name}: {metric}")

    test_pts = 1000
    iterations = int(test_pts / batch_size)

    y_test = np.zeros((iterations * batch_size, 2))
    x = []
    y = []
    for ii in range(iterations):
        x_it, y_it = test_dataset.as_numpy_iterator().next()
        if ii == 0:
            x = x_it
            y = y_it
        else:
            x = np.concatenate((x, x_it), axis=0)
            y = np.concatenate((y, y_it), axis=0)
    p_prob = model.predict(x)
    p = np.argmax(p_prob, 1)
    y = np.argmax(y, 1)

    # remove data with no objects to detect
    prob_roc = p_prob[y>0, 1:]
    truth = y[y>0]

    divs = 1000
    sz = np.size(truth)
    p_roc = np.ones((sz, divs))
    thresh = np.linspace(0, 1, divs)
    dfa = np.zeros((divs, 2))
    print(f'Size: {sz}')
    for dd in range(divs):
        p_roc[prob_roc[:, 1] > thresh[dd], dd] = 2
        cm_temp = sklearn.metrics.confusion_matrix(y_pred=p_roc[:, dd], y_true=truth)
        dfa[dd, 0] = cm_temp[1, 1] / (cm_temp[1, 0] + cm_temp[1, 1])  # prob detect
        dfa[dd, 1] = cm_temp[0, 1] / (cm_temp[0, 0] + cm_temp[0, 1])  # prob false alarm

    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(dfa[:, 1], dfa[:, 0])
    ax[0].plot(dfa[:, 1], dfa[:, 0])
    ax[0].set(xlabel='P-False', ylabel='P-Detect')
    ax[1].plot(thresh, dfa[:, 0])
    ax[1].plot(thresh, dfa[:, 1])
    ax[1].set(xlabel='Threshold', ylabel='Probability')
    ax[1].legend(['% Detect', '% False'])
    plt.tight_layout()
    plt.show()

    print(sklearn.metrics.classification_report(y, p, target_names=vocab_str))
    confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=p, y_true=y)
    plot_confusion_matrix(confusion_matrix, classes=vocab_str, title='Confusion matrix, without normalization')
    plt.show()

    print('done')


def create_rnn_model(data_shape, vocab):
    input_ = Input(shape=(data_shape[0], data_shape[1], data_shape[2], 1,))

    layer_ = Reshape((data_shape[0], data_shape[1] * data_shape[2],))(input_)
    layer_ = LSTM(64, return_sequences=True, dropout=0.2, activation='tanh')(layer_)
    layer_ = LSTM(32, return_sequences=True, dropout=0.2, activation='tanh')(layer_)
    layer_ = LSTM(16, return_sequences=False, dropout=0.2, activation='tanh')(layer_)

    layer_ = Dense(units=64, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    layer_ = Dense(units=32, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    output_ = Dense(units=len(vocab), activation='sigmoid')(layer_)

    model = Model(inputs=input_, outputs=output_)

    return model


def create_cnn_model(data_shape, vocab):
    input_ = Input(shape=(data_shape[1], data_shape[2], 1,))

    layer_ = Conv2D(filters=10, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='relu')(input_)
    layer_ = Conv2D(filters=20, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(layer_)
    layer_ = Conv2D(filters=30, kernel_size=(10, 10), strides=(5, 5), padding='same', activation='relu')(layer_)
    layer_ = Conv2D(filters=40, kernel_size=(10, 10), strides=(5, 5), padding='same', activation='relu')(layer_)
    layer_ = Flatten()(layer_)
    layer_ = Dense(units=64, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    layer_ = Dense(units=32, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    output_ = Dense(units=len(vocab), activation='softmax')(layer_)

    model = Model(inputs=input_, outputs=output_)

    return model


def create_crnn_model(data_shape, vocab):
    input_ = Input(shape=(data_shape[0], data_shape[1], data_shape[2], 1,))

    layer_ = Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(input_)
    layer_ = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(layer_)
    layer_ = Conv2D(filters=64, kernel_size=(8, 8), strides=(8, 8), padding='same', activation='relu')(layer_)

    layer_ = Reshape((data_shape[0], -1,))(layer_)
    layer_ = LSTM(64, return_sequences=True, dropout=0.2, activation='tanh')(layer_)
    layer_ = LSTM(32, return_sequences=True, dropout=0.2, activation='tanh')(layer_)
    layer_ = LSTM(16, return_sequences=False, dropout=0.2, activation='tanh')(layer_)

    layer_ = Dense(units=64, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    layer_ = Dense(units=32, activation='relu')(layer_)
    layer_ = Dropout(0.2)(layer_)
    output_ = Dense(units=len(vocab), activation='softmax')(layer_)

    model = Model(inputs=input_, outputs=output_)

    return model


def get_dataset(filenames: typing.List[Path], img_shape: tuple) -> tf.data.Dataset:
    """
    This function takes the filenames of tfrecords to process into a dataset object
    The _parse_function takes a serialized sample pulled from the tfrecord file and
    parses it into a sample with x (input) and y (output) data, thus a full sample for training

    :param filenames: the file names of each tf record to process
    :param img_shape: the size of the images a width, height, channels
    :return: the dataset object made from the tfrecord files and parsed to return samples
    """

    def _parse_function(serialized):
        """
        This function parses a serialized object into tensor objects to use for training
        """
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)
        # convert the image shape to a tensorflow object
        image_shape = tf.stack(img_shape)
        # get the raw feature bytes
        image_raw = parsed_example['image']
        # Decode the raw bytes so it becomes a tensor with type.
        image_inside = tf.io.decode_raw(image_raw, tf.uint8)
        # cast to float32 for GPU operations
        image_inside = tf.cast(image_inside, tf.float32)
        # reshape to correct image shape
        image_inside = tf.reshape(image_inside, image_shape)
        # get the label and convert it to a float32
        label = tf.cast(parsed_example['label'], tf.float32)
        # return a single tuple of the (features, label)
        return image_inside, label

    # the tf functions takes string names not path objects, so we have to convert that here
    filenames_str = [str(filename) for filename in filenames]

    # make a dataset from slices of our file names
    files_dataset = tf.data.Dataset.from_tensor_slices(filenames_str)

    # make an interleaved reader for the TFRecordDataset files
    # this will give us a stream of the serialized data interleaving from each file
    dataset = files_dataset.interleave(map_func=lambda x: tf.data.TFRecordDataset(x),
                                       # 12 was picked for the cycle length because there were 12 total files,
                                       # and I wanted to cycle through all of them
                                       cycle_length=12,  # how many files to cycle through at once
                                       block_length=1,  # how many samples from each file to get
                                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                       deterministic=False)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(map_func=_parse_function,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    light_prop_proc()
