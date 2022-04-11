import os
import typing
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import matplotlib.pyplot as plt

def light_prop_data():
    # variables to access data
    base_data_dir = Path('/opt', 'data', 'gen_data')
    data_dir = base_data_dir / 'raw'

    # flags for running model
    force_fit_model = True  # if True then retrain model
    use_sequenced_data = True  # if True then use sequence data with RNN, if not use vector with Dense
    validation_split = 0.3

    # training variables
    batch_size: int = 32
    max_frame_length = 32

    train_ds: tf.data.Dataset
    valid_ds: tf.data.Dataset
    get_dataset(base_dir=base_data_dir,
                data_dir=data_dir,
                sub_dir='train',
                max_frame_length=max_frame_length,
                batch_size=batch_size,
                validation_split=validation_split,
                use_vector_data=not use_sequenced_data)

    get_dataset(base_dir=base_data_dir,
                data_dir=data_dir,
                sub_dir='test',
                max_frame_length=max_frame_length,
                batch_size=batch_size,
                validation_split=validation_split,
                use_vector_data=not use_sequenced_data)

    print('done')


def get_dataset(base_dir: Path,
                data_dir: Path,
                sub_dir: str,
                max_frame_length: int,
                batch_size: int,
                validation_split: typing.Optional[float] = 0.3,
                use_vector_data=False) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, typing.List[str]]:

    records_per_file = 300

    # get all sub-directories
    dirs = os.listdir(data_dir / sub_dir)
    num_dirs = np.size(dirs)
    n = 0
    for dd in dirs:
        n = n + np.size(os.listdir(data_dir / sub_dir / dd))
    num = int(np.ceil(n / records_per_file))

    # create arrays and tfrecords in batches as the data arrays can exceed available RAM
    for nn in range(num):
        # initialize data arrays
        x_data = []
        y_data = []
        for ii in range(int(records_per_file / num_dirs)):  # start with ii to get a mix of categories in tfrecord file
            for dd in range(num_dirs):
                # get all files in subdirectory and get labels based on sub-directory name
                files = os.listdir(data_dir / sub_dir / dirs[dd])
                if ii < np.size(files):
                    x = np.load(data_dir / sub_dir / dirs[dd] / files[ii])
                    x_data.append(x)
                    y_data.append([int(dirs[dd])])

        # turn imported data into tfrecord files
        record = convert(np.asarray(x_data), np.asarray(y_data), base_dir / 'tfr' / sub_dir, np.shape(x_data)[0], nn)

    print('pulled data')


# Function for reading images from disk and writing them along with the class-labels to a TFRecord file.
def convert(x_data: np.ndarray,
            y_data: np.ndarray,
            out_path: Path,
            records_per_file: int,
            it: int = 0
            ) -> typing.List[Path]:
    """
    Function for reading images from disk and writing them along with the class-labels to a TFRecord file.

    :param x_data: the input, feature data to write to disk
    :param y_data: the output, label, truth data to write to disk
    :param out_path: File-path for the TFRecords output file.
    :param records_per_file: the number of records to use for each file
    :return: the list of tfrecord files created
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Open a TFRecordWriter for the output-file.
    record_files = []
    n_samples = x_data.shape[0]

    # Iterate over all the image-paths and class-labels.
    n_tfrecord_files = int(np.ceil(n_samples / records_per_file))
    for idx in tqdm(range(n_tfrecord_files), desc="Convert Batch", total=n_tfrecord_files, position=0):
        record_file = out_path / f'train{idx + it}.tfrecord'
        record_files.append(record_file)
        slicer = slice(idx * records_per_file, (idx + 1) * records_per_file)
        with tf.io.TFRecordWriter(str(record_file)) as writer:
            for x_sample, y_sample in tqdm(zip(x_data[slicer], y_data[slicer]),
                                           desc="Convert Image in batch",
                                           total=records_per_file,
                                           position=1,
                                           leave=False):
                # Convert the ndarray of the image to raw bytes. note this is bytes encodes as uint8 types
                img_bytes = x_sample.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = {
                    'image': _bytes_feature(img_bytes),
                    'label': _int64_list_feature(y_sample)
                }
                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)
                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)
                # Serialize the data.
                serialized = example.SerializeToString()
                # Write the serialized data to the TFRecords file.
                writer.write(serialized)

    print('created record')
    return record_files


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_feature(value: typing.List[int]):
    """Returns an int64_list from a list of bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == "__main__":
    light_prop_data()