from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# get path from current directory
FILE_PATH = Path(__file__).parent.absolute()

def main():
    # save path used in heatmap
    save_path = FILE_PATH / './data/'

    # variables to access data
    base_data_dir = Path('/opt', 'data', 'gen_data')
    data_dir = base_data_dir / 'raw'
    sub_dir = 'train'

    # get all sub-directories
    dirs = os.listdir(data_dir / sub_dir)
    n = 0
    files = []
    truth = []
    for dd in dirs:
        n = n + np.size(os.listdir(data_dir / sub_dir / dd))
        for ff in os.listdir(data_dir / sub_dir / dd):
            files.append(data_dir / sub_dir / dd / ff)
            truth.append(dd)
    temp = list(zip(files, truth))
    np.random.shuffle(temp)
    files, truth = zip(*temp)

    # load data zero
    zero_short = np.zeros((32, 100, 100))
    zero_long = np.zeros((100, 100))
    for ii in range(1000):
        if truth[ii] == '0':
            zero_short = np.load(files[ii]) / 128. -1
            zero_long = np.sum(zero_short, 0) / 32.
            break

    # load data one
    one_short = np.zeros((32, 100, 100))
    one_long = np.zeros((100, 100))
    for ii in range(1000):
        if truth[ii] == '1':
            one_short = np.load(files[ii])
            one_long = np.sum(one_short, 0)
            break

    # load data two
    two_short = np.zeros((32, 100, 100))
    two_long = np.zeros((100, 100))
    for ii in range(1000):
        if truth[ii] == '2':
            two_short = np.load(files[ii]) / 128. -1
            two_long = np.sum(two_short, 0) / 32.
            break

    # save off data
    np.save(save_path / 'zero_short.npy', zero_short)
    np.save(save_path / 'zero_long.npy', zero_long)
    np.save(save_path / 'one_short.npy', one_short)
    np.save(save_path / 'one_long.npy', one_long)
    np.save(save_path / 'two_short.npy', two_short)
    np.save(save_path / 'two_long.npy', two_long)


if __name__ == '__main__':
    main()