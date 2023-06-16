import pandas as pd
import numpy as np
import glob
import os
from get_freq import *
from tensorflow.keras.utils import to_categorical


def file_loader(flag):
    if flag == 'train':
        path = r'C:\Users\flash\OneDrive\桌面\classification_different_people_hand\data'  # use your path
        all_files = glob.glob(path + "/*.csv")
    if flag == 'test':
        path = r'C:\Users\flash\OneDrive\桌面\classification_different_people_hand\test'  # use your path
        all_files = glob.glob(path + "/*.csv")

    x_input = []
    y_label = []

    for filename in all_files:
        df = pd.read_csv(filename)

        # get freq features
        fft_x, fft_y, fft_z, fft_3, fft_4, fft_5 = get_freq(df)

        new = np.array(fft_x)
        new = np.concatenate((new, np.array(fft_y)))
        new = np.concatenate((new, np.array(fft_z)))
        new = np.concatenate((new, np.array(fft_3)))
        new = np.concatenate((new, np.array(fft_4)))
        new = np.concatenate((new, np.array(fft_5)))

        x_input.append(new)

        # get label
        label = df[['label']].values
        y_label.append(label[0][0])
        # print(y_label)

    print(np.array(x_input), np.array(y_label))

    # Convert to one-hot
    one_hot_labels = to_categorical(y_label, num_classes=3)

    print(one_hot_labels)
    return np.array(x_input), np.array(one_hot_labels)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_loader('test')
