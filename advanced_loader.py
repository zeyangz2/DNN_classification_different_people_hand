import os
import pandas as pd
from get_freq import *
from tensorflow.keras.utils import to_categorical

def advanced_loader():
    # Define the root directory that contains the folders with the CSV files
    root_dir = r'C:\Users\flash\OneDrive\桌面\classification_different_people_hand\data_by_names'

    x_input = []
    y_label = []

    # Dictionary to store folder-to-label mappings
    folder_to_label = {}
    label_to_folder = {}

    # Initialize label
    label = -1

    # Traverse through all folders and subfolders in the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the directory contains any CSV files
        if any(filename.endswith('.csv') for filename in filenames):
            # Assign a label to the folder
            label += 1
            name = os.path.basename(dirpath)
            folder_to_label[name] = label
            label_to_folder[label] = name

            # Load all CSV files in the directory
            for filename in filenames:
                if filename.endswith('.csv'):
                    file_path = os.path.join(dirpath, filename)
                    df = pd.read_csv(file_path, skipfooter=1, engine='python')
                    # process the dataframe as needed

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
                    y_label.append(label)

    # Print out folder-to-label mappings
    print(folder_to_label)

    # Print out label-to-folder mappings
    print(label_to_folder)

    print(np.array(x_input), np.array(y_label))

    # Convert to one-hot
    one_hot_labels = to_categorical(y_label, num_classes=3)

    print(one_hot_labels)
    return np.array(x_input), np.array(one_hot_labels), label_to_folder




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    advanced_loader()