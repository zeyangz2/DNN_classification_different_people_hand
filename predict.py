from tensorflow.keras.models import Sequential, model_from_json
import pandas as pd
from tensorflow.keras.models import load_model
from get_freq import *
import pickle
import glob

def predict():
    path = r'C:\Users\flash\OneDrive\桌面\classification_different_people_hand\zeyang_test'  # use your path
    all_files = glob.glob(path + "/*.csv")

    #load data
    # df = pd.read_csv('zeyang_right_33.csv')

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

    # # get freq features
    # fft_x, fft_y, fft_z, fft_3, fft_4, fft_5 = get_freq(df)
    #
    # new = np.array(fft_x)
    # new = np.concatenate((new, np.array(fft_y)))
    # new = np.concatenate((new, np.array(fft_z)))
    # new = np.concatenate((new, np.array(fft_3)))
    # new = np.concatenate((new, np.array(fft_4)))
    # new = np.concatenate((new, np.array(fft_5)))
    #
    # x_input.append(new)

    # get label
    # label = df[['label']].values
    # y_label.append(label[0][0])
    # print('ground truth: ', y_label)

    x_input = np.array(x_input)

    # load model
    model = load_model('model_new.h5')
    # summarize model.
    model.summary()
    print("Loaded model from disk")

    # # evaluate loaded model on test data
    # loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # score = loaded_model.evaluate(x_input, y_label, verbose=0)
    # print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))

    # Make a prediction
    probabilities = model.predict(x_input)

    # Take the class with the highest probability
    predicted_classes = np.argmax(probabilities, axis=1)

    print('predicted class: ', predicted_classes)

    # original_labels = encoder.inverse_transform(predicted_classes.reshape(-1, 1))
    # print(original_labels)

    # Load dictionary from a pickle file
    with open('my_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

    print(loaded_dict)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predict()