import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from file_loader import *
from advanced_loader import *
import pickle


def main():
    # df = pd.read_csv('demo1.csv')
    # df.head()
    #
    # # shuffle the DataFrame rows
    # df = df.sample(frac=1)

    # #load data
    # x_input = df[['Joint angle 0', 'Joint angle 1', 'Joint angle 2', 'Stylus X', 'Stylus Y', 'Stylus Z', 'Ball X', 'Ball Y', 'Ball Z']].values
    # #left hand = 0, right hand = 1
    # y_label = df[['label']].values

    # x_input, y_label = file_loader('train')
    #
    # # 0 for Zeyang, 1 for Frank
    # print(y_label)

    x_input, y_label, label_to_name = advanced_loader()

    # using the train test split function
    x_train, x_test, y_train, y_test = train_test_split(x_input, y_label,
                                                        random_state=104,
                                                        test_size=0.2,
                                                        shuffle=True)

    model = Sequential()

    model.add(Dense(units=100, input_dim=60, activation='relu'))
    model.add(Dense(units=50, activation='relu'))
    model.add(Dense(units=20, activation='relu'))

    model.add(Dense(units=3, activation='softmax'))  # softmax/sigmoid may also work

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # train
    history = model.fit(x_train, y_train, epochs=1200, batch_size=8, validation_split=0.2)

    # Plotting the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('Model Loss.png')

    print('testing:')
    score, acc = model.evaluate(x_test, y_test, batch_size=5, verbose=1)
    print('Test accuracy: ', acc)

    # save model and architecture to single file
    model.save("model_new.h5")
    print("Saved model to disk")
    print(label_to_name)

    # Save dictionary into a pickle file
    with open('my_dict.pkl', 'wb') as f:
        pickle.dump(label_to_name, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
