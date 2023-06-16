import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal


def get_freq(movements):
    # movements = pd.read_csv("data/light shake.csv")

    Therapist_position_x = movements['Therapist position x']
    Therapist_position_y = movements['Therapist position y']
    Therapist_position_z = movements['Therapist position z']

    Therapist_angle_3 = movements['Therapist joint angle 3']
    Therapist_angle_4 = movements['Therapist joint angle 4']
    Therapist_angle_5 = movements['Therapist joint angle 5']

    freqp_x, fft_p_x = fft_calculation(Therapist_position_x)
    freqp_y, fft_p_y = fft_calculation(Therapist_position_y)
    freqp_z, fft_p_z = fft_calculation(Therapist_position_z)

    freqp_3, fft_p_3 = fft_calculation(Therapist_angle_3)
    freqp_4, fft_p_4 = fft_calculation(Therapist_angle_4)
    freqp_5, fft_p_5 = fft_calculation(Therapist_angle_5)

    # plt.title('Real FFT')
    # plt.xlabel('Hz')
    # plt.ylabel('Real Part')
    # plt.plot(freqp_z, fft_p_z)
    # plt.savefig('fft_x.png')

    fft_x = output_characteristics(fft_p_x)
    fft_y = output_characteristics(fft_p_y)
    fft_z = output_characteristics(fft_p_z)

    fft_3 = output_characteristics(fft_p_3)
    fft_4 = output_characteristics(fft_p_4)
    fft_5 = output_characteristics(fft_p_5)
    # print("fft_x: ", fft_x)
    # print("fft_y: ", fft_y)
    # print("fft_z: ", fft_z)
    return fft_x, fft_y, fft_z, fft_3, fft_4, fft_5


# calculate fft
def fft_calculation(oneD_data):
    # fft calculate
    omega_real = np.fft.rfftfreq(oneD_data.size, d=1. / 30)
    real_fft = np.fft.rfft(oneD_data)
    db_fft = 20 * np.log10(np.absolute(real_fft))

    # median filter, downsampling
    fil_freq = signal.medfilt(omega_real, kernel_size=51)
    fil_fft = signal.medfilt(db_fft, kernel_size=51)

    # window 2～12HZ for stroke patient tremor
    ds_freq = fil_freq[fil_freq > 2]
    ds_arr1 = fil_fft[(-ds_freq.size):]

    DS_freq = ds_freq[ds_freq < 12]
    DS_arr1 = ds_arr1[:DS_freq.size]

    return (DS_freq, DS_arr1)


# output characteristics_size
def output_characteristics(freq_traits):
    # output a 1D traits of freq
    output_size = 10
    kernel = freq_traits.size // output_size
    output = []
    for i in range(0, output_size):
        output.append(freq_traits[kernel * i])
    return output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_freq()
