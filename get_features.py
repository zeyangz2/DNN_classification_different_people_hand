import pandas as pd
import numpy as np
from scipy import fftpack, signal
from scipy.stats import entropy
import pywt

def get_features(df):
    # df = pd.read_csv('trial2.csv')
    # load data
    therapist_input = df[
        ['Therapist position x', 'Therapist position y', 'Therapist position z']].values
    patient_input = df[
        ['Patient position x', 'Patient position y', 'Patient position z']].values

    # y_label = df[['label']].values

    # print(x_input)

    # Fourier Transform
    x_coordinates = patient_input[:, 0]
    # Subtract the mean from the signal to remove the DC component
    x_coordinates = x_coordinates - np.mean(x_coordinates)
    x_fft = fftpack.fft(x_coordinates)
    fft_freq = fftpack.fftfreq(len(x_coordinates))

    # Spectral Energy
    spectral_energy = np.sum(np.abs(x_fft) ** 2)

    # Power Spectral Density
    power_spectral_density = np.abs(x_fft) ** 2 / len(x_coordinates)

    # Spectral Entropy
    spectral_entropy = entropy(power_spectral_density)

    # Dominant Frequency
    # Find the indices of all maxima
    maxima_indices = np.where(power_spectral_density == power_spectral_density.max())[0]
    # print(maxima_indices)

    # Find the maximum frequency among all maxima
    dominant_frequency = fft_freq[maxima_indices.max()]
    # print(dominant_frequency)

    # Find the indices for the top 5 power spectral densities
    top_5_indices = np.argsort(power_spectral_density)[-5:]

    # Sort the indices in descending order so the largest power spectral density is first
    top_5_indices = top_5_indices[::-1]
    # print(top_5_indices)

    # Find the corresponding frequencies
    top_5_freq = fft_freq[top_5_indices]

    print('Spectral Energy (x):', spectral_energy)
    print('Spectral Entropy (x):', spectral_entropy)
    print('Dominant Frequency (x):', dominant_frequency)
    print('Top 5 Frequencies (x):', top_5_freq)

    mean_diff, std_diff, top_5_diff, lowest_5_diff = get_euclidean(therapist_input, patient_input)

    result = []
    result = np.array(result)
    result = np.insert(result, 0, mean_diff)
    result = np.insert(result, 0, std_diff)
    result = np.append(result, top_5_diff)
    result = np.append(result, lowest_5_diff)
    print('std, mean, top5, low5: ', result)
    return result


def get_euclidean(therapist_input, patient_input):
    if len(therapist_input) != len(patient_input):
        raise ValueError("The two trajectories must be of the same length")

    # Compute the Euclidean distance between each pair of corresponding points
    differences = np.linalg.norm(therapist_input - patient_input, axis=1)

    # Calculate the mean and standard deviation
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # Find the top 5 and lowest 5 differences
    top_5_diff = np.sort(differences)[-5:]
    lowest_5_diff = np.sort(differences)[:5]

    print("Mean of differences:", mean_diff)
    print("Standard deviation of differences:", std_diff)
    print("Top 5 differences:", top_5_diff)
    print("Lowest 5 differences:", lowest_5_diff)

    return mean_diff, std_diff, top_5_diff, lowest_5_diff

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    get_features()