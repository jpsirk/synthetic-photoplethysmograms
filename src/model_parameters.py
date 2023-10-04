from dataclasses import dataclass, field
import copy
import numpy as np


def default_field(obj):
    """Simple function for setting a default ndarray value
    for a dataclass attribute."""
    return field(default_factory=lambda: np.array(copy.deepcopy(obj)))


@dataclass
class ModelParameters:
    """
    A class for synthetic PPG model parameters.

    Attributes
    ----------
    n : int
        Number of signals to generate.
    fs : int
        Sampling frequency [Hz].
    s_len : int
        Length of the generated signals [samples].
    label_width : int
        Label width. NOTE: Must be odd.
    n_pulse_widths : int
        Number of pulse widths to create. Set this to a fairly large 
        number relative to the s_length in order to have enough signal 
        length for cropping.
    gaus_locs_fitted : list
        Fitted waveform values for the Gaussian bump locations.
    gaus_widths_fitted : list
        Fitted waveform values for the Gaussian bump widths.
    gaus_amps_fitted : list
        Fitted waveform values for the Gaussian bump amplitudes.
    gaus_loc_bounds : list
        Randomization lower and upper bounds for the Gaussian bump locations.
    gaus_width_bounds : list
        Randomization lower and upper bounds for the Gaussian bump widths.
    gaus_amp_bounds : list
        Randomization lower and upper bounds for the Gaussian bump amplitudes.
    pulse_width_bounds : list
        Bounds for pulse waveform widths, i.e. heart rate bounds.
    breathing_freq_bounds : list
        Bounds for breathing frequency and amplitudes.
    noise_amp_bounds : list
        Bounds for PSD-based noise.
    breathing_coupling_coeff : float
        Breathing coupling coefficient.
    psd_interpolator_filenames : list
        Noise PSD interpolator filenames.
    channel_names : list
        Wearable device channel names.
    channels_len : int
        Number of wearable device channels.
    sos : list
        SOS for 2nd order Butterworth bandpass filter with cutoff frequencies of 0.5 Hz and 5.0 Hz.
    sos_zi : list
        Initial filter conditions.
    """
    n: int = 200_000
    fs: int = 100
    s_len: int = 400
    label_width: int = 5
    n_pulse_widths: int = 20
    gaus_locs_fitted: np.ndarray = default_field([-1.80921309, 0.81512224])
    gaus_widths_fitted: np.ndarray = default_field([0.68435229, 1.88599918])
    gaus_amps_fitted: np.ndarray = default_field([8.34987248, 9.68538241])
    gaus_loc_bounds: np.ndarray = default_field([[-2.0, -1.4], [0.4, 1.0]])
    gaus_width_bounds: np.ndarray = default_field([[0.5, 0.9], [1.7, 2.1]])
    gaus_amp_bounds: np.ndarray = default_field([[5, 10], [5, 9]])
    pulse_width_bounds: np.ndarray = default_field([0.4, 1.3])
    breathing_freq_bounds: np.ndarray = default_field([0.15, 0.4])
    noise_amp_bounds: np.ndarray = default_field([0, 1.5])
    breathing_coupling_coeff: float = 0.1
    psd_interpolator_filenames: np.ndarray = default_field([
        '../data/psd_interpolator_sitting.npy', '../data/psd_interpolator_walking.npy',
        '../data/psd_interpolator_hand_movement.npy'])
    channel_names: np.ndarray = default_field(
        ['Green PPG', 'Red PPG', 'Infrared PPG', 'ECG', 'NN model input', 'NN model output'])
    channels_len = 6
    sos: np.ndarray = default_field([[0.016581931669303045, 0.03316386333860609, 0.016581931669303045, 1.0, -1.6288507685741764, 0.6994634776474277],
                                     [1.0, -2.0, 1.0, 1.0, -1.9573890395427322, 0.9585316851827481]])
    sos_zi: np.ndarray = default_field([[0.9227351905275814, -0.6404360892363036],
                                        [-0.9393171221969531, 0.9393171221969502]])
