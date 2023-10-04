import numpy as np
from scipy import integrate, signal, fft
import os
import concurrent.futures
from copy import deepcopy
from timeit import default_timer as timer
from src import utils


def gen_ppg(gaus_locs, gaus_widths, gaus_amps, pulse_widths, fs=100,
            label_width=5, s_length=None, clean_labels=False,
            derivative_smoothing=True):
    """Generates a clean synthetic PPG signal.

    A PPG signal is generated using a simple parametric model. Label lists
    with feet and peaks marked with ones are also returned along with
    the signal.

    Parameters
    ----------
    gaus_locs : list
        Locations of the Gaussian bumps with respect to the middle 
        point of a pulse waveform. The unit is radians, assuming the first 
        half of a pulse waveform is between [-PI, 0] and the latter half 
        between [0, PI].
    gaus_widths : list
        Widths (sigmas) of the Gaussian bumps, assumed to be > 0.
    gaus_amps : list
        Amplitudes of the Gaussian bumps.
    pulse_widths : list
        An array of samples for each pulse waveform.
    fs : int
        Sampling frequency [Hz].
    label_width : int
        The number of ones for each foot/peak.
    s_length : int
        The length of the final signal [samples].
    clean_labels: bool
        If true, the beginnings and endings of the labels are cleaned.
    derivative_smoothing : bool
        If true, the derivative signal is smoothed with a Savitzky-Golay
        filter before its numerically integrated.

    Returns
    -------
    synt : list
        Generated synthetic PPG signal.
    label_feet : list
        Generated label based on the pulse waveform feet.
    label_peaks : list
        Generated label based on the pulse waveform peaks.
    der : list
        Generated smooth derivative signal.
    der_raw : list
        Generated raw derivative signal.
    """

    # Check parameters.
    if label_width % 2 != 1:
        raise ValueError(f'Label width, {label_width}, must be odd.')
    if s_length is not None and (np.sum(pulse_widths) < (3 * fs + s_length)):
        min_len = 3 * fs + s_length
        raise ValueError(f"""Provide longer pulse_width array. The length must
            be at least: {min_len}. Now the size is: {np.sum(pulse_widths)}""")

    # Cumulative sum of pulse widths.
    pulse_widths_cumsum = np.zeros(len(pulse_widths) + 1, dtype=int)
    pulse_widths_cumsum[0] = 0
    pulse_widths_cumsum[1:] = np.cumsum(pulse_widths)

    # Derivatives.
    n_gaus = len(gaus_locs)
    s_len = np.sum(pulse_widths)
    ders = np.zeros((n_gaus, s_len))
    for i in range(n_gaus):
        # Phase signal.
        phase = np.zeros(s_len)
        for j in range(1, pulse_widths_cumsum.size):
            phase[pulse_widths_cumsum[j - 1]:pulse_widths_cumsum[j]] = \
                np.roll(np.linspace(-np.pi, np.pi, pulse_widths[j - 1]),
                        int(gaus_locs[i] / (2 * np.pi) * pulse_widths[j - 1]))
        # Derivative.
        ders[i] = -phase / (gaus_widths[i]**2) * gaus_amps[i] * \
            np.exp(-phase**2 / (2 * gaus_widths[i] ** 2))

    # The final derivative of the signal is the sum of the derivatives.
    der_raw = np.sum(ders, axis=0)

    # Smoothen the derivative signal.
    der = signal.savgol_filter(der_raw, 11, 2) if derivative_smoothing \
        else der_raw

    # Compute the final synthetic signal with numerical integration.
    synt = integrate.cumtrapz(der, dx=1/fs, initial=0)

    # Create the labels.
    # The feet are close to the beginnings and endings of pulse widths.
    # The peaks follow the feet --> use a threshold for the distance.
    feet, peaks = [], []
    for idx in pulse_widths_cumsum:
        # Index range for finding the foot.
        i1 = max(0, int(idx - 0.1 * fs))
        i2 = min(s_len, int(idx + 0.1 * fs))
        f = i1 + np.argmin(synt[i1:i2])
        feet.append(f)
        # End index of the range for finding the peak.
        i2 = min(s_len, int(f + 0.4 * fs))
        # Find the relative maxima.
        arg_rel_maxs = signal.argrelmax(synt[f:i2])[0]
        if len(arg_rel_maxs) > 0:
            peaks.append(f + arg_rel_maxs[0])

    feet, peaks = np.array(feet), np.array(peaks)

    # Create label signals based on the feet and the peaks.
    def create_label(size, locations):
        label = np.zeros(size)
        for l in locations:
            i1 = max(0, l - label_width // 2)
            i2 = min(size, l + label_width // 2 + 1)
            label[i1:i2] = 1
        return label

    label_feet = create_label(synt.size, feet)
    label_peaks = create_label(synt.size, peaks)

    # Cropping.
    if s_length is not None:
        # Random starting index so that the beginning of a signal is not always
        # the beginning of a pulse waveform.
        crop_start = np.random.randint(0, fs * 3)
        crop_end = crop_start + s_length
        synt = synt[crop_start:crop_end]
        label_feet = label_feet[crop_start:crop_end]
        label_peaks = label_peaks[crop_start:crop_end]
        der = der[crop_start:crop_end]
        der_raw = der_raw[crop_start:crop_end]
        feet = feet[np.logical_and(feet >= crop_start, feet <= crop_end)] \
            - crop_start
        peaks = peaks[np.logical_and(peaks >= crop_start, peaks <= crop_end)] \
            - crop_start

    # Clean the labels from the beginning and end if the label_width condition
    # is not fulfilled.
    if clean_labels:
        def clean_label(label):
            if sum(label[:label_width]) < label_width:
                label[:label_width] = 0
            if sum(label[-label_width:]) < label_width:
                label[-label_width:] = 0

        clean_label(label_feet)
        clean_label(label_peaks)

    # Normalization.
    synt = utils.min_max_normalize(synt, 0, 1)

    return synt, label_feet, label_peaks, der, der_raw


def gen_pulse_widths(n, mu=0.9, breathing_freq=0.2,
                     breathing_coupling_coeff=0.1, fs=100):
    """Generates an array of pulse waveform widths [samples].

    Parameters
    ----------
    n : int
        Number of pulse waveforms.
    mu : float
        Mean pulse waveform width.
    breathing_freq : float
        Breathing frequency [Hz].
    breathing_coupling_coeff : float
        Breathing coupling coefficient.
    fs : int
        Sampling frequency [Hz].

    Returns
    -------
    pulse_widths : list
        A list of pulse waveforms widths.
    """

    pulse_widths = np.zeros(n)
    for i in range(n):
        pulse_widths[i] = fs * (mu + breathing_coupling_coeff *
                                np.sin(2 * np.pi * breathing_freq * np.sum(pulse_widths / fs)))

    return pulse_widths.round(0).astype(int)


def psd_to_time(n, fs, psd_interp, amp=None):
    """Generates time domain realization from a power spectral density.

    Parameters
    ----------
    n : int
        Length (in samples) of the desired signal.
    fs : int
        Sampling frequency.
    psd_interp : fun
        Interpolation function for computing a PSD from a frequency array.
    amp : float
        Amplitude modifier.

    Returns
    -------
    y : list
        Generated time series.
    freqs : list
        PSD frequencies.
    psd : list
        The used PSD.
    """
    # Frequencies.
    freqs = np.linspace(fs / n, fs / 2, n // 2)

    # Power spectral density.
    psd = psd_interp(freqs)

    # Random numbers for the real and imaginary parts.
    r_part = np.sqrt(psd / 2) * \
        (np.random.randn(len(freqs)) + 1.j * np.random.randn(len(freqs)))
    i_part = np.sqrt(psd / 2) * \
        (np.random.randn(len(freqs)) + 1.j * np.random.randn(len(freqs)))

    # Add zero to the beginning to be inline with SciPy's IFFT.
    r_part = np.insert(r_part, 0, 0)

    # Fourier frequency terms.
    w = np.concatenate((r_part, np.conj(i_part)[::-1]))

    # Timeseries.
    y = np.sqrt(fs) * np.real(fft.ifft(w, n, norm='ortho'))

    # Normalize and multiply by the amplitude factor.
    if amp is not None:
        y = amp * utils.min_max_normalize(y, 0, 1)

    return y, freqs, psd


def gen_rand_ppg_signals(model_params):
    """Generates random PPG signals.

    Parameters
    ----------
    model_params : ModelConfig
        ModelConfig object defining the model parameters.

    Returns
    -------
    synts : list[list]
        A 2D list where each row is a generated synthetic PPG signal.
    baselines : list[list]
        A 2D list where each row is a generated baseline wander signal.
    white_noises : list[list]
        A 2D list where each row is white noise signal.
    noises : list[list]
        A 2D list where each row is a noise signal.
    labels_feet : list[list]
        A 2D list where each row is a signal containing ones for waveform 
        feet positions.
    labels_peaks : list[list]
        A 2D list where each row is a signal containing ones for waveform 
        peak positions.    
    """
    # Array of random mean pulse widths.
    pulse_widths = np.random.uniform(model_params.pulse_width_bounds[0],
                                     model_params.pulse_width_bounds[1], model_params.n)
    # Array of random breathing frequencies and amplitudes.
    breathing_freqs = np.random.uniform(model_params.breathing_freq_bounds[0],
                                        model_params.breathing_freq_bounds[1], model_params.n)
    # Array of random noise amplitudes.
    noise_amps = np.random.uniform(model_params.noise_amp_bounds[0],
                                   model_params.noise_amp_bounds[1], model_params.n)
    # Arrays of PSD interpolator functions.
    psd_interpolators = [np.load(fn, allow_pickle=True)[()]
                         for fn in model_params.psd_interpolator_filenames]

    # Arrays for storing the generated data.
    synts = np.zeros((model_params.n, model_params.s_len))
    labels_feet = np.zeros((model_params.n, model_params.s_len))
    labels_peaks = np.zeros((model_params.n, model_params.s_len))
    noises = np.zeros((model_params.n, model_params.s_len))
    waveform_shifts = np.random.rand(model_params.n)
    for i in range(model_params.n):
        # Generate pulse widths.
        # NOTE: Generate enough of them to have spare length for signal
        # cropping.
        pws = gen_pulse_widths(model_params.n_pulse_widths, pulse_widths[i],
                               breathing_freqs[i], model_params.breathing_coupling_coeff,
                               model_params.fs)
        # Generate random Gaussian function parameters.
        gaus_locs = [item[0] + waveform_shifts[i] * (item[1] - item[0])
                     for item in model_params.gaus_loc_bounds]
        gaus_widths = [item[0] + waveform_shifts[i] * (item[1] - item[0])
                       for item in model_params.gaus_width_bounds]
        gaus_amps = [item[0] + waveform_shifts[i] * (item[1] - item[0])
                     for item in model_params.gaus_amp_bounds]
        # Generate synthetic PPG signal.
        synts[i], labels_feet[i], labels_peaks[i], _, _ = \
            gen_ppg(gaus_locs, gaus_widths, gaus_amps, pws, model_params.fs,
                    model_params.label_width, model_params.s_len, False)
        # Generate random noise.
        psd_interp = psd_interpolators[np.random.randint(
            0, len(model_params.psd_interpolator_filenames))]
        noises[i], _, _ = psd_to_time(model_params.s_len, model_params.fs,
                                      psd_interp, noise_amps[i])

    return synts, labels_feet, labels_peaks, noises


def gen_rand_ppg_signals_parallel(model_params):
    """Generates random PPG signals using parallel computing.

    Parameters
    ----------
    model_params : ModelConfig
        ModelConfig object defining the model parameters.

    Returns
    -------
    results : list[list[list]]
        A list of 2D lists where each 2D list corresponds to the return
        value of the function gen_rand_ppg_signals.
    """
    # Record the time it takes to run the function.
    start_time = timer()

    # Use all logical processors.
    workers_count = os.cpu_count()

    # Create an array of batch sizes for each processor.
    batches = [model_params.n // workers_count] * workers_count
    batches[-1] += model_params.n - np.sum(batches)

    # Initialize empty lists to hold the results.
    results = [[], [], [], []]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_count) \
            as executor:
        # Store the futures into a dict --> Completed futures can then be
        # deleted to release RAM.
        futures = {}
        for b in batches:
            params = deepcopy(model_params)
            params.n = b
            f = executor.submit(gen_rand_ppg_signals, params)
            futures[f] = f

        futures_count = len(futures)
        futures_completed_count = 0
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            futures_completed_count += 1
            if futures_completed_count == futures_count:
                print('All futures completed.')
            else:
                print(
                    f'Future {futures_completed_count}/{futures_count} completed...      ', end='\r', flush=True)

            try:
                # Get results.
                res = f.result()
            except Exception as e:
                print(f'''Exception occurred while trying to get future\'s 
                    result: {e}''')
            else:
                for i in range(len(results)):
                    results[i].append(res[i])

            del futures[f]

    # Concatenate the results into arrays of shape (n, s_length).
    for i in range(len(results)):
        results[i] = np.concatenate(results[i], axis=0)

    time_elapsed = timer() - start_time
    print(f'Time elapsed: {int(time_elapsed // 60)} m '
          f'{round(time_elapsed % 60, 3)} s')

    return results
