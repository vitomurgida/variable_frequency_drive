def compute_fft(t, x, thr, freq_max_plot, y_axis, units, title):

    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib.pyplot as plt

# INPUTS DESCRIPTION
# t = time vector
# x = signal
# thr = threshold for peak detection as scaling factor of max amplitude harmonic
# freq_max_plot = the maximum frequency to visualize in the spectrum (doesn't affect the analyses, only the plot)
# y_axis = title of the vertical axis of the plots, string
# units = units of the vertical axis, string
# title = title of the plot, string

    # Sampling frequency
    fs = 1/(t[1] - t[0])

    # Windowing (Hanning)
    window = np.hanning(len(x))
    x_win = x * window

    # Window correction factor (coherent gain)
    U = np.sum(window) / len(window)

    # FFT
    N = len(x)
    X = np.fft.rfft(x_win) # numpy FFT gives unnormalized real FFT, so need to rescale by N
    freqs = np.fft.rfftfreq(N, 1/fs)

    # Amplitude scaling
    amp = (2 / (N * U)) * np.abs(X) # N needed to normalize FFT, U to correct for windowing reduction of signal power, 2 needed to recover negative side neglected by np.ff.rfft
    amp[0] = amp[0] / 2
    rms = amp / np.sqrt(2)

    # PPeak detection
    threshold = np.max(amp) * thr
    peaks, _ = find_peaks(amp, height=threshold)

    print("\nDetected FFT Peaks:")
    print("----------------------------------------")
    print("Freq [Hz]    Amplitude    RMS")
    print("----------------------------------------")

    for p in peaks:
        print(f"{freqs[p]:8.2f}    {amp[p]:10.4f}    {rms[p]:10.4f}")

    # Plotting
    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(freqs, amp)
    plt.plot(freqs[peaks], amp[peaks], "ro")
    plt.title("FFT Amplitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(y_axis+" "+units)
    plt.grid()
    plt.xlim(0, freq_max_plot)
    plt.title(title)

    plt.subplot(2,1,2)
    plt.plot(t, x)
    plt.title("signal" )
    plt.xlabel("time [s]")
    plt.ylabel(y_axis+" "+units)
    plt.grid()
    plt.title(title)

    plt.tight_layout()
    plt.show()

    return freqs, amp, peaks, fs
