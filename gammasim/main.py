from gammasim import GammaSim
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv)>1:
    argument=sys.argv[1]
else:
    argument=1
config=f"config_method{argument}.json"

print(f"using method with configuration: {config}")

a = GammaSim(config)

a.generate_dataset(False)

a.plot_wf()

x = a.get_dataset()
y = x.shape[0]
dd = a.get_sampling_time()
# Perform the FFT
# X_fft = np.fft.fft(x)
# frequencies = np.fft.fftfreq(y, 1 / dd)

# # Take the magnitude (absolute value) of the FFT result
# X_magnitude = np.abs(X_fft)

# # Plot the FFT result (only positive frequencies)
# plt.plot(frequencies[:y // 2], X_magnitude[:y // 2])  # Plot only the positive frequencies
# plt.title("FFT of the time series")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.grid(True)
# plt.show()