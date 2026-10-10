import librosa as lr
import numpy as np


class AudioImporter:

    def __init__(self, file):
        self.file = file

        # Settings
        self.window_size = 2048 # based on this industrial standard window size, make a STFT model (ML_STFT_parameter.py)
        self.hop_size = 512 # based on this industrial standard hop size, make a STFT model (ML_STFT_parameter.py)

        self.audio = None
        self.sr = None
        self.frequencies = []
        self.times = []

        self.peak_amplitudes = []  # Peak amplitude of this exact frame
        self.rms_amplitudes = []  # RMS amplitude of this exact frame   


    def load_audio(self):
        self.audio, self.sr = lr.load(self.file, sr=None)
        return self.audio, self.sr


    def detect_pitch(self):
        if self.audio is None:
            raise ValueError("Audio not loaded. Please run load_audio() first.")

        # Clear lists to avoid appending duplicates if run twice
        self.times.clear()
        self.frequencies.clear()
        self.peak_amplitudes.clear()
        self.rms_amplitudes.clear()

        for i in range(
            0,
            len(self.audio) - self.window_size,
            self.hop_size
        ):

            # Take a small audio frame
            frame = self.audio[i:i + self.window_size]


            # Detect pitch
            pitch = lr.yin(
                frame,
                fmin=50,
                fmax=1000,
                sr=self.sr
            )


            # Average frequency of this frame
            frequency = np.mean(pitch)


            # Calculate time
            time = i / self.sr

            # Calculate amplitude
            peak = np.max(np.abs(frame))            # Momentary peak volume
            rms = np.sqrt(np.mean(frame**2))        # Momentary RMS volume


            # Store results
            self.times.append(time)
            self.frequencies.append(frequency)
            self.peak_amplitudes.append(peak)
            self.rms_amplitudes.append(rms)


            print(
                f"Time: {time:.5f}s, "
                f"Frequency: {frequency:.5f} Hz, "
                f"Peak Amplitude: {peak:.5f}, "
                f"RMS Amplitude: {rms:.5f}"
            )
            


        return self.times, self.frequencies, self.peak_amplitudes, self.rms_amplitudes
    
    def detect_time_frequency(self):
        if self.audio is None:
            raise ValueError("Audio not loaded. Please run load_audio() first.")

        # Clear lists to avoid appending duplicates if run twice
        self.times.clear()
        self.frequencies.clear()
        

        for i in range(
            0,
            len(self.audio) - self.window_size,
            self.hop_size
        ):

            # Take a small audio frame
            frame = self.audio[i:i + self.window_size]


            # Detect pitch
            pitch = lr.yin(
                frame,
                fmin=50,
                fmax=1000,
                sr=self.sr
            )


            # Average frequency of this frame
            frequency = np.mean(pitch)


            # Calculate time
            time = i / self.sr

           


            # Store results
            self.times.append(time)
            self.frequencies.append(frequency)
           


            print(
                f"Time: {time:.5f}s, "
                f"Frequency: {frequency:.5f} Hz, "
            
            )

        return self.times, self.frequencies
    
    def detect_rms(self):
        if self.audio is None:
            raise ValueError("Audio not loaded. Please run load_audio() first.")

        self.times.clear()
        self.rms_amplitudes.clear()

        for i in range(0, len(self.audio) - self.window_size, self.hop_size):
            frame = self.audio[i:i + self.window_size]

            time = i / self.sr
            rms = np.sqrt(np.mean(frame ** 2))

            self.times.append(time)
            self.rms_amplitudes.append(rms)

        return self.times, self.rms_amplitudes



