## Type 2. Phase Vocoder (looks for beats)
# A phase vocoder is a time-domain technique that allows for independent control of time and frequency.
import numpy as np
from scipy.interpolate import interp1d
from Main_Pipeline import Alignment
from scipy import ndimage


class Variable_phasevocoder(Alignment):
    
    def __init__(
            self,
            magnitude,
            phase,
            warp_curve,
            t_dense,
            sample_rate,
            hop_size=512,
            fft_size=2048):

        """
        Initialize the phase vocoder.

        Parameters
        ----------
        magnitude:
            STFT magnitude matrix

            Shape:
            (number_of_frames, fft_bins)


        phase:
            STFT phase matrix

            Shape:
            (number_of_frames, fft_bins)


        warp_curve:
            T(t)

            Maps original time -> stretched time


        t_dense:
            Time values corresponding to warp_curve


        sample_rate:
            Audio sampling frequency


        hop_size:
            Distance between STFT frames


        fft_size:
            FFT size
        """


        # Store STFT magnitude
        self.mag = magnitude


        # Store original STFT phase
        self.phase = phase


        # Store time warping function T(t)
        self.warp_curve = warp_curve


        # Store original time axis
        self.t_dense = t_dense


        # Audio sample rate
        self.sr = sample_rate


        # STFT parameters
        self.H = hop_size
        self.N = fft_size


        # Number of frequency bins
        self.num_bins = magnitude.shape[1]


    # Create frame level time mapping (Converting seconds to frames)
    def create_analysis_times(self):

        """
        Create the input frame location for every
        output frame.

        Output:
        
        analysis_times[i]

        means:

        "For output frame i,
         read from this fractional input frame."
        """


        # Time distance between STFT frames
        frame_time = self.H / self.sr


        # Total stretched duration
        output_duration = self.warp_curve[-1]


        # Number of output frames after stretching
        num_output_frames = int(
            output_duration / frame_time
        )


        # Generate output frame timestamps
        output_times = np.arange(
            num_output_frames
        ) * frame_time



        # We currently have:
        #
        # original time --> warped time
        #
        # Need inverse:
        #
        # warped time --> original time
        #
        inverse_warp = interp1d(
            self.warp_curve,
            self.t_dense,
            fill_value="extrapolate"
        )


        # Find original time for every output frame
        original_times = inverse_warp(
            output_times
        )



        # Convert seconds into STFT frame indexes
        self.analysis_times = (
            original_times / frame_time
        )

    # Magnitude interpolation 
    def interpolate_magnitude(self):

        """
        Interpolate magnitude spectrum
        at fractional frame positions.
        """


        output_mag = []


        for position in self.analysis_times:

            position = np.clip(position, 0, self.mag.shape[0] - 1) # prevent out of bound


            # Left neighboring frame
            left = int(
                np.floor(position)
            )


            # Right neighboring frame
            right = min(
                left + 1,
                self.mag.shape[0]-1
            )


            # Fraction between frames
            alpha = position - left



            # Linear interpolation
            #
            # Example:
            #
            # position = 10.3
            #
            # 70% frame 10
            # 30% frame 11

            magnitude = (
                (1-alpha) * self.mag[left]
                +
                alpha * self.mag[right]
            )


            output_mag.append(
                magnitude
            )



        self.output_mag = np.array(
            output_mag
        )


    def propagate_phase(self):

        """
        Calculate new phase values
        while preserving frequency continuity.
        """


        # Create output phase container

        output_phase = np.zeros_like(
            self.output_mag
        )


        # First frame keeps original phase

        output_phase[0] = self.phase[0]



        # Frequency of every FFT bin

        omega = (
            2*np.pi
            *
            np.arange(self.num_bins)
            /
            self.N
        )



        for i in range(
            1,
            len(self.analysis_times)
        ):


            # Current fractional frame

            # Bug fixed: no bounds clipping 
            current = np.clip(self.analysis_times[i], 0, self.mag.shape[0] - 1)
            
            previous = np.clip(self.analysis_times[i - 1], 0, self.mag.shape[0] - 1)



            # Find neighboring frames

            prev_frame = int(
                np.floor(previous)
            )

            curr_frame = int(
                np.floor(current)
            )


            # Original phase difference

            delta_phase = (
                self.phase[curr_frame]
                -
                self.phase[prev_frame]
            )


            # Bug fixed:
            frame_diff = curr_frame - prev_frame


            # Remove expected FFT rotation

            delta_phase -= (
                omega * self.H * frame_diff
            )



            # Wrap phase to [-pi, pi]

            delta_phase = (
                (delta_phase + np.pi) % (2 * np.pi) - np.pi)



            # Restore true movement

            true_increment = (
                omega * self.H * frame_diff + delta_phase
            )



            # Accumulate phase

            output_phase[i] = (
                output_phase[i-1]
                +
                true_increment
            )



        self.output_phase = output_phase

    # Buld a complex spectrum
    def build_spectrum(self):

        """
        Combine magnitude and phase
        into complex FFT values.
        """


        self.complex_spec = (
            self.output_mag
            *
            np.exp(
                1j*self.output_phase
            )
        )

    # Inverse STFT
    def inverse_stft(self, window):

        """
        Convert frequency domain back
        into waveform using overlap-add.
        """


        num_frames = (
            self.complex_spec.shape[0]
        )


        # Output length calculation

        output_length = (
            (num_frames-1)
            *
            self.H
            +
            self.N
        )


        output = np.zeros(
            output_length
        )



        for i in range(num_frames):


            # Convert spectrum -> waveform

            frame = np.fft.irfft(
                self.complex_spec[i],
                n=self.N
            )


            # Apply window

            frame *= window



            # Location in output

            start = i*self.H



            # Overlap add

            output[
                start:start+self.N
            ] += frame



        return output
    
    def process_magnitudes_only(self):
        """
        Runs only the magnitude pipeline. 
        Use this when you only need `output_mag` for feature analysis!
        """
        self.create_analysis_times()
        self.interpolate_magnitude()
        return self.output_mag
    
    # Run Everything?
    def process(self, window):

        """
        Complete variable phase vocoder pipeline.
        """


        # Step 1:
        # Continuous T(t)
        # ->
        # frame mapping

        self.create_analysis_times()



        # Step 2:
        # interpolate magnitude

        self.interpolate_magnitude()



        # Step 3:
        # propagate phase

        self.propagate_phase()



        # Step 4:
        # create complex STFT

        self.build_spectrum()



        # Step 5:
        # reconstruct waveform

        audio_A_phase_vocoder = self.inverse_stft(
            window
        )


        return audio_A_phase_vocoder
    

    def transient_mask_from_vocoder(vocoder, alpha=2.0, hold_ms=30):
        """
        Computes a proper spectral-flux transient mask from the phase vocoder's
        interpolated magnitude spectra (output_mag), BEFORE any reduction to
        peak-frequency-per-frame happens.

        Parameters
        ----------
        vocoder : Variable_phasevocoder
            Must have .process_magnitudes_only() called, or .process() already run
            (either way, self.output_mag must exist as shape (frames, bins)).
        alpha : float
            Threshold sensitivity (higher = fewer frames flagged as transient).
        hold_ms : float
            Minimum duration a transient stays flagged, to avoid flicker.

        Returns
        -------
        final_mask : np.ndarray, shape (frames,)
            1 = transient frame, 0 = steady frame.
        sf : np.ndarray, shape (frames,)
            Raw spectral flux value per frame (for inspection/debugging).
        """

        # 1. Make sure output_mag exists and is (frames, bins)
        if not hasattr(vocoder, "output_mag"):
            raise ValueError("Call vocoder.process_magnitudes_only() or .process() first.")

        output_mag = vocoder.output_mag   # shape: (frames, bins)

        # 2. Spectral flux: positive energy INCREASE frame-to-frame, summed across bins
        #    axis=0 is the frame axis here (opposite of librosa's raw stft convention)
        sf = np.sum(np.maximum(0, np.diff(output_mag, axis=0)), axis=1)

    
        sf = np.pad(sf, (1, 0), mode='edge')  # realign length to match output_mag.shape[0]

        # 3. Adaptive threshold: rolling median + alpha * rolling std
        win_len = 15  # ~150ms window at typical hop sizes; tune if needed

        local_median = ndimage.median_filter(sf, size=win_len)
        local_mean   = ndimage.uniform_filter(sf, size=win_len)
        local_var    = ndimage.uniform_filter((sf - local_mean) ** 2, size=win_len)
        local_std    = np.sqrt(np.maximum(0, local_var))

        threshold = local_median + alpha * local_std

        # 4. Raw binary decision
        raw_mask = (sf > threshold).astype(int)

        # 5. Hold window so transient tails don't flicker
        frame_rate = vocoder.sr / vocoder.H
        hold_frames = max(1, int((hold_ms / 1000.0) * frame_rate))
        final_mask_vocoder = ndimage.maximum_filter1d(raw_mask, size=hold_frames, origin=-hold_frames // 2)
        

        return final_mask_vocoder, sf, threshold