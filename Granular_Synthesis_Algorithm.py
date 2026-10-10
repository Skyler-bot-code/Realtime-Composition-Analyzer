from Main_Pipeline import Alignment
import numpy as np
import librosa
from scipy import ndimage


"""Digital Signal Processing Algorithm (DSP)"""
# This algorithm focuses on manipulating the digital signal directly for time-stretching and pitch-shifting.


## Type 1. Granular Synthesis
# Granular synthesis involves breaking the audio into small grains and manipulating their playback.



# Repeating grains (small pieces of sound) to stretch the audio without changing the pitch.
# Jumping (Stretching) from a pitch of 0.9 to -0.8 creates a strong click sound, which we don't want 

# Solution for jumping: We overlap 

# grain_size = every something frequency 

from scipy.interpolate import interp1d

class Granular_synthesis(Alignment):

    def __init__(self, importer_A, ratio_curve, t_dense):
        # NOTE: no super().__init__() call — Alignment.__init__ requires
        # 4 unrelated args and doesn't provide self.importer_A anyway.
        self.importer_A = importer_A
        self.ratio_curve_array = ratio_curve
        self.t_dense = t_dense

        self.warp_curve = interp1d(
            t_dense, ratio_curve,
            bounds_error=False,
            fill_value=(ratio_curve[0], ratio_curve[-1])
        )

        self.grain_size = 1024
        self.analysis_hop = 512
        self.search_radius = 100
        self.sr = self.importer_A.sr
        self.window = np.hanning(self.grain_size)


    def allocate_large(self):
        audio = self.importer_A.audio
        grain_size = self.grain_size
        analysis_hop = self.analysis_hop
        search_radius = self.search_radius
        window = self.window
        sr = self.sr

        est_max_samples = int(len(audio) * 4)
        output = np.zeros(est_max_samples, dtype=np.float64)
        norm_buffer = np.zeros(est_max_samples, dtype=np.float64)

        output[:grain_size] = audio[:grain_size] * window
        norm_buffer[:grain_size] += window

        input_pos = analysis_hop
        output_pos = analysis_hop
        prev_input_pos = 0

        while input_pos + grain_size + search_radius < len(audio):
            current_time = input_pos / sr
            ratio = max(0.25, min(float(self.warp_curve(current_time)), 4.0))


            # FIX 2: floor derived from grain_size/analysis_hop, not a fixed 0.25
            # ensures synthesis_hop never exceeds grain_size (no gap dropouts)
            ratio = max(analysis_hop / grain_size, min(ratio, 4.0))

            synthesis_hop = max(64, int(analysis_hop / ratio))

            if output_pos + grain_size >= len(output):
                break

            ref_start = prev_input_pos + analysis_hop
            if ref_start + grain_size > len(audio):
                break
            reference_grain = audio[ref_start: ref_start + grain_size]

            best_score = -np.inf
            best_shift = 0
            for shift in range(-search_radius, search_radius + 1):
                start = input_pos + shift
                if start < 0 or start + grain_size > len(audio):
                    continue
                candidate = audio[start: start + grain_size]
                denom = np.linalg.norm(candidate) + 1e-9
                score = np.dot(reference_grain, candidate) / denom
                if score > best_score:
                    best_score = score
                    best_shift = shift

            actual_input_pos = input_pos + best_shift
            grain = audio[actual_input_pos: actual_input_pos + grain_size] * window

            output[output_pos: output_pos + grain_size] += grain
            norm_buffer[output_pos: output_pos + grain_size] += window

            prev_input_pos = actual_input_pos
            input_pos += analysis_hop
            output_pos += synthesis_hop

        valid_idx = norm_buffer > 1e-6
        output[valid_idx] /= norm_buffer[valid_idx]

        nonzero = np.where(norm_buffer > 0)[0]
        if len(nonzero) == 0:
            return output[:0], sr
        last_sample = nonzero.max() + 1
        audio_A_granular = output[:last_sample]

        # store so transient_mask_from_granular can read them off self
        self.audio_A_granular = audio_A_granular

        return audio_A_granular, sr
    


    def transient_mask_from_granular(self, hop_length=512, n_fft=2048, alpha=2.0, hold_ms=30):
        """
        Computes spectral-flux transient mask on the rendered granular output.
        This is the only option here — Granular_synthesis has no intermediate
        magnitude-matrix stage, unlike the phase vocoder's process_magnitudes_only().
        """
        audio_A_granular = self.audio_A_granular
        sr = self.sr

        stft = np.abs(librosa.stft(audio_A_granular, n_fft=n_fft, hop_length=hop_length))  # shape: (bins, frames)

        sf = np.sum(np.maximum(0, np.diff(stft, axis=1)), axis=0)   # axis=1 is the frame axis here
        sf = np.pad(sf, (1, 0), mode='edge')

        win_len = 15
        local_median = ndimage.median_filter(sf, size=win_len)
        local_mean   = ndimage.uniform_filter(sf, size=win_len)
        local_var    = ndimage.uniform_filter((sf - local_mean) ** 2, size=win_len)
        local_std    = np.sqrt(np.maximum(0, local_var))
        threshold    = local_median + alpha * local_std

        raw_mask = (sf > threshold).astype(int)
        frame_rate = sr / hop_length
        hold_frames = max(1, int((hold_ms / 1000.0) * frame_rate))
        final_mask_granular = ndimage.maximum_filter1d(raw_mask, size=hold_frames, origin=-hold_frames // 2)

        return final_mask_granular, sf, threshold

        #sf.write("result_synced.wav", audio_A_granular, sr)
        #print(f"Successfully rendered synced audio! Output duration: {len(audio_A_granular)/sr:.2f}s")



