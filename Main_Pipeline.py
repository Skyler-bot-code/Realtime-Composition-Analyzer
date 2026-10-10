from Import_Audio import AudioImporter
import numpy as np
import librosa.sequence
import librosa 
from librosa.sequence import dtw
import scipy.stats as stats 
import math 
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
from collections import deque
import soundfile as sf
from scipy.interpolate import interp1d
from Pitch_Analysis import Intonation_Analysis
import scipy.ndimage as ndimage
import librosa
from Variable_Phase_Vocoder_Algorithm import Variable_phasevocoder
from Granular_Synthesis_Algorithm import Granular_synthesis
from Pitch_Analysis import Intonation_Analysis


# Use Multivariate Dynamic Time Warping (DTW) 
#Importer A should come from double_notes_multiple_instruments based on users' choice 

importer_A = AudioImporter("/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav") # from computer
importer_B = AudioImporter("/Users/skyler/Desktop/audio.wav") # from youtube

print("Processing Audio File A...")
importer_A.load_audio()
importer_A.detect_pitch() 
times_A, frequencies_A, peak_amplitudes_A, rms_amplitudes_A = importer_A.detect_pitch() # This fills importer_A with data

print("Processing Audio File B...")
importer_B.load_audio()
importer_B.detect_pitch() 
times_B, frequencies_B, peak_amplitudes_B, rms_amplitudes_B = importer_B.detect_pitch() # This fills importer_B with data


class Alignment(AudioImporter):

    def __init__(self, importer_A_times, importer_A_frequency, importer_B_times, importer_B_frequency):
        self.importer_A_times = importer_A_times
        self.importer_A_frequencies = importer_A_frequency
        self.importer_B_times = importer_B_times
        self.importer_B_frequencies = importer_B_frequency


    """ Make sure both audio starts at the same note within similar frequency range (within 5 Hz)"""
    def Same_start_time(importer_A, importer_B, importer_A_frequency, importer_B_frequency, importer_A_times, importer_B_times, importer_A_rms, importer_B_rms,a_peak, b_peak):
        error = 5 # (Adapative_STFT_Algorithm)

        importer_A_times, importer_A_frequency, a_peak, importer_A_rms = importer_A.detect_pitch()
        importer_B_times, importer_B_frequency, b_peak, importer_B_rms = importer_B.detect_pitch()

        x = int(len(importer_B_times) / len(importer_A_times))

        if len(importer_A_times) < len(importer_B_times):
            limit = x # (Adapative_STFT_Algorithm)
            
            for i in range(len(importer_A_frequency)) and i == 0:
                match_A_freq = importer_A_frequency[i]
                match_B_freq  = importer_B_frequency[i]

                if abs(match_A_freq - match_B_freq) <= error:
                    importer_A_times[0] = importer_A_times[i]
                    importer_A_rms[0] = importer_A_rms[i]
                    a_peak[0] = a_peak[i]
                    break

            for j in range(len(importer_B_frequency)[limit:]):

                if abs(match_A_freq - match_B_freq) <= 0.3:
                    importer_A_frequency[0]= importer_B_frequency[i + j]
                    importer_B_rms[0] = importer_B_rms[i + j]
                    importer_B_times[0] = importer_B_times[i + j]
                    b_peak[0] = b_peak[i + j]
                    break
                else:
                    importer_A_frequency[0] = importer_B_frequency[i]
                    importer_B_rms[0] = importer_B_rms[i]
                    importer_B_times[0] = importer_B_times[i]
                    b_peak[0] = b_peak[i]


            


        return importer_A_times, importer_A_frequency, importer_B_times, importer_B_frequency, importer_A_rms, importer_B_rms, a_peak, b_peak
    
    def Back_to_Audio(importer_A, importer_B, importer_A_times, importer_A_frequency, importer_B_times, importer_B_frequency, importer_A_rms, importer_B_rms, a_peak, b_peak):
        importer_A = importer_A_times, importer_A_frequency, importer_A_rms, a_peak
        importer_B = importer_B_times, importer_B_frequency, importer_B_rms, b_peak

        return importer_A, importer_B

    """Onset detection to locate start times of musical notes, percussive beats, and sharp peaks"""
    
    def Get_onset(importer_A, importer_B):

        # Importer A
        onset_frames_A = librosa.onset.onset_detect(
            y=importer_A.audio,
            sr=importer_A.sr
        )

        onset_times_A = librosa.frames_to_time(
            onset_frames_A,
            sr=importer_A.sr
        )

        # Importer B 

        onset_frames_B = librosa.onset.onset_detect(
            y=importer_B.audio,
            sr=importer_B.sr
        )

        onset_times_B = librosa.frames_to_time(
            onset_frames_B,
            sr=importer_B.sr
        )

        return onset_times_A, onset_times_B, onset_frames_A, onset_frames_B

    """Statistical Analysis with Onset Detection"""

    def onset_stats (odf, detected_frames):
        odf = librosa.onset.onset_strength(importer_A.audio, importer_A.sr)
        detected_frames = librosa.onset.onset_detect(onset_envelope=odf, sr = importer_A.sr)
        num_samples = len(detected_frames)
        total_frames = len(odf)
        random_frames = np.random.choice(total_frames, size=num_samples, replace=False)
        odf_detected = odf[detected_frames]
        odf_random = odf[random_frames]
        ks_stat, p_value = stats.ks_2samp(odf_detected, odf_random)
        result = print(f" Computer-analyzed ODF: {np.mean(odf_detected):.4f}; Random ODF: {np.mean(odf_random):.4f}; P value: {p_value}; ks_stat: {ks_stat}")
        if abs(p_value) > 0.05:
            p_result = print ("The onset detection is accurate") 
        else:
            p_result = print("The onset detection lacks accuracy")
        return result, p_result

    """"Compute internnote intervals"""

    def inter_note_DTW(onset_times_A, onset_times_B):
        intervals_A = np.diff(onset_times_A)
        intervals_B = np.diff(onset_times_B)

        print("Intervals A:", intervals_A)
        print("Intervals B:", intervals_B)


        """
        Dynamic Time Warping (DTW) Alignment

        Purpose: To align two time series (e.g., audio signals) by finding the optimal alignment between them, minimizing the cumulative distance.
        
        """

        # DTW expects 2D arrays (features × frames)
        X = intervals_A.reshape(1, -1)
        Y = intervals_B.reshape(1, -1)

        cost_matrix, warp_path = dtw(X=X, Y=Y)

        # Reverse so the path goes from start → end
        warp_path = warp_path[::-1]

        print("Warp Path:")
        print(warp_path)

        hop_size = 512 # (Adapative_STFT_Algorithm)
        fft_size = 2048 # (Adapative_STFT_Algorithm)

        stft_complex = librosa.stft(importer_A.audio, n_fft=fft_size, hop_length=hop_size)

        # Transpose (.T) so shape is (frames, bins) for VariablePhaseVocoder
        stft_mag_A = np.abs(stft_complex).T
        stft_phase_A = np.angle(stft_complex).T

        num_frames = stft_mag_A.shape[0]
        duration_sec = len(importer_A.audio) / importer_A.sr
        
        return warp_path, stft_mag_A, stft_phase_A, num_frames, duration_sec, cost_matrix, 

    """
    Statistical Analysis for DTW Alignment accuracy

    1. stats_DTW
    2. calculate_rse
    3. check_cubic_spline
    """

        # 1. Alignment quality measure 
        # 2. Residual error of the distance and SSE

    def stats_DTW(warp_path, importer_B):
        y_true = np.array(importer_B) # Audio B 
        y_predict = np.array(warp_path) # stretched audio A 

        residuals = y_true - y_predict

        rss = np.sum(np.square(y_true - y_predict))

        return y_true, y_predict, residuals, rss

    def calculate_rse(y_true, y_predict, num_predictors=1):
        n = len(y_true)
        rss = np.sum(np.square(y_true - y_predict))
            
        # Degrees of freedom = n - p - 1
        degrees_of_freedom = n - num_predictors - 1

        rse_value = math.sqrt(rss / degrees_of_freedom)

        if rse_value < 0.15:
            result_DTW = print("The DTW Alignment is accurate with RSE value:", rse_value)
        elif rse_value < 0.30:
            result_DTW = print("The DTW Alignment is acceptable with RSE value:", rse_value)
        else:
            result_DTW = print("The DTW Alignment is not accurate with RSE value:", rse_value)

        return result_DTW

    """Find the pace ratio between each note and perform cubic spline interpolation to make smooth curves between each note"""

    def check_over_stretching(warp_path, intervals_A, intervals_B, onset_times_A):

        # =====================================================================
        # FIX CUBIC SPLINE X-AXIS (Use Onset Timestamps)
        # =====================================================================
        valid_times_A = []

        # ----------------------------
        # Compute Local Pace Ratios
        # ----------------------------

        pace_ratios = []

        for idxA, idxB in warp_path:

            # Make sure indices are valid
            if idxA < len(intervals_A) and idxB < len(intervals_B):

                valid_times_A.append(onset_times_A[idxA])

                ratio = intervals_B[idxB] / intervals_A[idxA]
                pace_ratios.append(ratio)

        return pace_ratios 

    """Statistical Analysis on Cubic Spline Interpolation (Overshooting, Undershooting, etc.)"""
    def check_cubic_spline(pace_ratios, valid_times_A):

    # Sort by time to guarantee strictly increasing x-axis for CubicSpline

        sorted_indices = np.argsort(valid_times_A)
        x_times = np.array(valid_times_A)[sorted_indices]
        y_ratios = np.array(pace_ratios)[sorted_indices]

        # Remove duplicate timestamps if DTW aligned multiple points to one note
        x_times, unique_idx = np.unique(x_times, return_index=True)
        y_ratios = y_ratios[unique_idx]

        return x_times, y_ratios, valid_times_A, pace_ratios, x_times

    """
    Analogy:
    # Pace_ratios -> A sign say next 500 meters, drive 20% slower
    # Cubic Spline -> Instead of 60mph to 40 mph instantly, we need a smooth transition
    # Time-warping function -> At this point you shoud be here; gives the GPS coordinates
    # Time-stretching algorithm -> Actual car movement 

    """

    def stretch_rms_amplitude(rms_original, warp_curve, t_dense, sr, hop_length=512):
        

        frame_time = hop_length / sr

        # Total stretched duration -> number of output frames (matches create_analysis_times)
        output_duration = warp_curve[-1]
        num_output_frames = int(output_duration / frame_time)
        output_times = np.arange(num_output_frames) * frame_time

        # Invert warp_curve: stretched time -> original time
        inverse_warp = interp1d(
            warp_curve, t_dense,
            bounds_error=False,
            fill_value=(t_dense[0], t_dense[-1])  # clamp at edges instead of extrapolating wildly
        )
        original_times = inverse_warp(output_times)

        # Convert original times into fractional frame positions in rms_original
        original_frame_positions = original_times / frame_time

        stretched_rms = np.zeros(num_output_frames)

        for i, position in enumerate(original_frame_positions):
            position = np.clip(position, 0, len(rms_original) - 1)

            left = int(np.floor(position))
            right = min(left + 1, len(rms_original) - 1)
            alpha = position - left

            # Linear interpolation between neighboring original RMS frames
            stretched_rms[i] = (1 - alpha) * rms_original[left] + alpha * rms_original[right]

        return stretched_rms

    """
    Cubic Spline 
    
    Pace ratios -> integral -> warp curve
    
    """
    def cubic_spline_integral(pace_ratios, duration_sec, num_frames, x_times):

        cs = CubicSpline(x_times, pace_ratios, bc_type='natural')
        t_dense = np.linspace(0, duration_sec, num_frames)

        raw_curve = cs(t_dense)
        raw_curve = np.nan_to_num(raw_curve, nan=1.0, posinf=4.0, neginf=0.25)

        if np.any(np.isinf(raw_curve)) or np.any(np.isnan(raw_curve)):
            raise ValueError("Cubic spline produced non-finite values before clipping.")

        ratio_curve = np.clip(raw_curve, 0.25, 4.0)

        warp_curve = cumulative_trapezoid(ratio_curve, t_dense, initial=0)

        return warp_curve, ratio_curve, t_dense



"""

Determine when to use granular synthesis vs phase vocoder, based on spectral flux and RMS loudness.

"""

class Compute_optimize_hybrid_mask(Granular_synthesis, Variable_phasevocoder, Intonation_Analysis, Alignment):

    def __init__(self, transient_A, transient_B):
        self.warp_curve = transient_A
        self.transient_B = transient_B

    def two_flux_columns(audio_granular, audio_phase, hop_length=512, n_fft=2048):
        """
        Computes spectral flux + RMS for both candidate renderings of audio_A
        (granular vs phase vocoder).
        """
        stft_granular = np.abs(librosa.stft(audio_granular, n_fft=n_fft, hop_length=hop_length))
        stft_phase = np.abs(librosa.stft(audio_phase, n_fft=n_fft, hop_length=hop_length))

        flux_granular = np.sum(np.maximum(0, np.diff(stft_granular, axis=1)), axis=0)
        flux_phase = np.sum(np.maximum(0, np.diff(stft_phase, axis=1)), axis=0)

        flux_granular = np.pad(flux_granular, (1, 0), mode='edge')
        flux_phase = np.pad(flux_phase, (1, 0), mode='edge')

        rms_granular = librosa.feature.rms(
            y=audio_granular, frame_length=n_fft, hop_length=hop_length
        )[0]
        rms_phase = librosa.feature.rms(
            y=audio_phase, frame_length=n_fft, hop_length=hop_length
        )[0]

        return flux_granular, flux_phase, rms_granular, rms_phase

    def conditional(audio_granular, audio_phase, threshold):
        """
        Per-frame pick: granular value if it beats threshold, else phase-vocoder value.
        audio_granular, audio_phase, threshold must be same-length 1D arrays.
        """
        n = min(len(audio_granular), len(audio_phase), len(threshold))
        analysis_results = np.zeros(n)

        for k in range(n):
            if audio_granular[k] > threshold[k]:
                analysis_results[k] = audio_granular[k]
            else:
                analysis_results[k] = audio_phase[k]

        return analysis_results

    def compare_audio_transient(audio_A_analysis_result, audio_B_analysis_result):
        """
        Compares two per-frame result arrays; for frames where A > B,
        records the relative percentage gap.
        """
        comparable_results = np.column_stack((audio_A_analysis_result, audio_B_analysis_result))
        n = comparable_results.shape[0]
        marked_result = np.zeros(n)

        for i in range(n):
            comparing_item_A = comparable_results[i, 0]
            comparing_item_B = comparable_results[i, 1]

            if comparing_item_B < comparing_item_A:
                x, y = comparing_item_B, comparing_item_A
            else:
                x, y = comparing_item_A, comparing_item_B

            # avoid divide-by-zero
            percentage = (y - x) / x if x != 0 else 0.0

            if comparing_item_A - comparing_item_B > 0:
                marked_result[i] = percentage

        return marked_result

    def rms_comparison(stretched_rms, rms_amplitudes_B):
        """
        Same pattern as compare_audio_transient, applied to RMS loudness arrays.
        """
        comparable_results = np.column_stack((stretched_rms, rms_amplitudes_B))
        n = comparable_results.shape[0]
        marked_rms = np.zeros(n)

        for i in range(n):
            comparing_item_A = comparable_results[i, 0]
            comparing_item_B = comparable_results[i, 1]

            if comparing_item_B < comparing_item_A:
                x, y = comparing_item_B, comparing_item_A
            else:
                x, y = comparing_item_A, comparing_item_B

            percentage = (y - x) / x if x != 0 else 0.0

            if comparing_item_A - comparing_item_B > 0:
                marked_rms[i] = percentage

        return marked_rms

    def align_decisions_to_original_time(marked_rms, marked_result, warp_curve, t_dense, sr, hop_length=512):
        """
        Maps each frame-level decision (indexed by frame in the stretched/rendered
        audio) back to its corresponding original (pre-stretch) timestamp.
        """
        inverse_warp = interp1d(
            warp_curve, t_dense,
            bounds_error=False,
            fill_value=(t_dense[0], t_dense[-1])
        )

        frame_rate = sr / hop_length
        n_rms = len(marked_rms)
        aligned = []

        for k, decision in enumerate(marked_result):
            stretched_time = k / frame_rate
            original_time = float(inverse_warp(stretched_time))

            # convert original_time (seconds) into a valid integer frame index
            rms_frame_idx = int(round(original_time * frame_rate))
            rms_frame_idx = max(0, min(rms_frame_idx, n_rms - 1))

            aligned.append({
                "frame": k,
                "stretched_time_sec": round(stretched_time, 4),
                "original_time_sec": round(original_time, 4),
                "decision": decision,
                "loudness_rms": marked_rms[rms_frame_idx]
            })

        return aligned








"""
Call architecture 

"""

# ----------------------------------------------------------------
# Full pipeline: onset -> DTW -> pace ratios -> spline -> stretch ->
# hybrid mask -> tune noter (on ORIGINAL time/frequency, last step)
# ----------------------------------------------------------------

onset_times_A, onset_times_B, onset_frames_A, onset_frames_B = Alignment.Get_onset(importer_A, importer_B)

warp_path, stft_mag_A, stft_phase_A, num_frames, duration_sec, cost_matrix = Alignment.inter_note_DTW(
    onset_times_A, onset_times_B
)

pace_ratios = Alignment.check_over_stretching(
    warp_path, np.diff(onset_times_A), np.diff(onset_times_B), onset_times_A
)

x_times, y_ratios, valid_times_A, pace_ratios, x_times = Alignment.cubic_spline(
    pace_ratios, onset_times_A[: len(pace_ratios)]
)

warp_curve, ratio_curve, t_dense = Alignment.cubic_spline_integral(
    y_ratios, duration_sec, num_frames, x_times
)

# --- Phase vocoder branch ---
vocoder = Variable_phasevocoder(stft_mag_A, stft_phase_A, warp_curve, t_dense, importer_A.sr)
vocoder.process_magnitudes_only()
final_mask_vocoder, sf_vocoder, threshold_vocoder = vocoder.transient_mask_from_vocoder(alpha=2.0, hold_ms=30)

vocoder_window = np.hanning(2048)  # must match fft_size (2048) used above
audio_phase = vocoder.inverse_stft(vocoder_window)

# --- Granular branch ---
granular = Granular_synthesis(importer_A, ratio_curve, t_dense)
granular.allocate_large()
final_mask_granular, sf_granular, threshold_granular = granular.transient_mask_from_granular(alpha=2.0, hold_ms=30)
audio_granular = granular.audio_A_granular

# --- Hybrid decision: granular vs phase vocoder, per frame ---
flux_granular, flux_phase, rms_granular, rms_phase = Compute_optimize_hybrid_mask.two_flux_columns(
    audio_granular, audio_phase
)

marked_result = Compute_optimize_hybrid_mask.conditional(
    flux_granular, flux_phase, threshold_vocoder
)

stretched_rms = Alignment.stretch_rms_amplitude(importer_A_rms, warp_curve, t_dense, importer_A.sr)

marked_rms = Compute_optimize_hybrid_mask.rms_comparison(
    stretched_rms, rms_amplitudes_B
)

aligned = Compute_optimize_hybrid_mask.align_decisions_to_original_time(
    marked_rms, marked_result, warp_curve, t_dense, importer_A.sr
)

# --- Tune noter: LAST step, reports ORIGINAL time + ORIGINAL frequency ---
tuner = Intonation_Analysis(standard_A4)
tuned_report = tuner.analyze_aligned_decisions(
    aligned, importer_A_times, importer_A_frequencies
)

for row in tuned_report[:10]:
    print(
        f"t={row['original_time_sec']}s  "
        f"freq={row['original_frequency_hz']:.2f}Hz  "
        f"deviation={row['half_step_deviation']} semitones  "
        f"decision={row['decision']}"
    )