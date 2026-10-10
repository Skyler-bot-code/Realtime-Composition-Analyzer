# Below function is for making sure starting at the same pace


# First approach 
# 1. - use ratio of time of entire to time between notes 
# 2. - use ratio between the time of one 小節 to the number of beats for 4/4 6/8 2/4 ...etc
# 3. find the frequency points over time on a graph
# 4. Use symbolic regression to find the curve on a graph
# 5. Use calculus to find the relative min and max
# 6. - do some relationship (maybe mathematical function, infinite series functions) between step 2 and 5.
# 7. return to the model a ratio (that comes from a result from step 6)
# 8.  - do a mathematical function or AI learning model (Open AI or others) for changing the time and adding the pace
# 8 -1, if faster, then stretch the frequencies points over time (make-it-similar model)
# 8 -2, if slower, then shrink the frequencies points over time (make-it-similar model)


# Second approach
# Directly use infinite sequences to find the pattern
# - For each group (A, B)
# - Group 1: Sudden sound like (drums) #Amplitude: when a string was plucked, struck, or heavily bowed [onset detection], Amplitude Transient or Energy Onset [onset detection]
# - Group 2: Base frequency like (Strings) # Frequency: what note is being played, Pitch Shift, Fundamental Frequency Transition, or a spike in Spectral Flux.
# - Compare: position and determine the pace through some kind of ratio (step 4 to 8)

# Combining approach first and second 


# Group 1

# Prototype 1 (RAW)
""" 
# First, identify the strong notes (Higher RMS amplitude)
timesA, rms_list_A_raw = importer_A.detect_rms()
timesB, rms_list_B_raw = importer_B.detect_rms()

# Detech notes using onset !!
rms_list_A = []
rms_list_B = []

# Importer A
onset_framesA = librosa.onset.onset_detect(
    y=importer_A.audio,
    sr=importer_A.sr
)

rms_list_A = librosa.frames_to_time(
    onset_framesA,
    sr=importer_A.sr
)

# Importer B
onset_framesB = librosa.onset.onset_detect(
    y=importer_B.audio,
    sr=importer_B.sr
)

rms_list_B = librosa.frames_to_time(
    onset_framesB,
    sr=importer_B.sr
)
# rms_list_A is a list that tells you where the beats (notes) are at 
print(rms_list_A)

# rms_list_A is a list that tells you where the beats (notes) are at 
print(rms_list_B)



# Get the inteval sequences for A and B !!
terms_A = []
terms_B = []
for i in range(onset_times_A):
    element_A = rms_list_A[i+1] - rms_list_A[i]
    terms_A[i] = element_A
    i+=1

for i in range(rms_list_B):
    element_B = rms_list_B[i+1] - rms_list_B[i]
    terms_B[i] = element_B
    i+=1

# Compare sequences to find pace ratio and perform DTW !!
pace = []


for n in 
    ratio = timesA[n] / timesB[n]


    
elif ratio < 1:
    
else:
    """

# Prototype 2 (RAW -> REVISED)
# ----------------------------
# Detect note onsets
# ----------------------------




# Step 1 -> pYIN algorithm: Separate fundemental frequencies
# Step 1 -> Onset Detection: detect any frequency spikes so I know where the beats (notes) are
# Step 2 -> DTW wrapping, Aligns the note correctly, doesn't stretch yet, it only maps it on a separate graph 
# DTW is a machine learning model/algorithm that uses cost function (need to learn more about this)
# Step 2 -> Find the ratio between each note using time interval audio A / audio B
# Step 3 -> cubic spline interpolation to make smooth curves between each note
# Step 4 -> Non-linear time-stretching (on the actual audio) (Step 2 DTW is the map)
# 

from Audio_Importer import AudioImporter
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



# Audio B

file = "/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav"



audio_file = AudioImporter(file)

audio_file.load_audio()

times_B, frequencies_B, peak_amplitudes_B, rms_amplitudes_B = audio_file.detect_pitch()



# Use Multivariate Dynamic Time Warping (DTW) 

# Assuming importer_A and importer_B have run their analysis...
 # from youtube

### Importer A should come from double_notes_multiple_instruments based on users' choice 

importer_A = AudioImporter("/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav") # from computer
importer_B = AudioImporter("/Users/skyler/Desktop/audio.wav") # from youtube

print("Processing Audio File A...")
 # This fills importer_A with data

print("Processing Audio File B...")
importer_B.load_audio()
importer_B.detect_pitch()  # This fills importer_B with data


# Prototype 1 for comparison model (Time series analysis using DTW)


"""
# 1. Stack your multiple parameters together to create a Multivariate Array
# Shape becomes: (2 features, N time steps)
features_A = np.vstack([importer_A.frequencies, importer_A.rms_amplitudes])
features_B = np.vstack([importer_B.frequencies, importer_B.rms_amplitudes])

# 2. Use DTW to compare the multivariate matrices
# 'metric=cosine' or 'euclidean' evaluates the multidimensional distance
D, wp = librosa.sequence.dtw(X=features_A, Y=features_B, metric='cosine')

# 3. Calculate a normalized distance score (Closer to 0 means identical)
distance_score = D[-1, -1] / len(wp)
print(f"Distance Score: {distance_score:.4f}")


"""
# write a model that makes both audio track comparable
# 1. Start from equal frequency and amplitude pitch
# 2. Play speed will also be different
# - use ratio of time of entire to time between notes 
# - Time between notes: 


class Alignment(AudioImporter):

    def __init__(self, importer_A_times, importer_A_frequency, importer_B_times, importer_B_frequency):
        self.importer_A_times = importer_A_times
        self.importer_A_frequencies = importer_A_frequency
        self.importer_B_times = importer_B_times
        self.importer_B_frequencies = importer_B_frequency


# Below function is for making sure starting at the same note
    def Same_start_time(importer_A, importer_B, importer_A_frequency, importer_B_frequency, importer_A_times, importer_B_times, importer_A_rms, importer_B_rms,a_peak, b_peak):
        error = 5 # neural network

        importer_A_times, importer_A_frequency, a_peak, importer_A_rms = importer_A.detect_pitch()
        importer_B_times, importer_B_frequency, b_peak, importer_B_rms = importer_B.detect_pitch()

        x = int(len(importer_B_times) / len(importer_A_times))

        if len(importer_A_times) < len(importer_B_times):
            limit = x # neural network
            
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

    #++++++++++++++++++++++++++++++++++++
    # Stats analysis for onset detection
    #++++++++++++++++++++++++++++++++++++

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

    # ----------------------------
    # Compute Inter-Note Intervals
    # ----------------------------

    def inter_note_DTW(onset_times_A, onset_times_B):
        intervals_A = np.diff(onset_times_A)
        intervals_B = np.diff(onset_times_B)

        print("Intervals A:", intervals_A)
        print("Intervals B:", intervals_B)



        # ----------------------------
        # DTW Alignment (Because one recording might be playing a note with longer time, or there could be some variations, DTW compares and aligns the one with the lower cost, lower cost means the two notes are more similar)
        # ----------------------------

        # DTW expects 2D arrays (features × frames)
        X = intervals_A.reshape(1, -1)
        Y = intervals_B.reshape(1, -1)

        cost_matrix, warp_path = dtw(X=X, Y=Y)

        # Reverse so the path goes from start → end
        warp_path = warp_path[::-1]

        print("Warp Path:")
        print(warp_path)

        hop_size = 512 # neural network
        fft_size = 2048 # neural network

        stft_complex = librosa.stft(importer_A.audio, n_fft=fft_size, hop_length=hop_size)

        # Transpose (.T) so shape is (frames, bins) for VariablePhaseVocoder
        stft_mag_A = np.abs(stft_complex).T
        stft_phase_A = np.angle(stft_complex).T

        num_frames = stft_mag_A.shape[0]
        duration_sec = len(importer_A.audio) / importer_A.sr
        
        return warp_path, stft_mag_A, stft_phase_A, num_frames, duration_sec, cost_matrix, 

        #+++++++++++++++++++++++++++++++++++++++++++
        # Stats analysis for DTW alignmnet accuracy
        #+++++++++++++++++++++++++++++++++++++++++++

        # Looking for alignment quality measure 
        # 1. Calculate residual Error of the distance and SSE

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

    def check_over_stretching(warp_path, intervals_A, intervals_B, onset_times_A):

        # =====================================================================
        # 2. FIX CUBIC SPLINE X-AXIS (Use Onset Timestamps)
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

        return pace_ratios # Strech audio A by "" (Later could do a conditional statement to determine which one is more efficient for stretching)

#++++++++++++++++++++++++++++++++++++
# Stats analysis for cubic spline 1 
#++++++++++++++++++++++++++++++++++++

    def cubic_spline(pace_ratios, valid_times_A):

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
        """
        Maps the ORIGINAL rms amplitude envelope onto the STRETCHED timeline,
        using the same warp_curve inversion as create_analysis_times().

        Parameters
        ----------
        rms_original : np.ndarray
            RMS amplitude per frame, computed on the ORIGINAL (pre-stretch) audio,
            e.g. from importer_A.detect_pitch() / detect_rms(), at hop_length granularity.
        warp_curve : np.ndarray
            T(t): maps original time -> stretched time.
        t_dense : np.ndarray
            Original time axis paired with warp_curve.
        sr : int
            Sample rate.
        hop_length : int
            Hop length used for the ORIGINAL rms_original array's frame spacing.

        Returns
        -------
        stretched_rms : np.ndarray
            RMS amplitude resampled onto the stretched timeline, one value per
            output frame -- same length/timing convention as output_mag.
        """

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

# Cubic Spline (After having pace ratios, we cannot jump directly into stretching alogrithms because we don't know what is going on between the notes, we only know the boundaries when computing ratio )
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
    
    
    
    # also need stretched amplitude 
    

    # warp curve is not returned as a frequency
    # warp curve passes in granular and phase vocoder respectively
    # granular and phase vocoder stretch the aligned audio A respectively and passes out frequency list?


# The above code gives time -> ratio
# WSOLA wants input time -> output time.


# statistical analysis on flaws of cubic spline interpolation (like overshooting, undershooting, etc.)


# Continuous ratio curve

#Integrate ratio curve 


# Step 4: Time-warp function (Actual audio instruction for stretching algorithms) This function gives the location.

# Solving challenge of tempo speeding up, pitch shifts higher
## Short time fast fourier transform 
## Waveform Similarity Overlap-Add - Time-Domain Method



# Step 4: Actual stretching algorithms (Time-Scale Modification (TSM)
# TSM techniques aim to change the speed of audio without affecting its pitch.


# Granular - stretched audio A
# Phase_vocoder - stretched audio A

# Audio - B


# Two aspects of comparison:
# 1. Compare the stretched audio A with audio B on whether they have the same spike, same time for a certain frequency, and return the time from the original audio A
# 2. Compare original audio A with a 調音器 (Tuner) to see the frequency accuracy and return the original time for that certain note

# Print out the most detected frequency and the time section from the original audio A (before stretching) and audio B


## 1. Construct a piece-wise function on determining to use either granular or phase vocoder
# Transient-Aware Hybrid TSM
### 1. 
# Look for energy shifts (SF)between notes


### 2. Adaptive dynamic threshold using moving median and a std over short sliding window 

### 3. Compute the piece wise function 
# for phase vocoder - SF ~= 0 
# for granular - SF >> 0

# Granular - audio_A_granular
# Phase vocoder - audio_A_phase_vocoder
# Base of audio A is phase_vocoder, uses granular for specific peaks, beats




#######
#Call Phase vocoder, and granular (ready for transient)
# Undergo transient and form a single list that uses either which one to get to frequency
# The frequency will be a frequency list, and then compare that to Audio B (Online audio)
# Get in tune_noter
# See the emphasis (use percentage and emphasis)
# And user interface 

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