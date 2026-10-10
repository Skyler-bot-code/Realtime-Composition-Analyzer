# Slowest
# Revised Version 



# Multi-label instrument detection for audio B
#
# Pipeline approach
# Uses statistics -> feature vector -> per-instrument logistic regression classifier

# Parameters include:
"""
1. Spectral Centroid: Sharpness of notes
2. Spectral Spread: Notes level of concentration
3. Spectral Flux: Detects unstable notes, emotion representation
4. Temporal features: Detects the fade of the notes
5. MFCC Vector: isolate irrelevant sounds

Proposed algorithm:
1. Use MFCC vector
2. Then a model that detects the three spectral functions, and temporal features
   (these four are functions with parameters from short time fourier transform / EMD)
3. From that return, classify through logistic regression with gradient descent

Fatal problem noted in the original draft: functions cannot be changed dynamically,
so a purely rule-based model cannot capture really small details. This revision keeps
the feature engineering but makes the trainer/inference math actually correct so the
logistic-regression heads can learn to compensate.
"""

# ------------------------------------------------------------------------------------
# WAY 1 (reference only, not executed): Google's PANNs Cnn14 audio tagging model.
# Kept as a documented alternative approach; requires the optional `panns_inference`
# package and network access, so it is left commented out rather than imported.
# ------------------------------------------------------------------------------------
"""
import torch
import numpy as np
import librosa
from pann_inference import AudioTagging

def pipeline_instrument_counter(audio_path, target_instruments, threshold=0.3):
    y, sr = librosa.load(audio_path, sr=32000, mono=True)
    audio_input = y[None, :]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    at = AudioTagging(checkpoint_path=None, device=device)
    _, framewise_output = at.inference(audio_input)
    global_probs = np.mean(framewise_output, axis=0)
    from pann_inference import labels
    detected_kinds = []
    for instrument in target_instruments:
        if instrument in labels:
            idx = labels.index(instrument)
            prob = global_probs[idx]
            if prob > threshold:
                detected_kinds.append((instrument, float(prob)))
    total_kinds_count = len(detected_kinds)
    return total_kinds_count, detected_kinds

my_targets = ['Piano', 'Acoustic guitar', 'Violin', 'Flute', 'Drum kit', 'Trumpet']
# count, details = pipeline_instrument_counter("my_song.wav", my_targets)
"""

# ------------------------------------------------------------------------------------
# WAY 2 (HHT-Logistic): H2MB-Tracker
# System and method for polyphonic timbre detection via Hilbert-Huang demodulation
# and multivariate logistic decision boundaries. Bypasses 2D CNN kernels and STFT
# grids for the frequency-tracking stage; only MFCC uses STFT internally.
# ------------------------------------------------------------------------------------

import os
import json
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import librosa
from scipy.signal import medfilt, hilbert
from PyEMD import EMD
from sklearn.cluster import DBSCAN


# --------------------------------------------------------------------------------
# Windowed feature-extraction worker (used for parallel processing of long audio,
# one ~1-second slice per worker). This is a plain module-level function so it can
# be pickled and sent to worker processes by ProcessPoolExecutor.
# --------------------------------------------------------------------------------
def _amplitude_weighted_centroid_spread_flux(freq_tracks, amp_tracks):
    """
    Computes an amplitude-weighted spectral centroid, spread, and flux from one or
    more instantaneous-frequency / instantaneous-amplitude tracks (as produced by
    Hilbert-transforming each IMF).

    freq_tracks / amp_tracks: lists of 1D numpy arrays (one array per IMF track).
    The arrays may differ in length between tracks, so they are concatenated
    (not stacked) before the weighted statistics are computed.
    """
    if len(freq_tracks) == 0:
        return 0.0, 0.0, 0.0

    freq_concat = np.concatenate(freq_tracks)
    amp_concat = np.concatenate(amp_tracks)

    amp_sum = np.sum(amp_concat)
    if amp_sum <= 0:
        amp_sum = 1e-8

    centroid = np.sum(freq_concat * amp_concat) / amp_sum
    spread = np.sqrt(max(0.0, np.sum(((freq_concat - centroid) ** 2) * amp_concat) / amp_sum))

    # Flux: frame-to-frame instability of the *primary* track (the first IMF),
    # since diffing the concatenation of unequal-length tracks isn't meaningful.
    primary = freq_tracks[0]
    flux = float(np.mean(np.diff(primary) ** 2)) if len(primary) > 1 else 0.0

    return float(centroid), float(spread), flux


def _process_single_window_worker(window_data, window_idx, start_time, sr, n_mfcc):
    """
    Parallel execution worker. Processes a small (~1 second) slice of audio
    independently so EMD converges quickly (typically well under 10 ms per window
    at this length). Returns a plain dict so it can cross process boundaries.
    """
    try:
        # Amplitude-safety normalization, local to this worker.
        max_amp = np.max(np.abs(window_data))
        y_norm = window_data / (max_amp + 1e-8) if max_amp > 1e-4 else window_data

        # Simple 1D de-noising via median filter; treat the residual as noise.
        y_pure = medfilt(y_norm, kernel_size=3)
        y_noise = y_norm - y_pure
        zcr_noise = (
            float(np.mean(librosa.feature.zero_crossing_rate(y=y_noise)))
            if len(y_noise) > 0
            else 0.0
        )

        # Static timbre features for this window.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfcc_matrix = librosa.feature.mfcc(
                y=y_pure, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256
            )
        mfcc_vector = (
            np.mean(mfcc_matrix, axis=1) if mfcc_matrix.size > 0 else np.zeros(n_mfcc)
        )

        # Local EMD sifting.
        emd_engine = EMD()
        imfs = emd_engine(y_pure)
        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_pure])

        inst_frequencies = []
        inst_amplitudes = []
        primary_imf_amp = np.zeros(max(len(y_pure) - 1, 0))
        local_point_cloud = []

        time_axis = (np.arange(len(y_pure) - 1) / sr) + start_time
        target_imf_count = min(2, len(imfs))

        for i in range(target_imf_count):
            current_imf = imfs[i]
            if np.all(current_imf == 0):
                continue

            analytic = hilbert(current_imf)
            amp_envelope = np.abs(analytic)[:-1]
            phase = np.unwrap(np.angle(analytic))
            frequency = (np.diff(phase) / (2.0 * np.pi)) * sr
            frequency = np.clip(frequency, 0, sr / 2.0)

            inst_amplitudes.append(amp_envelope)
            inst_frequencies.append(frequency)
            if i == 0:
                primary_imf_amp = amp_envelope

            for t_idx in range(len(frequency)):
                if amp_envelope[t_idx] > 0.01:
                    local_point_cloud.append(
                        [time_axis[t_idx], frequency[t_idx], amp_envelope[t_idx]]
                    )

        if len(inst_frequencies) == 0:
            inst_frequencies = [np.zeros(max(len(y_pure) - 1, 0))]
            inst_amplitudes = [np.zeros(max(len(y_pure) - 1, 0))]

        # Track 1 = primary IMF only; Track 2 = all extracted IMFs (or same as
        # track 1 if there is only one). Computed with proper amplitude-weighted
        # statistics instead of multiplying ragged lists together.
        t1_centroid, t1_spread, t1_flux = _amplitude_weighted_centroid_spread_flux(
            [inst_frequencies[0]], [inst_amplitudes[0]]
        )
        if len(inst_frequencies) > 1:
            t2_centroid, t2_spread, t2_flux = _amplitude_weighted_centroid_spread_flux(
                inst_frequencies, inst_amplitudes
            )
        else:
            t2_centroid, t2_spread, t2_flux = t1_centroid, t1_spread, t1_flux

        pitch_gap = abs(t1_centroid - t2_centroid)
        micro_beating = (
            float(np.mean(np.diff(primary_imf_amp, n=2) ** 2))
            if len(primary_imf_amp) > 2
            else 0.0
        )

        hht_feats = np.array(
            [
                t1_centroid, t1_spread, t1_flux,
                t2_centroid, t2_spread, t2_flux,
                pitch_gap, micro_beating, zcr_noise,
            ]
        )
        X = np.concatenate((mfcc_vector, hht_feats))
        X = np.append(X, t1_centroid * micro_beating)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return {"status": "SUCCESS", "vector": X.tolist(), "point_cloud": local_point_cloud}
    except Exception:
        return {"status": "FAILED", "vector": [], "point_cloud": []}


class Double_notes_architecture:
    """
    Edge-computing audio classification and transcription processor.
    Guards against floating-point overflow, zero-division, and IMF mode-mixing noise.
    """

    def __init__(self, target_sr=16000, n_mfcc=13, noise_threshold=0.80, instrument_threshold=0.50):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.noise_threshold = noise_threshold
        self.instrument_threshold = instrument_threshold
        self.emd_engine = EMD()

        # Hyperplane parameter containers.
        self.trained_weights = {}
        self.trained_biases = {}
        self.vector_size = self.n_mfcc + 9 + 1  # 13 MFCC + 9 HHT descriptors + 1 cross term
        self.is_model_trained = False

    def _safe_sigmoid(self, z):
        """Clips input tightly to avoid overflow in np.exp."""
        clipped_z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped_z))

    def save_weights_to_json(self, export_path="h2mb_weights.json"):
        if not self.is_model_trained or not self.trained_weights:
            raise ValueError("[ERROR]: Cannot export parameter layers. Model is untrained.")

        export_payload = {
            "metadata": {
                "algorithm_identity": "H2MB_AUDIO_ENGINE",
                "vector_input_size": self.vector_size,
                "n_mfcc_coefficients": self.n_mfcc,
            },
            "weights": {label: w.tolist() for label, w in self.trained_weights.items()},
            "biases": {label: float(b) for label, b in self.trained_biases.items()},
        }

        with open(export_path, "w", encoding="utf-8") as file_stream:
            json.dump(export_payload, file_stream, indent=4)
        print(f"[IO]: Exported trained weights to '{export_path}'")

    def load_weights_from_json(self, import_path="h2mb_weights.json"):
        if not os.path.exists(import_path):
            raise FileNotFoundError(f"[ERROR]: Weight file not found at '{import_path}'")

        with open(import_path, "r", encoding="utf-8") as file_stream:
            payload = json.load(file_stream)

        self.trained_weights = {
            label: np.array(w, dtype=np.float64) for label, w in payload["weights"].items()
        }
        self.trained_biases = {
            label: np.float64(b) for label, b in payload["biases"].items()
        }
        self.vector_size = int(payload["metadata"]["vector_input_size"])
        self.n_mfcc = int(payload["metadata"]["n_mfcc_coefficients"])
        self.is_model_trained = True
        print(f"[IO]: Loaded {len(self.trained_weights)} trained categories.")

    def _extract_23d_feature_vector(self, y_pure, zcr_noise, use_parallel_windows=False):
        """
        Builds the 23-D feature vector for one audio clip: 13 MFCC coefficients +
        9 HHT descriptors + 1 nonlinear cross term.

        Steps:
          1. MFCC (static timbre) over the whole clip.
          2. EMD -> Hilbert transform per IMF -> instantaneous frequency/amplitude.
          3. Amplitude-weighted spectral centroid/spread/flux per track.
          4. Feature fusion into a single 23-D vector.

        If use_parallel_windows=True and the clip is long enough, the instantaneous-
        frequency extraction is split into ~1s windows and processed in parallel via
        _process_single_window_worker; their point clouds are merged. This is
        optional because process-pool overhead isn't worth it for short clips.
        """
        # [STEP 1] STATIC TIMBRE FEATURE EXTRACTION
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mfcc_matrix = librosa.feature.mfcc(
                y=y_pure, sr=self.target_sr, n_mfcc=self.n_mfcc, n_fft=512, hop_length=256
            )
        mfcc_vector = (
            np.mean(mfcc_matrix, axis=1) if mfcc_matrix.size > 0 else np.zeros(self.n_mfcc)
        )

        point_cloud = []

        if use_parallel_windows and len(y_pure) > self.target_sr:
            # Split into ~1-second windows and dispatch across worker processes.
            window_len = self.target_sr
            num_windows = max(1, len(y_pure) // window_len)
            num_cores = max(1, mp.cpu_count() - 1)

            results = []
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = []
                for w in range(num_windows):
                    start = w * window_len
                    end = min(start + window_len, len(y_pure))
                    window_data = y_pure[start:end]
                    start_time = start / self.target_sr
                    futures.append(
                        executor.submit(
                            _process_single_window_worker,
                            window_data, w, start_time, self.target_sr, self.n_mfcc,
                        )
                    )
                for f in futures:
                    results.append(f.result())

            for res in results:
                if res["status"] == "SUCCESS":
                    point_cloud.extend(res["point_cloud"])

            # Fold per-window vectors back into a single clip-level vector by
            # averaging; falls back to the whole-clip EMD path if nothing succeeded.
            successful_vectors = [r["vector"] for r in results if r["status"] == "SUCCESS"]
            if successful_vectors:
                X = np.mean(np.array(successful_vectors), axis=0)
                return X, point_cloud
            # else fall through to the serial whole-clip path below.

        # [STEP 2] HHT SIGNAL DEMODULATION (whole clip, serial)
        try:
            imfs = self.emd_engine(y_pure)
        except Exception:
            imfs = np.array([y_pure])

        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_pure])

        inst_frequencies = []
        inst_amplitudes = []
        primary_imf_amplitudes_raw = np.zeros(max(len(y_pure) - 1, 0))

        time_axis = np.arange(max(len(y_pure) - 1, 0)) / self.target_sr
        target_imf_count = min(2, len(imfs))

        for i in range(target_imf_count):
            current_imf = imfs[i]
            if np.all(current_imf == 0):
                continue

            analytic_signal = hilbert(current_imf)
            amplitude_envelope = np.abs(analytic_signal)[:-1]

            phase_angle = np.unwrap(np.angle(analytic_signal))
            frequency = (np.diff(phase_angle) / (2.0 * np.pi)) * self.target_sr
            frequency = np.clip(frequency, 0, self.target_sr / 2.0)

            inst_amplitudes.append(amplitude_envelope)
            inst_frequencies.append(frequency)

            if i == 0:
                primary_imf_amplitudes_raw = amplitude_envelope

            for t_idx in range(len(frequency)):
                if amplitude_envelope[t_idx] > 0.01:
                    point_cloud.append(
                        [time_axis[t_idx], frequency[t_idx], amplitude_envelope[t_idx]]
                    )

        if len(inst_frequencies) == 0:
            inst_frequencies = [np.zeros(max(len(y_pure) - 1, 0))]
            inst_amplitudes = [np.zeros(max(len(y_pure) - 1, 0))]

        # [STEP 3] AMPLITUDE-WEIGHTED SPECTRAL DESCRIPTORS PER TRACK
        t1_centroid, t1_spread, t1_flux = _amplitude_weighted_centroid_spread_flux(
            [inst_frequencies[0]], [inst_amplitudes[0]]
        )
        if len(inst_frequencies) > 1:
            t2_centroid, t2_spread, t2_flux = _amplitude_weighted_centroid_spread_flux(
                inst_frequencies, inst_amplitudes
            )
        else:
            t2_centroid, t2_spread, t2_flux = t1_centroid, t1_spread, t1_flux

        pitch_distance_gap = abs(t1_centroid - t2_centroid)
        micro_beating = (
            float(np.mean(np.diff(primary_imf_amplitudes_raw, n=2) ** 2))
            if len(primary_imf_amplitudes_raw) > 2
            else 0.0
        )

        # [STEP 4] FEATURE FUSION
        hht_features = np.array(
            [
                t1_centroid, t1_spread, t1_flux,
                t2_centroid, t2_spread, t2_flux,
                pitch_distance_gap, micro_beating, zcr_noise,
            ]
        )
        X = np.concatenate((mfcc_vector, hht_features))
        X = np.append(X, t1_centroid * micro_beating)  # nonlinear interaction term
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, point_cloud

    def automated_gradient_descent_trainer(self, dataset_root_dir, epochs=50, learning_rate=0.01, batch_size=4):
        """
        Scans the dataset directory tree (one subfolder per instrument label,
        e.g. dataset_root_dir/piano/*.wav), extracts feature vectors, and trains
        one logistic-regression hyperplane per label via mini-batch gradient
        descent, using multi-label binary cross-entropy.
        """
        if not os.path.exists(dataset_root_dir):
            raise FileNotFoundError(f"[ERROR]: Dataset path not found at '{dataset_root_dir}'")

        print("[TRAINING]: Scanning dataset directory...")
        detected_categories = [
            folder for folder in os.listdir(dataset_root_dir)
            if os.path.isdir(os.path.join(dataset_root_dir, folder))
        ]

        if 'noise' not in detected_categories:
            print("[TRAINING WARNING]: 'noise' folder absent — no explicit noise-rejection class.")

        print(f"[TRAINING]: Found {len(detected_categories)} categories: {detected_categories}")

        self.trained_weights = {
            label: np.random.randn(self.vector_size) * 0.01 for label in detected_categories
        }
        self.trained_biases = {label: 0.0 for label in detected_categories}

        # Build the full list of (file_path, multi-label binary truth) pairs.
        # FIX: this loop must be nested — the original code built folder_path in
        # one loop and then read os.listdir(folder_path) in a second, separate
        # loop, so it only ever scanned the *last* label's folder.
        training_samples = []
        for label in detected_categories:
            folder_path = os.path.join(dataset_root_dir, label)
            for audio_file in os.listdir(folder_path):
                if audio_file.endswith(('.wav', '.flac', '.mp3')):
                    sample_path = os.path.join(folder_path, audio_file)
                    binary_truth = {cat: (1.0 if cat == label else 0.0) for cat in detected_categories}
                    training_samples.append((sample_path, binary_truth))

        num_samples = len(training_samples)
        if num_samples == 0:
            raise ValueError("[ERROR]: No audio files found under the dataset directory.")

        print(f"[TRAINING]: Indexed {num_samples} training samples. Starting optimization...")

        for epoch in range(epochs):
            np.random.shuffle(training_samples)

            for i in range(0, num_samples, batch_size):
                batch = training_samples[i:i + batch_size]

                batch_gradients_w = {label: np.zeros(self.vector_size) for label in detected_categories}
                batch_gradients_b = {label: 0.0 for label in detected_categories}
                valid_in_batch = 0

                # FIX: this loop, and the gradient accumulation inside it, must be
                # nested inside the batch loop above. The original code had them
                # de-indented to function scope, so every epoch trained on only
                # the single last sample of the single last batch.
                for sample_path, target_truths in batch:
                    try:
                        y, _ = librosa.load(sample_path, sr=self.target_sr, mono=True)
                        if len(y) == 0:
                            continue
                        y = y / (np.max(np.abs(y)) + 1e-8)
                        y_pure = medfilt(y, kernel_size=3)
                        zcr_noise = np.mean(librosa.feature.zero_crossing_rate(y=(y - y_pure)))
                        X, _ = self._extract_23d_feature_vector(y_pure, zcr_noise)
                    except Exception:
                        continue  # skip corrupted/unreadable files

                    valid_in_batch += 1
                    for label in detected_categories:
                        current_w = self.trained_weights[label]
                        current_b = self.trained_biases[label]

                        prediction = self._safe_sigmoid(np.dot(current_w, X) + current_b)
                        error = target_truths[label] - prediction  # dL/dz for cross-entropy + sigmoid

                        batch_gradients_w[label] += error * X
                        batch_gradients_b[label] += error

                if valid_in_batch == 0:
                    continue  # every file in this batch failed to load; skip the update

                for label in detected_categories:
                    self.trained_weights[label] += (learning_rate / valid_in_batch) * batch_gradients_w[label]
                    self.trained_biases[label] += (learning_rate / valid_in_batch) * batch_gradients_b[label]

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f" -> [EPOCH {epoch + 1}/{epochs}] complete.")

        self.is_model_trained = True
        print("[TRAINING]: Optimization complete. Model is ready for inference.")

    def process_audio_pipeline(self, audio_path):
        """
        Runs inference on a mixed polyphonic audio file. Returns detected
        instrument names plus their isolated frequency-over-time tracks
        (separated from the shared point cloud via DBSCAN).
        """
        if not self.is_model_trained:
            return {"status": "ERROR", "message": "Model not trained/loaded.", "detections": []}

        try:
            y, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        except Exception as e:
            return {"status": "ERROR", "message": f"Failed to load audio: {str(e)}", "detections": []}

        if y is None or len(y) == 0:
            return {"status": "ERROR", "message": "Empty audio input.", "detections": []}

        max_amp = np.max(np.abs(y))
        if max_amp < 1e-4:
            return {
                "status": "SUCCESS",
                "message": "Silence detected.",
                "detected_instrument_names": [],
                "trajectories": {},
            }

        y = y / max_amp
        y_pure = medfilt(y, kernel_size=3)
        y_noise = y - y_pure
        zcr_noise = (
            float(np.mean(librosa.feature.zero_crossing_rate(y=y_noise)))
            if len(y_noise) > 0
            else 0.0
        )

        # FIX: the original code computed X here, then overwrote/ignored it with
        # `X_global = np.mean(accumulated_vectors, axis=0)`, where
        # `accumulated_vectors` was never defined anywhere — that line only
        # existed to crash. Removed; X is used directly below.
        X, point_cloud = self._extract_23d_feature_vector(y_pure, zcr_noise)

        probabilities = {}
        for label in self.trained_weights.keys():
            probabilities[label] = self._safe_sigmoid(
                np.dot(self.trained_weights[label], X) + self.trained_biases[label]
            )

        if 'noise' in probabilities and probabilities['noise'] > self.noise_threshold:
            return {
                "status": "SUCCESS",
                "message": "Signal rejected by noise-floor filter.",
                "detected_instrument_names": ["ENVIRONMENTAL_NOISE"],
                "total_instrument_count": 1,
                "trajectories": {},
            }

        detected_instruments = [
            label for label, prob in probabilities.items()
            if label != 'noise' and prob > self.instrument_threshold
        ]

        # =========================================================================
        # DBSCAN POINT-CLOUD TRACK SEPARATION
        # FIX: this whole block used to run unconditionally (outside the
        # `if len(detected_instruments) > 0 ...` guard), so it would crash on an
        # empty point cloud, and the per-cluster trajectory-building code was
        # outside the `for c_id in set(cluster_labels)` loop, so only one
        # cluster's trajectory was ever kept. Also removed the first, unused
        # DBSCAN(eps=40.0, ...) instantiation — only one DBSCAN call is needed.
        # =========================================================================
        extracted_trajectories = {}
        if len(detected_instruments) > 0 and len(point_cloud) > 10:
            data_points = np.array(point_cloud)

            # Rescale time vs. frequency so Euclidean distance treats
            # milliseconds and Hz on comparable footing.
            scaled_features = data_points[:, :2] * np.array([1000.0, 1.0])

            dbscan = DBSCAN(eps=25.0, min_samples=15, n_jobs=-1)
            cluster_labels = dbscan.fit_predict(scaled_features)

            trajectory_index = 0
            for c_id in set(cluster_labels):
                if c_id == -1:
                    continue  # drop noise points (mode-mixing artifacts)

                member_mask = (cluster_labels == c_id)
                cluster_nodes = data_points[member_mask]

                sorted_idx = np.argsort(cluster_nodes[:, 0])
                t_vector = cluster_nodes[sorted_idx, 0]
                f_vector = cluster_nodes[sorted_idx, 1]
                a_vector = cluster_nodes[sorted_idx, 2]

                if trajectory_index < len(detected_instruments):
                    assigned_name = detected_instruments[trajectory_index]
                else:
                    assigned_name = detected_instruments[-1]

                extracted_trajectories[f"track_{trajectory_index}_{assigned_name}"] = {
                    "time_vector_seconds": t_vector.tolist(),
                    "frequency_vector_hz": f_vector.tolist(),
                    "amplitude_vector_db": a_vector.tolist(),
                    "estimated_average_pitch_hz": float(np.mean(f_vector)),
                }
                trajectory_index += 1

        display_labels = [name.capitalize() for name in detected_instruments]
        sys_msg = (
            f"Analysis complete. Separated {len(display_labels)} source track(s)."
            if display_labels
            else "No instruments above threshold."
        )

        return {
            "status": "SUCCESS",
            "message": sys_msg,
            "detected_instrument_names": display_labels,
            "total_instrument_count": len(display_labels),
            "trajectories": extracted_trajectories,
        }


if __name__ == "__main__":
    print("[LOG]: Booting system test sequence...")
    # FIX: original referenced a nonexistent class `PatentedUltimateAudioEngine`.
    engine = Double_notes_architecture()

    # ---------------------------------------------------------------------
    # DEMO 1: Train on real source directories.
    # Arrange your dataset like:
    #   ./mock_dataset/noise/sample1.wav
    #   ./mock_dataset/piano/chord3.wav
    #   ./mock_dataset/violin/note7.wav
    # ---------------------------------------------------------------------
    # engine.automated_gradient_descent_trainer("./mock_dataset/", epochs=30, learning_rate=0.02)
    # engine.save_weights_to_json("production_brain.json")

    # ---------------------------------------------------------------------
    # DEMO 2: Run inference using mock (untrained-but-valid) weights, just to
    # exercise the pipeline end to end without a real dataset on hand.
    # ---------------------------------------------------------------------
    print("[LOG]: Generating mock weights to simulate a trained model...")
    mock_labels = ['noise', 'piano', 'violin', 'guitar', 'flute']
    engine.trained_weights = {label: np.random.randn(engine.vector_size) * 0.05 for label in mock_labels}
    engine.trained_biases = {label: -0.2 for label in mock_labels}
    engine.is_model_trained = True

    # Round-trip test of the JSON save/load.
    engine.save_weights_to_json("temp_test_weights.json")
    engine.load_weights_from_json("temp_test_weights.json")

    print("[LOG]: Sanity checks passed. Pipeline ready for real audio input.")


















# New innovative architecture 
# Transient-Gated H2MB (Hilbert-Huang Transform and Mel-Frequency Cepstral Coefficients Baseline) architecture


import os
import json
import warnings
import numpy as np
import librosa
from scipy.signal import medfilt, hilbert
from PyEMD import EMD

class TransientGatedH2MB:
    """
    Transient-Gated Conditional Compute Audio Engine.
    Executes heavy EMD/HHT demodulation ONLY on high-flux attack frames (minority),
    routing steady-state frames (majority) to lightweight STFT feature extractors.
    """
    def __init__(self, target_sr=16000, n_mfcc=13, flux_threshold_std=1.5):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.flux_threshold_std = flux_threshold_std
        self.emd_engine = EMD()
        
        # Dual-Path Vector Dimensions:
        # Steady STFT Path = 13 MFCCs + 3 STFT Stats (Centroid, Spread, Flux) = 16 D
        # Transient HHT Path = 9 HHT Descriptors + 1 Cross Term = 10 D
        # Total Unified Vector = 26 D
        self.vector_size = self.n_mfcc + 3 + 10
        self.trained_weights = {}
        self.trained_biases = {}
        self.is_model_trained = False

    def _safe_sigmoid(self, z):
        clipped_z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped_z))

    def _compute_spectral_flux(self, S):
        """Calculates positive spectral difference across consecutive STFT frames."""
        diff = np.diff(S, axis=1)
        # Half-wave rectification (keep only positive energy increases)
        pos_diff = np.maximum(0, diff)
        flux = np.sum(pos_diff, axis=0)
        return np.pad(flux, (1, 0), mode='edge')

    def _extract_stft_features(self, S, freqs):
        """Fast vectorized feature extraction for steady-state regions."""
        power = S ** 2
        sum_power = np.sum(power, axis=0) + 1e-8
        
        # Frame-wise Centroid & Spread
        centroid = np.sum(freqs[:, None] * power, axis=0) / sum_power
        spread = np.sqrt(np.sum(((freqs[:, None] - centroid)**2) * power, axis=0) / sum_power)
        
        flux = self._compute_spectral_flux(S)
        
        return np.mean(centroid), np.mean(spread), np.mean(flux)

    def _extract_hht_features_on_slice(self, y_slice):
        """Expensive EMD/Hilbert demodulation executed ONLY on isolated transient slices."""
        if len(y_slice) < 64:
            return np.zeros(10)

        try:
            imfs = self.emd_engine(y_slice)
        except Exception:
            imfs = np.array([y_slice])

        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_slice])

        inst_freqs, inst_amps = [], []
        target_imfs = min(2, len(imfs))

        for i in range(target_imfs):
            imf = imfs[i]
            if np.all(imf == 0): continue
            analytic = hilbert(imf)
            amp = np.abs(analytic)[:-1]
            phase = np.unwrap(np.angle(analytic))
            freq = np.clip((np.diff(phase) / (2.0 * np.pi)) * self.target_sr, 0, self.target_sr / 2.0)
            
            inst_amps.append(amp)
            inst_freqs.append(freq)

        if not inst_freqs:
            return np.zeros(10)

        # Track 1 (Primary IMF) and Track 2 (All IMFs)
        f1, a1 = inst_freqs[0], inst_amps[0]
        sum_a1 = np.sum(a1) + 1e-8
        t1_centroid = np.sum(f1 * a1) / sum_a1
        t1_spread = np.sqrt(max(0.0, np.sum(((f1 - t1_centroid)**2) * a1) / sum_a1))
        t1_flux = float(np.mean(np.diff(f1)**2)) if len(f1) > 1 else 0.0

        if len(inst_freqs) > 1:
            f2 = np.concatenate(inst_freqs)
            a2 = np.concatenate(inst_amps)
            sum_a2 = np.sum(a2) + 1e-8
            t2_centroid = np.sum(f2 * a2) / sum_a2
            t2_spread = np.sqrt(max(0.0, np.sum(((f2 - t2_centroid)**2) * a2) / sum_a2))
            t2_flux = float(np.mean(np.diff(f1)**2)) if len(f1) > 1 else 0.0
        else:
            t2_centroid, t2_spread, t2_flux = t1_centroid, t1_spread, t1_flux

        pitch_gap = abs(t1_centroid - t2_centroid)
        micro_beating = float(np.mean(np.diff(a1, n=2)**2)) if len(a1) > 2 else 0.0
        zcr_transient = float(np.mean(librosa.feature.zero_crossing_rate(y=y_slice)))

        hht_vector = np.array([
            t1_centroid, t1_spread, t1_flux,
            t2_centroid, t2_spread, t2_flux,
            pitch_gap, micro_beating, zcr_transient,
            t1_centroid * micro_beating  # Cross-term
        ])
        return np.nan_to_num(hht_vector)

    def extract_cascade_features(self, y):
        """Core conditional routing mechanism."""
        # Normalize signal
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        # 1. Single cheap STFT pass over full audio clip
        n_fft, hop_length = 512, 256
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=n_fft)

        # 2. Extract global MFCCs (cheap)
        mfcc_matrix = librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=self.target_sr, n_mfcc=self.n_mfcc)
        mfcc_vector = np.mean(mfcc_matrix, axis=1)

        # 3. Compute transient mask via spectral flux thresholding
        frame_flux = self._compute_spectral_flux(S)
        threshold = np.mean(frame_flux) + (self.flux_threshold_std * np.std(frame_flux))
        transient_mask = frame_flux > threshold

        # 4. Route steady frames -> STFT Path
        steady_mask = ~transient_mask
        if np.any(steady_mask):
            stft_centroid, stft_spread, stft_flux = self._extract_stft_features(S[:, steady_mask], freqs)
        else:
            stft_centroid, stft_spread, stft_flux = self._extract_stft_features(S, freqs)
        
        steady_vector = np.array([stft_centroid, stft_spread, stft_flux])

        # 5. Route transient frames -> Heavy EMD/HHT Path
        transient_frame_indices = np.where(transient_mask)[0]
        
        if len(transient_frame_indices) > 0:
            hht_vectors = []
            # Group contiguous transient frames into slices
            for idx in transient_frame_indices:
                sample_start = idx * hop_length
                sample_end = min(sample_start + n_fft, len(y))
                y_slice = y[sample_start:sample_end]
                
                hht_vec = self._extract_hht_features_on_slice(y_slice)
                hht_vectors.append(hht_vec)
            
            transient_vector = np.mean(hht_vectors, axis=0)
        else:
            # Fallback for perfectly flat/sustained audio with no attacks
            transient_vector = np.zeros(10)

        # 6. Fuse into single 26-D Unified Feature Vector
        unified_vector = np.concatenate([mfcc_vector, steady_vector, transient_vector])
        return np.nan_to_num(unified_vector)

    def process_audio(self, audio_path, instrument_threshold=0.5):
        if not self.is_model_trained:
            return {"status": "ERROR", "message": "Model parameters uninitialized."}

        y, _ = librosa.load(audio_path, sr=self.target_sr, mono=True)
        if len(y) == 0:
            return {"status": "ERROR", "message": "Empty audio file."}

        X = self.extract_cascade_features(y)

        # Logistic Regression Inference
        detections = []
        for label, w in self.trained_weights.items():
            b = self.trained_biases[label]
            prob = self._safe_sigmoid(np.dot(w, X) + b)
            if prob > instrument_threshold:
                detections.append((label, float(prob)))

        return {
            "status": "SUCCESS",
            "detected_instruments": detections,
            "feature_vector_size": len(X)
        }




# revised version of the innovative architecture

import numpy as np
import librosa
from scipy.signal import hilbert
from PyEMD import EMD

class TransientGatedH2MB:
    """
    Revised Transient-Gated Conditional Compute Audio Engine.
    Preserves full chronological time-series matrices (Frames x Features).
    Implements a local rolling threshold window to allow streaming data inputs.
    """
    def __init__(self, target_sr=16000, n_mfcc=13, flux_threshold_std=1.5, local_window_frames=125):
        self.target_sr = target_sr
        self.n_mfcc = n_mfcc
        self.flux_threshold_std = flux_threshold_std
        self.local_window_frames = local_window_frames  # For rolling streaming threshold (~2 seconds)
        self.emd_engine = EMD()
        
        # Unified Frame Feature Layout (26 Dimensions per frame):
        # [0:13]  -> Frame MFCCs (13 D)
        # [13:16] -> Frame STFT Metrics: Centroid, Spread, Flux (3 D)
        # [16:26] -> Frame HHT Demodulation Descriptors (10 D)
        self.feature_dim = self.n_mfcc + 3 + 10
        self.is_model_trained = False

    def _compute_frame_flux(self, S):
        """Calculates positive spectral flux per frame with causal padding for streaming."""
        diff = np.diff(S, axis=1)
        pos_diff = np.maximum(0, diff)
        flux = np.sum(pos_diff, axis=0)
        return np.pad(flux, (1, 0), mode='edge')

    def _extract_frame_stft_metrics(self, S_frame, freqs):
        """Extracts localized spectral indicators for a single frame."""
        power = S_frame ** 2
        sum_power = np.sum(power) + 1e-8
        centroid = np.sum(freqs * power) / sum_power
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / sum_power)
        return centroid, spread
    

    def _extract_hht_features_on_slice(self, y_slice):
        """Executes targeted EMD/Hilbert demodulation on a raw window slice."""
        # Fixed boundary-edge attenuation to mitigate end-effect artifacts
        window = np.hanning(len(y_slice))
        y_windowed = y_slice * window

        if len(y_windowed) < 64:
            return np.zeros(10)

        try:
            imfs = self.emd_engine(y_windowed)
        except Exception:
            imfs = np.array([y_windowed])

        if imfs is None or len(imfs) == 0:
            imfs = np.array([y_windowed])

        inst_freqs, inst_amps = [], []
        target_imfs = min(2, len(imfs))

        for i in range(target_imfs):
            imf = imfs[i]
            if np.all(imf == 0): continue
            analytic = hilbert(imf)
            
            # Truncate outer 5% of boundary samples to drop unstable sifting artifacts
            edge = max(1, int(len(analytic) * 0.05))
            analytic_clean = analytic[edge:-edge]
            
            if len(analytic_clean) <= 1: 
                continue
                
            amp = np.abs(analytic_clean)[:-1]
            phase = np.unwrap(np.angle(analytic_clean))
            freq = np.clip((np.diff(phase) / (2.0 * np.pi)) * self.target_sr, 0, self.target_sr / 2.0)
            
            inst_amps.append(amp)
            inst_freqs.append(freq)

        if not inst_freqs:
            return np.zeros(10)

        # Track 1 Parameters
        f1, a1 = inst_freqs[0], inst_amps[0]
        sum_a1 = np.sum(a1) + 1e-8
        t1_centroid = np.sum(f1 * a1) / sum_a1
        t1_spread = np.sqrt(max(0.0, np.sum(((f1 - t1_centroid)**2) * a1) / sum_a1))
        t1_flux = float(np.mean(np.diff(f1)**2)) if len(f1) > 1 else 0.0

        # Track 2 Parameters
        if len(inst_freqs) > 1:
            f2 = np.concatenate(inst_freqs)
            a2 = np.concatenate(inst_amps)
            sum_a2 = np.sum(a2) + 1e-8
            t2_centroid = np.sum(f2 * a2) / sum_a2
            t2_spread = np.sqrt(max(0.0, np.sum(((f2 - t2_centroid)**2) * a2) / sum_a2))
            t2_flux = float(np.mean(np.diff(f2)**2)) if len(f2) > 1 else 0.0
        else:
            t2_centroid, t2_spread, t2_flux = t1_centroid, t1_spread, t1_flux

        pitch_gap = abs(t1_centroid - t2_centroid)
        micro_beating = float(np.mean(np.diff(a1, n=2)**2)) if len(a1) > 2 else 0.0
        zcr_transient = float(np.mean(librosa.feature.zero_crossing_rate(y=y_slice)))

        hht_vector = np.array([
            t1_centroid, t1_spread, t1_flux,
            t2_centroid, t2_spread, t2_flux,
            pitch_gap, micro_beating, zcr_transient,
            t1_centroid * micro_beating
        ])
        return np.nan_to_num(hht_vector)

    def extract_sequence_features(self, y):
        """
        Processes audio chronologically frame-by-frame.
        Returns a 2D Array [Total Frames, 26 Dimensions] maintaining the time sequence.
        """
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        n_fft, hop_length = 512, 256
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=self.target_sr, n_fft=n_fft)
        
        mfcc_matrix = librosa.feature.mfcc(S=librosa.power_to_db(S**2), sr=self.target_sr, n_mfcc=self.n_mfcc)
        frame_flux = self._compute_frame_flux(S)
        
        total_frames = S.shape[1]
        # Allocate sequence output matrix space
        sequence_matrix = np.zeros((total_frames, self.feature_dim))

        for idx in range(total_frames):
            # 1. Map individual baseline MFCC features
            sequence_matrix[idx, 0:self.n_mfcc] = mfcc_matrix[:, idx]
            
            # 2. Extract baseline STFT structural metrics
            centroid, spread = self._extract_frame_stft_metrics(S[:, idx], freqs)
            current_flux = frame_flux[idx]
            sequence_matrix[idx, self.n_mfcc:self.n_mfcc+3] = [centroid, spread, current_flux]
            
            # 3. Dynamic Local Gating Calculation (Enables Real-Time/Streaming Simulation)
            start_win = max(0, idx - self.local_window_frames)
            local_flux_history = frame_flux[start_win:idx + 1]
            local_threshold = np.mean(local_flux_history) + (self.flux_threshold_std * np.std(local_flux_history))
            
            # 4. Conditional Route Engagement
            if current_flux > local_threshold:
                sample_start = idx * hop_length
                sample_end = min(sample_start + n_fft, len(y))
                y_slice = y[sample_start:sample_end]
                
                # Execute targeted heavy physics
                hht_features = self._extract_hht_features_on_slice(y_slice)
                sequence_matrix[idx, self.n_mfcc+3:] = hht_features
            else:
                # Keep HHT path slots dead/zero for steady-state profiles
                sequence_matrix[idx, self.n_mfcc+3:] = 0.0
                
        return np.nan_to_num(sequence_matrix)

















































# Best in this situation (Returns different instruments with separation of different notes in each instruments)
import librosa
import numpy as np
from sklearn.decomposition import NMF

def extract_polyphonic_time_frequency(audio_path: str, n_sources: int = 4, sr: int = 22050, threshold_ratio: float = 0.2):
    """
    Extracts time-frequency lists for polyphonic/multi-instrument audio using CQT + NMF.
    
    Returns:
        time_stamps (np.ndarray): Array of timestamps in seconds for each frame.
        tf_dict (dict): Dictionary mapping each component/source to a list of (time, active_frequencies_hz) tuples.
    """
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    hop_length = 512
    
    # 2. Compute CQT (84 bins = 7 octaves, 12 bins per octave)
    fmin = librosa.note_to_hz('C1')
    cqt = librosa.cqt(y, sr=sr, fmin=fmin, n_bins=84, bins_per_octave=12, hop_length=hop_length)
    V = np.abs(cqt)
    
    # Frequency corresponding to each of the 84 CQT bins
    bin_freqs = librosa.cqt_frequencies(n_bins=84, fmin=fmin, bins_per_octave=12)
    # Timestamps for each frame in seconds
    time_stamps = librosa.frames_to_time(np.arange(V.shape[1]), sr=sr, hop_length=hop_length)
    
    # 3. Apply NMF: V ≈ W * H
    nmf = NMF(n_components=n_sources, init='nndsvda', max_iter=1000, random_state=42)
    W = nmf.fit_transform(V)   # Shape: [84 bins, n_sources]
    H = nmf.components_        # Shape: [n_sources, n_frames]
    
    tf_dict = {f"Instrument_Component_{k+1}": [] for k in range(n_sources)}
    
    # 4. Extract active frequencies per time frame
    for k in range(n_sources):
        # Identify prominent frequency peaks for component k (fundamental + overtones)
        comp_spectrum = W[:, k]
        peak_threshold = np.max(comp_spectrum) * 0.15
        active_bins = np.where(comp_spectrum > peak_threshold)[0]
        component_freqs = [round(float(bin_freqs[b]), 2) for b in active_bins]
        
        # Determine temporal activation threshold for source k
        activation_thresh = np.max(H[k, :]) * threshold_ratio
        
        # Iterate across time frames
        for t_idx, t_sec in enumerate(time_stamps):
            activation_level = H[k, t_idx]
            if activation_level > activation_thresh:
                # Store (time_in_seconds, list_of_active_frequencies_hz)
                tf_dict[f"Instrument_Component_{k+1}"].append((round(float(t_sec), 3), component_freqs))
                
    return time_stamps, tf_dict



# Output 

{
  "Instrument_Component_1": [
      (0.12, [65.41, 130.81, 196.22]),  # Time 0.12s: C2 fundamental + overtones
      (0.14, [65.41, 130.81, 196.22]),
      ...
  ],
  "Instrument_Component_2": [
      (0.12, [293.66, 587.33]),         # Time 0.12s: D4 played simultaneously
      (0.14, [293.66, 587.33]),
      ...
  ]
}