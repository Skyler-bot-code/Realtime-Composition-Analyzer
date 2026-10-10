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