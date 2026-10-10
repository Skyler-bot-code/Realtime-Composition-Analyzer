import math
import numpy as np
from Import_Audio import AudioImporter


standard_A4 = 440.0  # Standard frequency for note A4 in Hz


class Intonation_Analysis:

    def __init__(self, f=standard_A4):
        self.f = f  # reference pitch in Hz

    def detect_intonation(self, frequency_list, time_list, tolerance_hz=3.0):
        """
        Flags frequencies that deviate from self.f by more than tolerance_hz.
        Returns matching-length (freqs, times) lists.
        """
        out_of_tune_freqs = []
        out_of_tune_times = []

        lower_bound = self.f - tolerance_hz
        upper_bound = self.f + tolerance_hz

        for freq, t in zip(frequency_list, time_list):
            if freq <= lower_bound or freq >= upper_bound:
                out_of_tune_freqs.append(freq)
                out_of_tune_times.append(t)

        return out_of_tune_freqs, out_of_tune_times

    def get_half_step(self, frequency_list):
        """
        Converts each frequency into semitone deviation from self.f.
        Returns a NEW list (does not mutate input).
        """
        half_steps = []
        for freq in frequency_list:
            if freq == self.f or freq <= 0:
                half_steps.append(0)
                continue
            n = 12 * math.log2(freq / self.f)
            half_steps.append(round(n))
        return half_steps

    def detect_consecutive_notes(self, frequency_list, time_list):
        """
        Groups consecutive equal-frequency frames into runs, prints the
        out-of-tune ones.
        """
        j = 0
        while j < len(frequency_list):
            if frequency_list[j] == 0:
                j += 1
                continue

            count = 0
            while (j + count + 1 < len(frequency_list)) and \
                  (frequency_list[j + count + 1] == frequency_list[j]):
                count += 1

            if count == 0:
                print(f"This note at time: {time_list[j]} is out of tune, "
                      f"and should be at frequency {frequency_list[j]}")
            else:
                time1 = time_list[j]
                time2 = time_list[j + count]
                print(f"This note between {time1} and {time2} is out of tune, "
                      f"and should be at frequency {frequency_list[j]}")

            j += count + 1


    def analyze_aligned_decisions(self, aligned, original_times, original_frequencies):
        """
        Parameters
        ----------
        aligned : list of dict
            Output of Compute_optimize_hybrid_mask.align_decisions_to_original_time().
        original_times, original_frequencies : array-like
            Un-stretched audio A's time/frequency tracks (same length/order).

        Returns
        -------
        list of dict — each aligned entry plus 'original_frequency_hz' and
        'half_step_deviation', both computed from the ORIGINAL audio, never
        the stretched/rendered audio.
        """
        original_times = np.asarray(original_times)
        results = []

        for entry in aligned:
            t = entry["original_time_sec"]

            idx = int(np.argmin(np.abs(original_times - t)))
            freq_at_t = float(original_frequencies[idx])
            half_step = self.get_half_step([freq_at_t])[0]

            results.append({
                **entry,
                "original_frequency_hz": freq_at_t,
                "half_step_deviation": half_step,
            })

        return results


if __name__ == "__main__":
    # Standalone test only — does NOT run on import

    audio = AudioImporter("/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav")
    audio.load_audio()
    times, freqs = audio.detect_time_frequency()

    tuner = Intonation_Analysis(standard_A4)
    out_freqs, out_times = tuner.detect_intonation(freqs, times)
    half_steps = tuner.get_half_step(out_freqs)
    tuner.detect_consecutive_notes(freqs, times)