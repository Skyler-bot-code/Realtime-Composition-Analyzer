
from Import_Audio import AudioImporter

"""
Load local audio (Analysis from class AudioImporter)
"""

file = "/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav"



audio_file = AudioImporter(file)

audio_file.load_audio()

times, frequencies, peak_amplitudes, rms_amplitudes_B = audio_file.detect_pitch()