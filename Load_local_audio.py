
from Audio_Importer import AudioImporter

# -----------------------------
# Run the program from computer (Analysis from class AudioImporter)
# -----------------------------

file = "/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav"



audio_file = AudioImporter(file)

audio_file.load_audio()

times, frequencies, peak_amplitudes, rms_amplitudes_B = audio_file.detect_pitch()