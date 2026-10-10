from Audio_Importer import AudioImporter

audio = AudioImporter("/Users/skyler/Desktop/EE + Cello project/cold-percussion-mix.wav")

audio.load_audio()

times, rms_list = audio.detect_rms()


print(rms_list)