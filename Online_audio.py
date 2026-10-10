import yt_dlp
from Import_Audio import AudioImporter

"""
Steps to install ffmpeg and ffmprobe on python:
1. python3 -m pip install -U yt-dlp #Install yt-dlp
2. python3 -m yt_dlp --version # Check for version
3. Download ffmpeg and ffprobe from online
4. mkdir ~/ffmpeg # Create a ffmpeg folder 
5. mv ~/Downloads/ffmpeg ~/ffmpeg/    mv ~/Downloads/ffprobe ~/ffmpeg/ # Move the downloaded files into the ffmpeg folder
6. ls ~/ffmpeg # Check if the files are in the folder
7. If zsh: killed, then xattr -d com.apple.quarantine ~/ffmpeg/ffmpeg     xattr -d com.apple.quarantine ~/ffmpeg/ffprobe
8. Run your python code
"""

url = "https://www.youtube.com/watch?v=-vH2b55TSHI"

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "/Users/skyler/Desktop/audio.%(ext)s",

    "ffmpeg_location": "/Users/skyler/ffmpeg",

    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])



wav_path = "/Users/skyler/Desktop/audio.wav"

audio_file = AudioImporter(wav_path)

audio_file.load_audio()

times, frequencies, peak_amplitudes, rms_amplitudes = audio_file.detect_pitch()
