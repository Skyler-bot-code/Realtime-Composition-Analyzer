from Online_audio import online_audio
from Audio_Importer import audio_importer 


# This model in this file is to develope stft ML-based parameter learning model, 
# to learn the best parameters for stft, 
# and then use those parameters to stretch the audio in a way that preserves the pitch and tempo of the original audio.



## Larger and smaller window sizes for different frequencies (Dynamic STFT)
# Use Short time fast fourier model, but design a model that has dynamic parameters allocation (STFT parameters—like window size, hop length, and FFT size)
# Weltch's method 




# Parameter 1 (n_fft)
# Frequency resolution = sample rate / each frames(n_fft)

# Parameter 2 (win_length)
# How much audio is actually used 
# Ex: win_length = 1024, remaining (2048-1024) will be zeros

# Parameter 3 (Hop_length)
# Frame size, overlaps

# Overlap percentages 

# Parameter 4 (Sampling Rate)
# Ex: Taking 100 photos per second makes the video smooth

# Parameter 5 (Window Function)
# Adaptive window 


# The problem needed to solve is time vs frequency resolution tradeoff
# Small window: good time resolution, poor frequency resolution
# Large window: Poor time resolution, good frequency resolution


# Solution:

"""
Combine differentialbe adaptive stft with spectral-feature driven adaptive stft, saying that 
1. an algorithm runs while stft is running, but this algorithm goes through the next position (frame) in stft through a mathematical equation 
2. And then detect the chaos for certain shift in amplitude and energy to determine the window (frame) size and various different parameters, and return back to the stft



go look at: 
https://www.google.com/search?q=when+will+frequency+be+a+perfect+sine+curve&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRirAjIHCAQQIRirAjIHCAUQIRifBTIHCAYQIRifBTIHCAcQIRifBTIHCAgQIRifBTIHCAkQIRifBdIBCDk0MDhqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8&fbs=ABfTbFVyMZGZf1hfvX9uKjN_-G8c4u0nXx4bEIpwm1lnNH832UV2GkpKEK36HumGR10Evz0DE9hkPgm5pwJD6Y4g4gdUnrwTgnTiuY9xU8D56OFy27CWno0tmsB3TNHX8y-8cBcj9NEJapOKomqRyLG0NeYB48TKlPl-3BTtFoVamK7F0S_gV7G_TI02xC0sBDhBchs8Px36ZdR6TMrSvwuxatlKqnSsFg&aep=10&ntc=1&sxsrf=APpeQnt3STE1_qlnhPCZ0SdclketLlvG8A%3A1786136992380&mstk=AUtExfDr-uRCFZLCOG-kmcHlUtfaOqUEpeY1EwYnk14-QcuvzeC48swAsQUdJzuj4zM5YYRZW1XYlYg_mg5PHjBn_74kEX3q5nPWfsaQURzbicOxghP5EyGwYtoqcLK5ljagWDrwoCVtDK89MGv2zaekDbl3r4Tyv5eSoupJbs9fSiivdDnw_gVPAEpSdNrHforeeyDmQz7Y28kXJPy2x4ubpmX8SY42ZDwMTbwldtyO1szLK30RSxrw_Uwyg67ac12PrNy7c9DdFObOmT63TDXmzdWWD8xpAddJHqI&aioh=3&csuir=1&cs=0&ccb=1&hl=zh-TW&biw=1470&bih=773&atvm=2&mtid=qkl2aquCIdeQm9cP4YKriQs&udm=50

|
V
connect to a nueral network (AI) for playing instrument recommendation

"""



# 1. Looking ahead engine

# 2. Differentiable adapative stft layer

# 3. Cognitive Music Neural network

# With frames greater than 2048 will be more detailed but takes more time 




# Purpose: 自動校正功能 stats analysis --> feedback --> change parameters --> analyze again 

"""
Onset - 
Cubic spline - overshoot
Might need to do it on getting area for warp function 

"""

# Can include DTW validation inside this algorithm also 



# Logic:

# Class mathematical functions

# try making the loops using list-for combination
# Ex: res = [my_func(i) for i in range(1000000)] 

# Index 1:
## detect parameters (0)
## change parameters (none)

# Index 2:
## detect parameters (1)
## change parameters (0)

# Index n:
## detect parameters (n)
## change parameters (n-1)


# Can use import intertools to make infinite list 



