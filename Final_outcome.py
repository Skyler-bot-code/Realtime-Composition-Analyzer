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
from tune_noter import Intonation_Analysis
import scipy.ndimage as ndimage
import librosa
from double_notes_multiple_instruments import Double_notes_architecture
from Two_Data_Comparison_ML import Granular_synthesis, Variable_phasevocoder, Alignment

