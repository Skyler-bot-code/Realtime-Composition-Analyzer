"""
The statistical models here only evaluates efficiency and accuracy, cannot change the results after analyzing and improve efficiency or accuracy, which will be considered a nueral network


"""



#[ Step 1 Validation ] ──> [ Step 2 Validation ] ──> [ Step 3 Validation ] ──> [ Step 4 Validation ]
# Onset Precision/F1        DTW Path MAE & Cost       Spline Cross-Val (RMSE)    Log-Spectral Distance


# For step 1 and 3, try using Cubic Spline Interpolation has its own formal Error Bounding Theorem

# This theorem validates three critical assumptions in your engineering pipeline:
# Step 1 Quality (Sampling Density): It proves whether your note detector found enough timestamp markers ($T_A, T_B$) to safely reconstruct the performance timeline.
# Step 3 Stability (Mathematical Fidelity): It proves that your continuous Cubic Spline curve $S(t)$ doesn't wiggle, overshoot, or oscillate unrealistically (Runge's phenomenon) between two far-apart notes.
# Audio Quality (Psychoacoustic Safety): It proves that the mathematical error between your interpolated curve and the performer's actual continuous pace stays below $10\text{ ms}$—the threshold at which the human ear begins to detect timing artifacts.
from Audio_Importer import AudioImporter

class stats_validate:
    
    def __init__(self):
        self
    
    def on_set(self):
        self
        # weltch's t-test, looking at the onset detection function 
        # Mann-Whitney U test 
        # Inter-Onset Interval (IOI) value for multimodability test and outlier filtering 

    
from Main_Pipeline import onset_stats # Onset accuracy evaluation 
onset_result = print(onset_stats)

from Main_Pipeline import calculate_rse # DTW alignment accuracy 
DTW_result = print(calculate_rse)




    