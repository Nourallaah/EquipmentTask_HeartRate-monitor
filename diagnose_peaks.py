import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import SignalPreprocessor
from src.qrs_detector import QRSDetector
import os

def diagnose():
    loader = DataLoader()
    preprocessor = SignalPreprocessor()
    detector = QRSDetector()
    
    # Load a real-ish file
    data_path = r"d:\Downloads\Equipment_Task6\EquipmentTask_HeartRate-monitor\mit-bih-arrhythmia-database\normal\normal_75.dat"
    df = loader.load_ecg_record(data_path)
    if df is None or df.empty:
        print("Failed to load data")
        return
        
    fs = df['sampling_rate'].iloc[0]
    signal = df['ecg'].values
    
    # Preprocess like the GUI does
    processed = preprocessor.preprocess_pipeline(signal, fs, ['baseline', 'denoise'])
    
    # Run detector
    peaks = detector.pan_tompkins_detector(processed, fs)
    
    print(f"Detected {len(peaks)} peaks")
    
    for i, peak in enumerate(peaks[:10]):
        # Check value at peak
        val_at_peak = processed[peak]
        
        # Check max in vicinity
        start = max(0, peak - 20)
        end = min(len(processed), peak + 21)
        vicinity = processed[start:end]
        max_val = np.max(vicinity)
        max_idx = np.argmax(vicinity) + start
        
        print(f"Peak {i}: idx={peak}, val={val_at_peak:.4f} | Max vicinity: idx={max_idx}, val={max_val:.4f} | Diff: {peak-max_idx} samples")

if __name__ == "__main__":
    diagnose()
