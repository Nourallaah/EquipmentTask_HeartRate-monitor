# src/preprocessing.py
import numpy as np
from scipy import signal
import pywt

class SignalPreprocessor:
    def __init__(self, sampling_rate=1000):
        self.fs = sampling_rate  # Store in constructor
        
    def preprocess_pipeline(self, ecg_signal, steps=['baseline', 'bandpass', 'denoise']):
        """Complete preprocessing pipeline - NO sampling_rate parameter here!"""
        processed = ecg_signal.copy()
        
        for step in steps:
            if step == 'baseline':
                processed = self.remove_baseline_wander(processed)
            elif step == 'bandpass':
                processed = self.bandpass_filter(processed)
            elif step == 'denoise':
                processed = self.wavelet_denoising(processed)
            elif step == 'powerline':
                processed = self.powerline_interference_removal(processed)
                
        return processed
    
    def remove_baseline_wander(self, signal_data, cutoff=0.5):
        """Remove baseline wander using self.fs"""
        nyquist = self.fs / 2
        high = cutoff / nyquist
        b, a = signal.butter(4, high, btype='high')
        return signal.filtfilt(b, a, signal_data)
    
    def bandpass_filter(self, signal_data, lowcut=0.5, highcut=45.0):
        """Apply bandpass filter using self.fs"""
        nyquist = self.fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, signal_data)
    
    def powerline_interference_removal(self, signal_data, powerline_freq=50):
        """Remove powerline interference using self.fs"""
        nyquist = self.fs / 2
        freq = powerline_freq / nyquist
        b, a = signal.iirnotch(freq, 30)
        return signal.filtfilt(b, a, signal_data)
    
    def wavelet_denoising(self, signal_data, wavelet='db4', level=4):
        """Wavelet denoising - doesn't need sampling rate"""
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        denoised = pywt.waverec(coeffs, wavelet)
        
        # Trim to original length
        if len(denoised) > len(signal_data):
            denoised = denoised[:len(signal_data)]
        elif len(denoised) < len(signal_data):
            denoised = np.pad(denoised, (0, len(signal_data) - len(denoised)))
            
        return denoised