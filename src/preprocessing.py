import numpy as np
from scipy import signal
import pywt

class SignalPreprocessor:
    def preprocess_pipeline(self, ecg_signal, sampling_rate, steps=['baseline', 'denoise']):
        """Complete preprocessing pipeline."""
        processed = ecg_signal.copy()
        
        for step in steps:
            if step == 'baseline':
                processed = self.remove_baseline_wander(processed, sampling_rate)
            elif step == 'denoise':
                processed = self.wavelet_denoising(processed)
                
        return processed
    
    def remove_baseline_wander(self, signal_data, fs, cutoff=0.5):
        """Remove baseline wander using a high-pass filter."""
        nyquist = fs / 2
        high = cutoff / nyquist
        b, a = signal.butter(4, high, btype='high')
        return signal.filtfilt(b, a, signal_data)
    
    def wavelet_denoising(self, signal_data, wavelet='db4', level=4):
        """Remove noise using wavelet transform."""
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