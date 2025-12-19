import numpy as np
from scipy.signal import butter, lfilter

class QRSDetector:
    def pan_tompkins_detector(self, ecg_signal, sampling_rate):
        """Implementation of the Pan-Tompkins QRS detection algorithm."""
        # 1. Bandpass Filter (5-15 Hz)
        nyquist = 0.5 * sampling_rate
        low = 5.0 / nyquist
        high = 15.0 / nyquist
        b, a = butter(1, [low, high], btype='band')
        filtered_ecg = lfilter(b, a, ecg_signal)
        
        # 2. Derivative
        derivative = np.diff(filtered_ecg)
        
        # 3. Squaring
        squared = derivative ** 2
        
        # 4. Moving Window Integration
        window_size = int(0.150 * sampling_rate)  # 150 ms window
        integrated_ecg = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Thresholding and Peak Detection
        if np.max(integrated_ecg) > 0:
            integrated_ecg = integrated_ecg / np.max(integrated_ecg)
            
        threshold = 0.3 # relative threshold
        peaks = []
        min_distance = int(0.2 * sampling_rate) # 200 ms refractory period
        last_peak = -min_distance
        
        for i in range(1, len(integrated_ecg)-1):
            if integrated_ecg[i] > threshold and \
               integrated_ecg[i] > integrated_ecg[i-1] and \
               integrated_ecg[i] > integrated_ecg[i+1]:
                if i - last_peak > min_distance:
                    peaks.append(i)
                    last_peak = i
        
        # Refine peaks by looking at original signal
        refined_peaks = []
        search_window = int(0.05 * sampling_rate) # 50ms
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            if start < end:
                window = ecg_signal[start:end]
                local_max_idx = np.argmax(np.abs(window))
                refined_peaks.append(start + local_max_idx)
                
        return np.array(refined_peaks)

    def find_rr_intervals(self, r_peaks, sampling_rate):
        """Calculate RR intervals from R-peak indices."""
        if len(r_peaks) < 2:
            return np.array([])
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / sampling_rate) * 1000
        return rr_ms

    def clean_rr_intervals(self, rr_intervals):
        """Remove physiologically impossible RR intervals."""
        if len(rr_intervals) == 0:
            return rr_intervals
        # Keep intervals between 300ms and 2000ms
        valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
        return rr_intervals[valid_mask]