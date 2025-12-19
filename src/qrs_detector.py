import numpy as np
from scipy.signal import butter, lfilter

class QRSDetector:
    def pan_tompkins_detector(self, ecg_signal, sampling_rate):
        # 1. Bandpass Filter (5-15 Hz)
        nyquist = 0.5 * sampling_rate
        b, a = butter(1, [5.0/nyquist, 15.0/nyquist], btype='band')
        filtered_ecg = lfilter(b, a, ecg_signal)
        
        # 2. Derivative & 3. Squaring
        # We prepend to maintain array length after differentiation
        squared = np.diff(filtered_ecg, prepend=filtered_ecg[0])**2
        
        # 4. Moving Window Integration (150ms window)
        window_size = int(0.150 * sampling_rate)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        # 5. Thresholding
        integrated /= (np.max(integrated) if np.max(integrated) > 0 else 1)
        threshold = 0.15 # Adjusted for clean synthetic peaks
        peaks = []
        min_distance = int(0.250 * sampling_rate) # 250ms refractory period
        last_peak = -min_distance
        
        for i in range(1, len(integrated)-1):
            if integrated[i] > threshold and integrated[i] > integrated[i-1] and integrated[i] > integrated[i+1]:
                if i - last_peak > min_distance:
                    peaks.append(i)
                    last_peak = i
                    
        return np.array(peaks)

    def find_rr_intervals(self, r_peaks, sampling_rate):
        """Convert R-peak indices into RR intervals in milliseconds."""
        if len(r_peaks) < 2: 
            return np.array([])
        return (np.diff(r_peaks) / sampling_rate) * 1000

    def clean_rr_intervals(self, rr_intervals):
        """Restored method to filter out physiologically impossible intervals."""
        if len(rr_intervals) == 0:
            return rr_intervals
        # Filters for human heart range: 30bpm to 200bpm
        valid_mask = (rr_intervals > 300) & (rr_intervals < 2000)
        return rr_intervals[valid_mask]