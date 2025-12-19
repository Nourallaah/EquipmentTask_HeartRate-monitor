import numpy as np
from scipy.signal import find_peaks, butter, filtfilt

class QRSDetector:
    def _bandpass_filter(self, signal, fs, lowcut=0.5, highcut=30.0, order=2):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal)

    def r_peaks_detector(self, ecg_signal, sampling_rate):
        """
        Simple R-peak detection.
        """
        # distance in samples corresponding to min RR interval
        min_rr_s = 0.2
        distance = int(min_rr_s * sampling_rate)

        # Pre-process for peak detection
        sig = self._bandpass_filter(ecg_signal, sampling_rate)
        
        # Detect peaks on the absolute signal
        abs_sig = np.abs(sig)
        
        # Adaptive prominence based on signal characteristics
        prominence = max(0.3 * np.std(abs_sig), 0.1 * np.max(abs_sig))
        
        # Find peaks on absolute signal
        peaks, properties = find_peaks(
            abs_sig, 
            distance=distance, 
            prominence=prominence,
            height=0.2 * np.max(abs_sig) if np.max(abs_sig) > 0 else 0
        )
        
        # Refine peak positions to actual signal extrema
        refined_peaks = []
        search_window = int(0.04 * sampling_rate)  # 40ms search window
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(sig) - 1, peak + search_window)
            
            # Use ecg_signal for refinement
            segment = ecg_signal[start:end + 1]
            if len(segment) == 0:
                continue
                
            # Find whether positive or negative deflection is stronger
            max_idx = np.argmax(segment)
            min_idx = np.argmin(segment)
            max_val = segment[max_idx]
            min_val = segment[min_idx]
            
            # Choose the larger absolute deflection
            if abs(min_val) > abs(max_val):
                actual_peak = min_idx + start
            else:
                actual_peak = max_idx + start
                
            refined_peaks.append(int(actual_peak))
        
        # Remove duplicates and sort
        refined_peaks = sorted(set(refined_peaks))
        
        # Final refractory period enforcement
        final_peaks = []
        for peak in refined_peaks:
            if not final_peaks or (peak - final_peaks[-1]) >= distance:
                final_peaks.append(peak)
        
        return np.array(final_peaks)

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