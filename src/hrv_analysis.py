import numpy as np
from scipy import signal

class HRVAnalyzer:
    def __init__(self, rr_intervals):
        self.rr = np.array(rr_intervals)
        
    def time_domain_analysis(self):
        """Calculate time-domain HRV metrics."""
        if len(self.rr) < 2:
            return {}
        
        rr_diff = np.diff(self.rr)
        metrics = {
            'mean_rr': np.mean(self.rr),
            'std_rr': np.std(self.rr),  # SDNN
            'mean_hr': 60000 / np.mean(self.rr),
            'rmssd': np.sqrt(np.mean(rr_diff ** 2)),
            'pnn50': (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100,
        }
        return metrics
    
    def frequency_domain_analysis(self):
        """Calculate frequency-domain HRV metrics using Welch's method."""
        if len(self.rr) < 10:
            return {}
        
        # Interpolate RR intervals to equidistant sampling (4Hz)
        fs_interp = 4.0
        t = np.cumsum(self.rr) / 1000
        t -= t[0]
        from scipy.interpolate import interp1d
        f_interp = interp1d(t, self.rr, kind='cubic')
        t_new = np.arange(t[0], t[-1], 1/fs_interp)
        rr_interp = f_interp(t_new)
        
        # Detrend and compute PSD
        rr_detrended = signal.detrend(rr_interp)
        f, psd = signal.welch(rr_detrended, fs=fs_interp, nperseg=min(256, len(rr_detrended)))
        
        # Calculate power in bands
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        lf_power = self._band_power(f, psd, lf_band)
        hf_power = self._band_power(f, psd, hf_band)
        
        metrics = {
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
        }
        return metrics
    
    def _band_power(self, frequencies, psd, band):
        idx = (frequencies >= band[0]) & (frequencies < band[1])
        if np.any(idx):
            return np.trapz(psd[idx], frequencies[idx])
        return 0
    
    def nonlinear_analysis(self):
        """Calculate simple nonlinear HRV metrics (Poincaré)."""
        if len(self.rr) < 2:
            return {}
        
        rr_n = self.rr[:-1]
        rr_n1 = self.rr[1:]
        sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
        sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)
        
        metrics = {
            'sd1': sd1,
            'sd2': sd2,
            'sd1_sd2_ratio': sd1 / sd2 if sd2 > 0 else 0,
        }
        return metrics

    def comprehensive_analysis(self):
        """Run all analysis methods."""
        results = {}
        results.update(self.time_domain_analysis())
        results.update(self.frequency_domain_analysis())
        results.update(self.nonlinear_analysis())
        return results