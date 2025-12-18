# src/hrv_analysis.py
import numpy as np
from scipy import signal, stats, fft
import pywt
import pandas as pd

class HRVAnalyzer:
    def __init__(self, rr_intervals, sampling_rate=4):
        """
        Initialize HRV Analyzer
        
        Parameters:
        -----------
        rr_intervals : array-like
            RR intervals in milliseconds
        sampling_rate : float
            Resampling rate for frequency analysis (Hz)
        """
        self.rr = np.array(rr_intervals)
        self.fs = sampling_rate
        
    def time_domain_analysis(self):
        """Calculate time-domain HRV metrics"""
        if len(self.rr) < 10:
            return {}
        
        rr_diff = np.diff(self.rr)
        
        metrics = {
            'mean_rr': np.mean(self.rr),
            'std_rr': np.std(self.rr),  # SDNN
            'mean_hr': 60000 / np.mean(self.rr),  # Mean heart rate
            'std_hr': 60000 / self.rr.std() if self.rr.std() > 0 else 0,
            'rmssd': np.sqrt(np.mean(rr_diff ** 2)),  # Root mean square of successive differences
            'sdsd': np.std(rr_diff),  # Standard deviation of successive differences
            'nn50': np.sum(np.abs(rr_diff) > 50),  # Number of pairs differing by >50ms
            'pnn50': (np.sum(np.abs(rr_diff) > 50) / len(rr_diff)) * 100,  # Percentage
            'nn20': np.sum(np.abs(rr_diff) > 20),  # Number of pairs differing by >20ms
            'pnn20': (np.sum(np.abs(rr_diff) > 20) / len(rr_diff)) * 100,
            'triangular_index': len(self.rr) / np.max(np.histogram(self.rr, bins=128)[0]),
            'tinn': self._calculate_tinn(),  # Triangular interpolation of NN interval histogram
        }
        
        return metrics
    
    def _calculate_tinn(self):
        """Calculate TINN (Triangular Interpolation of NN Interval Histogram)"""
        if len(self.rr) < 10:
            return 0
        
        # Create histogram
        hist, bin_edges = np.histogram(self.rr, bins=int(np.sqrt(len(self.rr))))
        
        # Find maximum bin
        max_bin_idx = np.argmax(hist)
        
        # Fit triangle
        left_idx = max_bin_idx
        right_idx = max_bin_idx
        
        # Find left base
        while left_idx > 0 and hist[left_idx] > 0:
            left_idx -= 1
        
        # Find right base
        while right_idx < len(hist) - 1 and hist[right_idx] > 0:
            right_idx += 1
        
        # Calculate base width
        tinn = bin_edges[right_idx] - bin_edges[left_idx]
        
        return tinn
    
    def frequency_domain_analysis(self):
        """Calculate frequency-domain HRV metrics"""
        # Lower threshold for short segments (PTB-XL is ~10s)
        if len(self.rr) < 10:
            return {}
        
        # Interpolate RR intervals to equidistant sampling
        t = np.cumsum(self.rr) / 1000  # Convert to seconds
        t -= t[0]
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        f = interp1d(t, self.rr, kind='cubic')
        
        # Create new time axis with fixed sampling rate
        t_new = np.arange(t[0], t[-1], 1/self.fs)
        rr_interp = f(t_new)
        
        # Remove linear trend
        rr_detrended = signal.detrend(rr_interp)
        
        # Compute Power Spectral Density using Welch's method
        f, psd = signal.welch(rr_detrended, fs=self.fs, nperseg=min(256, len(rr_detrended)//2))
        
        # Define frequency bands (Hz)
        vlf_band = (0.003, 0.04)  # Very Low Frequency
        lf_band = (0.04, 0.15)    # Low Frequency
        hf_band = (0.15, 0.4)     # High Frequency
        
        # Calculate power in each band
        vlf_power = self._band_power(f, psd, vlf_band)
        lf_power = self._band_power(f, psd, lf_band)
        hf_power = self._band_power(f, psd, hf_band)
        total_power = vlf_power + lf_power + hf_power
        
        # Calculate normalized powers
        lf_nu = (lf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        hf_nu = (hf_power / (lf_power + hf_power)) * 100 if (lf_power + hf_power) > 0 else 0
        
        metrics = {
            'total_power': total_power,
            'vlf_power': vlf_power,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_power / hf_power if hf_power > 0 else 0,
            'lf_nu': lf_nu,  # Normalized LF
            'hf_nu': hf_nu,  # Normalized HF
            'peak_vlf': f[np.argmax(psd[(f >= vlf_band[0]) & (f < vlf_band[1])])] if vlf_power > 0 else 0,
            'peak_lf': f[np.argmax(psd[(f >= lf_band[0]) & (f < lf_band[1])])] if lf_power > 0 else 0,
            'peak_hf': f[np.argmax(psd[(f >= hf_band[0]) & (f < hf_band[1])])] if hf_power > 0 else 0,
        }
        
        return metrics
    
    def _band_power(self, frequencies, psd, band):
        """Calculate power in a specific frequency band"""
        idx = (frequencies >= band[0]) & (frequencies < band[1])
        if np.any(idx):
            return np.trapz(psd[idx], frequencies[idx])
        return 0
    
    def nonlinear_analysis(self):
        """Calculate nonlinear HRV metrics"""
        if len(self.rr) < 10:
            return {}
        
        # Poincaré plot analysis
        rr_n = self.rr[:-1]
        rr_n1 = self.rr[1:]
        
        # SD1 and SD2
        sd1 = np.std(rr_n1 - rr_n) / np.sqrt(2)
        sd2 = np.std(rr_n1 + rr_n) / np.sqrt(2)
        
        # Sample entropy
        sampen = self._calculate_sample_entropy(self.rr, m=2, r=0.2*np.std(self.rr))
        
        # Detrended fluctuation analysis (DFA)
        alpha1, alpha2 = self._calculate_dfa()
        
        # Recurrence quantification analysis (simplified)
        recurrence_rate = self._calculate_recurrence_rate()
        
        metrics = {
            'sd1': sd1,
            'sd2': sd2,
            'sd1_sd2_ratio': sd1 / sd2 if sd2 > 0 else 0,
            'sample_entropy': sampen,
            'dfa_alpha1': alpha1,
            'dfa_alpha2': alpha2,
            'recurrence_rate': recurrence_rate,
        }
        
        return metrics
    
    def _calculate_sample_entropy(self, time_series, m=2, r=0.2):
        """Calculate Sample Entropy"""
        if len(time_series) < 20: 
            return 0
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            x = [[time_series[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = 0
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j and _maxdist(x[i], x[j]) <= r:
                        C += 1
            return C
        
        N = len(time_series)
        return -np.log(_phi(m + 1) / _phi(m)) if _phi(m) > 0 else 0
    
    def _calculate_dfa(self):
        """Calculate Detrended Fluctuation Analysis"""
        if len(self.rr) < 20:
            return 0, 0
        
        # Integrated series
        y = np.cumsum(self.rr - np.mean(self.rr))
        
        # Define window sizes
        window_sizes = np.logspace(np.log10(4), np.log10(len(self.rr)//4), 20).astype(int)
        window_sizes = window_sizes[window_sizes <= len(self.rr)//4]
        
        # Calculate fluctuation function
        F = []
        for n in window_sizes:
            # Divide into windows
            y_segments = [y[i:i+n] for i in range(0, len(y) - n, n)]
            
            # Detrend each segment
            fluctuations = []
            for segment in y_segments:
                if len(segment) > 1:
                    x = np.arange(len(segment))
                    p = np.polyfit(x, segment, 1)
                    y_fit = np.polyval(p, x)
                    fluctuations.append(np.sqrt(np.mean((segment - y_fit) ** 2)))
            
            if fluctuations:
                F.append(np.mean(fluctuations))
            else:
                F.append(0)
        
        # Fit lines to get alpha1 (short-term) and alpha2 (long-term)
        log_n = np.log10(window_sizes)
        log_F = np.log10(F)
        
        # Split into short and long term
        split_idx = len(log_n) // 2
        
        # Short-term scaling exponent
        if split_idx > 1:
            p1 = np.polyfit(log_n[:split_idx], log_F[:split_idx], 1)
            alpha1 = p1[0]
        else:
            alpha1 = 0
        
        # Long-term scaling exponent
        if split_idx < len(log_n) - 1:
            p2 = np.polyfit(log_n[split_idx:], log_F[split_idx:], 1)
            alpha2 = p2[0]
        else:
            alpha2 = 0
        
        return alpha1, alpha2
    
    def _calculate_recurrence_rate(self):
        """Calculate recurrence rate for recurrence quantification analysis"""
        if len(self.rr) < 20:
            return 0
        
        # Normalize data
        rr_norm = (self.rr - np.mean(self.rr)) / np.std(self.rr)
        
        # Create recurrence matrix
        N = len(rr_norm)
        threshold = 0.2 * np.std(rr_norm)
        recurrence_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if abs(rr_norm[i] - rr_norm[j]) < threshold:
                    recurrence_matrix[i, j] = 1
        
        # Remove diagonal
        np.fill_diagonal(recurrence_matrix, 0)
        
        # Calculate recurrence rate
        recurrence_rate = np.sum(recurrence_matrix) / (N * (N - 1))
        
        return recurrence_rate
    
    def comprehensive_analysis(self):
        """Run comprehensive HRV analysis"""
        results = {
            'time_domain': self.time_domain_analysis(),
            'frequency_domain': self.frequency_domain_analysis(),
            'nonlinear': self.nonlinear_analysis()
        }
        
        # Combine all metrics
        all_metrics = {}
        for category, metrics in results.items():
            all_metrics.update(metrics)
        
        return all_metrics
    
    def interpret_results(self, metrics):
        """Provide clinical interpretation of HRV results"""
        interpretation = []
        risk_level = "Low"
        risk_score = 0
        
        # Time domain interpretation
        if metrics.get('std_rr', 0) < 50:
            interpretation.append("Reduced overall HRV (SDNN < 50 ms)")
            risk_score += 1
        
        if metrics.get('rmssd', 0) < 20:
            interpretation.append("Reduced parasympathetic activity (RMSSD < 20 ms)")
            risk_score += 1
        
        if metrics.get('pnn50', 0) < 5:
            interpretation.append("Reduced beat-to-beat variability (pNN50 < 5%)")
            risk_score += 0.5
        
        # Frequency domain interpretation
        lf_hf_ratio = metrics.get('lf_hf_ratio', 0)
        if lf_hf_ratio < 0.5 or lf_hf_ratio > 3.0:
            interpretation.append(f"Altered autonomic balance (LF/HF ratio: {lf_hf_ratio:.2f})")
            risk_score += 1
        
        # Nonlinear interpretation
        if metrics.get('sample_entropy', 0) < 1.0:
            interpretation.append("Reduced complexity (Sample Entropy < 1.0)")
            risk_score += 0.5
        
        # Determine overall risk
        if risk_score >= 2.5:
            risk_level = "High"
            interpretation.append("HIGH RISK: Possible heart failure or severe autonomic dysfunction")
        elif risk_score >= 1.5:
            risk_level = "Moderate"
            interpretation.append("MODERATE RISK: Further cardiac evaluation recommended")
        else:
            risk_level = "Low"
            interpretation.append("LOW RISK: Normal HRV patterns detected")
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'interpretation': interpretation,
            'recommendations': self._generate_recommendations(risk_level)
        }
    
    def _generate_recommendations(self, risk_level):
        """Generate clinical recommendations based on risk level"""
        recommendations = {
            'Low': [
                "Continue regular monitoring",
                "Maintain healthy lifestyle",
                "Annual cardiac checkup recommended"
            ],
            'Moderate': [
                "Consult cardiologist for evaluation",
                "Consider 24-hour Holter monitoring",
                "Implement lifestyle modifications",
                "Follow up in 3-6 months"
            ],
            'High': [
                "URGENT: Consult cardiologist immediately",
                "Consider echocardiography and stress testing",
                "Close monitoring required",
                "May need medication or intervention"
            ]
        }
        return recommendations.get(risk_level, [])

# Test the HRV analyzer
if __name__ == "__main__":
    # Generate synthetic RR intervals
    np.random.seed(42)
    n_beats = 300
    base_rr = 800  # ms
    variability = 50  # ms
    
    # Healthy HRV pattern
    healthy_rr = base_rr + variability * np.random.randn(n_beats)
    healthy_rr = np.clip(healthy_rr, 500, 1200)
    
    # Heart failure pattern (reduced variability)
    hf_rr = base_rr + 15 * np.random.randn(n_beats)
    hf_rr = np.clip(hf_rr, 500, 1200)
    
    # Analyze healthy pattern
    print("=== HEALTHY HRV ANALYSIS ===")
    healthy_analyzer = HRVAnalyzer(healthy_rr)
    healthy_metrics = healthy_analyzer.comprehensive_analysis()
    healthy_interpretation = healthy_analyzer.interpret_results(healthy_metrics)
    
    print(f"Risk Level: {healthy_interpretation['risk_level']}")
    print(f"Risk Score: {healthy_interpretation['risk_score']:.2f}")
    print("\nInterpretation:")
    for item in healthy_interpretation['interpretation']:
        print(f"  - {item}")
    
    print("\n=== HEART FAILURE PATTERN ANALYSIS ===")
    hf_analyzer = HRVAnalyzer(hf_rr)
    hf_metrics = hf_analyzer.comprehensive_analysis()
    hf_interpretation = hf_analyzer.interpret_results(hf_metrics)
    
    print(f"Risk Level: {hf_interpretation['risk_level']}")
    print(f"Risk Score: {hf_interpretation['risk_score']:.2f}")
    print("\nInterpretation:")
    for item in hf_interpretation['interpretation']:
        print(f"  - {item}")