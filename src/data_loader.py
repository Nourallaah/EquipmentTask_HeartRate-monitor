import pandas as pd
import numpy as np
import wfdb
import os

class DataLoader:
    def load_ecg_record(self, filepath):
        """Load ECG record from CSV or WFDB (PTB-XL) formats."""
        try:
            _, ext = os.path.splitext(filepath)
            
            if ext.lower() == '.csv':
                return self._load_csv(filepath)
            elif ext.lower() in ['.dat', '.hea']:
                return self._load_wfdb(filepath)
            else:
                raise ValueError("Unsupported file format. Use CSV, DAT, or HEA.")
            
        except Exception as e:
            print(f"Error loading file: {e}. Creating sample data instead.")
            return self.create_sample_data()

    def _load_csv(self, filepath):
        """Helper to load CSV files."""
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        
        # Standardize columns
        if 'ecg' not in df.columns:
            # Fallback for PTB-XL CSVs that might have different headers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            candidates = [c for c in numeric_cols if 'time' not in c and 'fs' not in c]
            if candidates:
                df = df.rename(columns={candidates[0]: 'ecg'})
            else:
                raise ValueError("No ECG column found in CSV.")

        # Determine Sampling Rate
        if 'sampling_rate' in df.columns:
            fs = df['sampling_rate'].iloc[0]
        elif 'fs' in df.columns:
            fs = df['fs'].iloc[0]
        else:
            fs = 500 # PTB-XL high-res default
            print(f"Warning: FS not found in CSV, assuming {fs}Hz")

        if 'time' not in df.columns:
            df['time'] = np.arange(len(df)) / fs
            
        df['sampling_rate'] = fs
        return df

    def _load_wfdb(self, filepath):
        """Helper to load WFDB files (PTB-XL standard)."""
        # wfdb.rdsamp expects the record name without extension
        record_name = os.path.splitext(filepath)[0]
        
        # Read signal and metadata
        signals, fields = wfdb.rdsamp(record_name)
        fs = fields['fs']
        
        # PTB-XL usually has 12 leads. We prioritize Lead II, then V5, then index 0.
        sig_names = [s.lower() for s in fields['sig_name']]
        
        if 'ii' in sig_names:
            ecg_idx = sig_names.index('ii')
        elif 'v5' in sig_names:
            ecg_idx = sig_names.index('v5')
        else:
            ecg_idx = 0
            
        ecg_signal = signals[:, ecg_idx]
        
        # Create standard DataFrame
        t = np.arange(len(ecg_signal)) / fs
        df = pd.DataFrame({
            'time': t, 
            'ecg': ecg_signal,
            'sampling_rate': fs
        })
        return df
    
    def create_sample_data(self, duration_sec=10, fs=500, heart_rate=75):
        """Create a synthetic ECG signal for testing."""
        t = np.arange(0, duration_sec, 1/fs)
        ecg = np.zeros_like(t)
        
        # Baseline and noise
        ecg += 0.05 * np.sin(2 * np.pi * 0.2 * t)
        ecg += 0.01 * np.random.randn(len(t))
        
        # QRS complexes
        rr_samples = int(fs * 60 / heart_rate)
        # Use a simple burst for QRS
        for i in range(0, len(t), rr_samples):
            if i + 20 < len(t):
                ecg[i:i+10] += 1.0  # R-peak up
                ecg[i+10:i+20] -= 0.5 # S-wave down

        return pd.DataFrame({'time': t, 'ecg': ecg, 'sampling_rate': fs})