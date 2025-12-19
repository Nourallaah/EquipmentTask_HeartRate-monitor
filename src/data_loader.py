import pandas as pd
import numpy as np
import wfdb
import os

class DataLoader:
    def load_ecg_record(self, filepath):
        """Loads MIT-BIH style records (.dat/.hea) for the 360Hz standard."""
        try:
            record_name = os.path.splitext(filepath)[0]
            signals, fields = wfdb.rdsamp(record_name)
            fs = fields['fs'] 
            
            sig_names = [s.upper() for s in fields['sig_name']]
            ecg_idx = sig_names.index('MLII') if 'MLII' in sig_names else 0
            
            ecg_signal = signals[:, ecg_idx]
            t = np.arange(len(ecg_signal)) / fs
            
            return pd.DataFrame({
                'time': t, 
                'ecg': ecg_signal,
                'sampling_rate': fs
            })
        except Exception as e:
            print(f"Loading error: {e}. Falling back to default sample.")
            return self.create_sample_data()

    def create_sample_data(self):
        """Standard 360Hz 75BPM internal fallback."""
        fs = 360
        t = np.arange(0, 10, 1/fs)
        ecg = 0.5 * np.sin(2 * np.pi * 1.25 * t) # Pure 75 BPM sine
        return pd.DataFrame({'time': t, 'ecg': ecg, 'sampling_rate': fs})