import ast
import os
import wfdb
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

class DataLoader:
    def __init__(self):
        self.raw_dir = RAW_DATA_DIR
        self.processed_dir = PROCESSED_DATA_DIR
        self.meta_df = None
        self.scp_map = {
            'NORM': 'Normal ECG',
            'MI': 'Myocardial Infarction',
            'STTC': 'ST/T Change',
            'CD': 'Conduction Disturbance',
            'HYP': 'Hypertrophy'
        }
        self._load_metadata()
        
    def _load_metadata(self):
        """Load PTB-XL metadata"""
        try:
            meta_path = os.path.join(self.raw_dir, "ptbxl", "ptbxl_database.csv")
            if os.path.exists(meta_path):
                self.meta_df = pd.read_csv(meta_path, index_col='ecg_id')
                # Pre-parse scp_codes
                self.meta_df['scp_codes'] = self.meta_df['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                print(f"Loaded metadata: {len(self.meta_df)} records")
        except Exception as e:
            print(f"Failed to load metadata: {e}")
        
    def load_ecg_record(self, filepath):
        """Load ECG record from various file formats"""
        
        # Get file extension
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        try:
            if ext == '.csv':
                # Load CSV file
                df = pd.read_csv(filepath)
                
                # Ensure 'ecg' column exists or find best candidate
                if 'ecg' not in df.columns:
                    # Priority list for single-lead analysis
                    # II is standard for rhythm, V5/V1/V2 are also good
                    priority_leads = ['II', 'ii', 'Lead II', 'V5', 'v5', 'V2', 'v2', 'I', 'i', 'MLII', 'mlii']
                    
                    found_lead = False
                    # Check priority leads first
                    for lead in priority_leads:
                        if lead in df.columns:
                            df = df.rename(columns={lead: 'ecg'})
                            print(f"Selected lead '{lead}' as main ECG signal")
                            found_lead = True
                            break
                    
                    if not found_lead:
                        # Fallback: Look for any column containing 'ecg' or 'signal'
                        possible_ecg_cols = [col for col in df.columns 
                                            if any(x in col.lower() for x in ['ecg', 'eeg', 'signal', 'channel'])]
                        
                        if possible_ecg_cols:
                            df = df.rename(columns={possible_ecg_cols[0]: 'ecg'})
                        else:
                            # Use first numeric column that isn't time/fs
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            valid_cols = [c for c in numeric_cols if c.lower() not in ['time', 'fs', 'sampling_rate', 'sampling_frequency']]
                            if valid_cols:
                                df = df.rename(columns={valid_cols[0]: 'ecg'})
                
                # Ensure 'time' column exists
                if 'time' not in df.columns:
                    # Assume 250 Hz sampling rate
                    sampling_rate = 250
                    df['time'] = np.arange(len(df)) / sampling_rate
                
                # Attach Ground Truth Diagnosis
                if self.meta_df is not None:
                    try:
                        filename = os.path.basename(filepath)
                        # Expecting format like records100_00000_00001_lr.csv
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            # Try to find the numeric ID
                            possible_id = parts[-2]
                            if possible_id.isdigit():
                                ecg_id = int(possible_id)
                                if ecg_id in self.meta_df.index:
                                    scp = self.meta_df.loc[ecg_id, 'scp_codes']
                                    # scp is a dict like {'NORM': 100, 'MI': 50}
                                    # Get entries with likelihood > 0
                                    diagnoses = []
                                    if isinstance(scp, dict):
                                        for code, likelihood in scp.items():
                                            if likelihood > 0:
                                                full_name = self.scp_map.get(code, code)
                                                diagnoses.append(f"{full_name} ({likelihood:.0f}%)")
                                    
                                    if diagnoses:
                                        df.original_diagnosis = ", ".join(diagnoses)
                                    else:
                                        df.original_diagnosis = "Unknown"
                                    print(f"Loaded Diagnosis for {ecg_id}: {df.original_diagnosis}")
                    except Exception as e:
                        print(f"Could not load diagnosis: {e}")

                return df
                
            elif ext in ['.dat', '.hea']:
                # Load WFDB file
                record_name = os.path.splitext(os.path.basename(filepath))[0]
                record_dir = os.path.dirname(filepath)
                
                # Read the record
                signals, fields = wfdb.rdsamp(record_name, pn_dir=record_dir)
                
                # Create DataFrame
                if 'sig_name' in fields and fields['sig_name']:
                    column_names = fields['sig_name']
                else:
                    column_names = [f'channel_{i}' for i in range(signals.shape[1])]
                
                df = pd.DataFrame(signals, columns=column_names)
                
                # Add time column
                df['time'] = np.arange(len(df)) / fields['fs']
                
                # Identify ECG column
                ecg_columns = [col for col in df.columns if col.lower() in ['ecg', 'mlii', 'v1', 'v2', 'v5']]
                if ecg_columns:
                    df = df.rename(columns={ecg_columns[0]: 'ecg'})
                elif len(df.columns) > 0:
                    # Use first column as ECG
                    first_col = df.columns[0]
                    if first_col != 'time':
                        df = df.rename(columns={first_col: 'ecg'})
                
                return df
                
            elif ext == '.ecg':
                # Try to load .ecg as binary file
                print(f"Warning: .ecg files are not standard WFDB format. Trying to read as binary...")
                return self._load_binary_ecg(filepath)
                
            elif ext in ['.txt', '.edf']:
                # Try to load as text file
                try:
                    df = pd.read_csv(filepath, delim_whitespace=True)
                    if len(df.columns) >= 2:
                        df.columns = ['time', 'ecg'][:len(df.columns)]
                    return df
                except:
                    # Try with numpy
                    data = np.loadtxt(filepath)
                    if data.ndim == 1:
                        df = pd.DataFrame({'ecg': data})
                    else:
                        df = pd.DataFrame(data, columns=['time', 'ecg'])
                    return df
                    
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            # Create sample data as fallback
            print(f"Error loading file {filepath}: {e}")
            print("Creating sample data as fallback...")
            return self._create_sample_data()
    
    def _load_binary_ecg(self, filepath):
        """Load binary .ecg file"""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            # Try different formats
            # Try as 16-bit integers
            try:
                data = np.frombuffer(raw_data, dtype=np.int16)
                print(f"Loaded as 16-bit int: {len(data)} samples")
            except:
                # Try as 8-bit
                try:
                    data = np.frombuffer(raw_data, dtype=np.int8)
                    print(f"Loaded as 8-bit int: {len(data)} samples")
                except:
                    # Try as float32
                    try:
                        data = np.frombuffer(raw_data, dtype=np.float32)
                        print(f"Loaded as float32: {len(data)} samples")
                    except:
                        raise ValueError("Cannot parse binary ECG file")
            
            # Normalize
            data = data.astype(np.float32)
            if np.max(np.abs(data)) > 1000:
                data = data / 1000.0
            
            # Create DataFrame
            df = pd.DataFrame({
                'ecg': data,
                'time': np.arange(len(data)) / 250  # Assume 250 Hz
            })
            
            return df
            
        except Exception as e:
            print(f"Failed to load binary ECG: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample ECG data as fallback"""
        fs = 250
        duration = 30
        t = np.arange(0, duration, 1/fs)
        
        # Create ECG signal
        heart_rate = 75
        ecg = np.zeros_like(t)
        
        # Add beats
        for i in range(0, len(t), int(fs * 60/heart_rate)):
            if i < len(ecg):
                # Add QRS complex
                qrs_start = max(0, i - 20)
                qrs_end = min(len(ecg), i + 40)
                ecg[qrs_start:qrs_end] += np.sin(np.linspace(0, np.pi, qrs_end - qrs_start))
        
        # Add noise
        ecg += 0.05 * np.random.randn(len(t))
        
        return pd.DataFrame({
            'time': t,
            'ecg': ecg
        })
    
    def create_sample_dataset(self):
        """Create sample dataset for testing"""
        print("Creating sample datasets...")
        
        # Create normal ECG
        normal_df = self._create_condition_dataset("normal", 75, 15)
        normal_path = os.path.join(self.raw_dir, "normal_ecg.csv")
        normal_df.to_csv(normal_path, index=False)
        print(f"Created: {normal_path}")
        
        # Create heart failure ECG
        hf_df = self._create_condition_dataset("heart_failure", 85, 5)
        hf_path = os.path.join(self.raw_dir, "heart_failure_ecg.csv")
        hf_df.to_csv(hf_path, index=False)
        print(f"Created: {hf_path}")
        
        # Create arrhythmia ECG
        arr_df = self._create_condition_dataset("arrhythmia", 80, 25)
        arr_path = os.path.join(self.raw_dir, "arrhythmia_ecg.csv")
        arr_df.to_csv(arr_path, index=False)
        print(f"Created: {arr_path}")
        
        print("Sample datasets created successfully!")
    
    def _create_condition_dataset(self, condition, base_hr, hrv_strength):
        """Create ECG dataset for specific condition"""
        fs = 250
        duration = 300  # 5 minutes
        t = np.arange(0, duration, 1/fs)
        
        if condition == "arrhythmia":
            # Irregular rhythm
            hr = base_hr + hrv_strength * np.random.randn(len(t))
            hr = np.clip(hr, 40, 180)
        else:
            hr = base_hr + hrv_strength * np.sin(2 * np.pi * 0.01 * t)
        
        ecg = np.zeros_like(t)
        time_since_last = 0
        
        for i in range(len(t)):
            current_hr = max(40, hr[i])
            rr_interval = 60 / current_hr
            
            if time_since_last >= rr_interval:
                # Add QRS
                self._add_qrs(ecg, i, fs, condition)
                time_since_last = 0
            else:
                time_since_last += 1/fs
        
        # Add noise
        ecg += 0.02 * np.random.randn(len(t))
        
        return pd.DataFrame({
            'time': t,
            'ecg': ecg,
            'heart_rate': hr,
            'condition': condition,
            'sampling_rate': fs
        })
    
    def _add_qrs(self, signal, center_idx, fs, condition):
        """Add QRS complex based on condition"""
        if condition == "heart_failure":
            amplitude = 0.8
            width = 0.1
        elif condition == "arrhythmia":
            amplitude = np.random.uniform(0.7, 1.2)
            width = np.random.uniform(0.06, 0.12)
        else:
            amplitude = 1.0
            width = 0.08
        
        qrs_samples = int(width * fs)
        start_idx = max(0, center_idx - qrs_samples//2)
        end_idx = min(len(signal), start_idx + qrs_samples)
        
        for i in range(start_idx, end_idx):
            rel_pos = (i - center_idx) / (qrs_samples//2)
            if abs(rel_pos) < 0.3:
                signal[i] += amplitude * (1 - abs(rel_pos/0.3))

# Test the data loader
if __name__ == "__main__":
    loader = DataLoader()
    print("Creating sample datasets...")
    loader.create_sample_dataset()