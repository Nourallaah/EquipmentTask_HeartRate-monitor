import os
import wfdb
import pandas as pd
import numpy as np
from config import RAW_DATA_DIR

PTBXL_BASE = "ptb-xl/1.0.3"
META_FILE = "ptbxl_database.csv"

def download_ptbxl(records_limit=3):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("Downloading PTB-XL metadata...")

    # Check for local metadata first
    local_meta = os.path.join(RAW_DATA_DIR, "ptbxl", "ptbxl_database.csv")
    
    try:
        if os.path.exists(local_meta):
            print(f"Using local metadata: {local_meta}")
            meta = pd.read_csv(local_meta, index_col='ecg_id')
        else:
            print("Fetching metadata from PhysioNet...")
            meta_url = f"https://physionet.org/files/{PTBXL_BASE}/{META_FILE}"
            meta = pd.read_csv(meta_url, index_col='ecg_id')
            
        print("Metadata loaded successfully.")
        print(f"Metadata shape: {meta.shape}")
        print(f"Columns: {meta.columns.tolist()}")
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return

    # Use low-resolution ECG records
    if "filename_lr" in meta.columns:
        records = meta["filename_lr"].head(records_limit).values # Use head() to be safe and clear about limit usage
    else:
        print("Error: 'filename_lr' column not found in metadata!")
        return

    print(f"Downloading {len(records)} PTB-XL ECG records...")
    print(f"Target records: {records}")

    print(f"Downloading {len(records)} PTB-XL ECG records...")
    print(f"Target records: {records}")

    import urllib.request

    for record_path in records:
        try:
            record_name = str(record_path)
            # URL construction
            base_url = f"https://physionet.org/files/{PTBXL_BASE}/"
            url_hea = base_url + record_name + ".hea"
            url_dat = base_url + record_name + ".dat"
            
            # Local temp paths
            temp_dir = os.path.join(RAW_DATA_DIR, "temp_dl")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Preserve directory structure in temp (optional, or just flat)
            # Flattening is easier for temp
            flat_name = record_name.replace("/", "_")
            
            # Wfdb requires the local filenames to match what is in the .hea header
            # usually the header refers to the .dat file by its base name (e.g. 00001_lr.dat)
            # So we must save the downloaded files with their original base names in temp_dir
            base_filename = os.path.basename(record_name)
            
            local_hea = os.path.join(temp_dir, base_filename + ".hea")
            local_dat = os.path.join(temp_dir, base_filename + ".dat")
            
            # Download files
            print(f"Downloading {url_hea}...")
            urllib.request.urlretrieve(url_hea, local_hea)
            urllib.request.urlretrieve(url_dat, local_dat)
            
            # Read locally
            # wfdb.rdsamp expects record name without extension
            local_record_path = os.path.join(temp_dir, base_filename)
            
            print(f"Reading local record: {local_record_path}")
            if not os.path.exists(local_record_path + ".hea"):
                print(f"MISSING HEA: {local_record_path}.hea")
            if not os.path.exists(local_record_path + ".dat"):
                print(f"MISSING DAT: {local_record_path}.dat")

            signals, fields = wfdb.rdsamp(local_record_path)
            print(f"Read signals shape: {signals.shape}")

            df = pd.DataFrame(signals, columns=fields["sig_name"])
            df["time"] = np.arange(len(df)) / fields["fs"]
            df["sampling_rate"] = fields["fs"]

            # Save as CSV
            fname = flat_name + ".csv"
            save_path = os.path.join(RAW_DATA_DIR, fname)
            df.to_csv(save_path, index=False)

            print(f"Saved {fname} to {save_path}")
            
            # Clean up temp files (optional)
            try:
                # pass # Keep them for now for debug
                 os.remove(local_hea)
                 os.remove(local_dat)
            except:
                pass

        except Exception as e:
            print(f"Failed {record_path}: {e}")
            import traceback
            traceback.print_exc() # This will ensure we see the stack trace

    print("PTB-XL download complete.")
    print(f"Contents of {RAW_DATA_DIR}: {os.listdir(RAW_DATA_DIR)}")

if __name__ == "__main__":
    download_ptbxl(records_limit=500)
