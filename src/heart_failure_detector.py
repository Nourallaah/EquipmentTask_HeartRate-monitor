import pandas as pd

class HeartFailureDetector:
    def __init__(self):
        # Adjusted thresholds for ultra-short (10s) recordings
        self.hrv_rules = {
            'std_rr': {'threshold': 20, 'direction': 'below'}, 
            'rmssd': {'threshold': 15, 'direction': 'below'}
        }
        
        # Rate-based thresholds (BPM)
        self.rate_rules = {
            'tachycardia': 100,
            'bradycardia': 60
        }

    def detect(self, hrv_metrics):
        """Analyze metrics to identify HR/HRV status and clinical diagnosis."""
        hr = hrv_metrics.get('mean_hr', 0)
        sdnn = hrv_metrics.get('std_rr', 0)
        rmssd = hrv_metrics.get('rmssd', 0)
        
        # 1. Determine Rate Status
        if hr > self.rate_rules['tachycardia']:
            rate_status = "Tachycardia"
        elif hr < self.rate_rules['bradycardia']:
            rate_status = "Bradycardia"
        else:
            rate_status = "Normal Sinus Rate"
            
        # 2. Determine HRV Significance
        # Reduced variability in short samples can indicate cardiac stress
        if sdnn < self.hrv_rules['std_rr']['threshold'] or rmssd < self.hrv_rules['rmssd']['threshold']:
            hrv_status = "Low Variability"
            risk_points = 2
        else:
            hrv_status = "Normal Variability"
            risk_points = 0

        # 3. Formulate Diagnosis & Risk Level
        if risk_points >= 2 and rate_status != "Normal Sinus Rate":
            diagnosis = f"Potential Cardiac Dysfunction ({rate_status} + {hrv_status})"
            risk_level = "HIGH RISK"
        elif risk_points >= 2:
            diagnosis = f"Isolated {hrv_status}"
            risk_level = "MODERATE RISK"
        elif rate_status != "Normal Sinus Rate":
            diagnosis = f"Isolated {rate_status}"
            risk_level = "MODERATE RISK"
        else:
            diagnosis = "No significant abnormalities detected"
            risk_level = "LOW RISK"
            
        return {
            'risk_level': risk_level,
            'metrics_summary': {
                'HR': f"{hr:.1f} BPM",
                'Rate Status': rate_status,
                'HRV Status': hrv_status,
                'SDNN': f"{sdnn:.1f} ms"
            },
            'diagnosis': diagnosis,
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
        }