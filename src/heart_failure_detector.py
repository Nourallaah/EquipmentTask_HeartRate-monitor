import pandas as pd

class HeartFailureDetector:
    def detect(self, hrv_metrics):
        hr = hrv_metrics.get('mean_hr', 0)
        sdnn = hrv_metrics.get('std_rr', 0)
        
        if hr >= 100:
            rate_status, diagnosis, risk = "Tachycardia", "Isolated Tachycardia Detected", "MODERATE RISK"
        elif hr <= 60:
            rate_status, diagnosis, risk = "Bradycardia", "Isolated Bradycardia Detected", "MODERATE RISK"
        else:
            rate_status, diagnosis, risk = "Normal Sinus Rate", "Normal cardiac rhythm", "LOW RISK"
            
        # HRV Check (SDNN < 50ms is standard clinical reduced variability)
        hrv_status = "Reduced Variability" if sdnn < 50 else "Normal Variability"
        
        if hrv_status == "Reduced Variability" and risk != "LOW RISK":
            risk = "HIGH RISK"
            diagnosis = f"Combined Finding: {rate_status} with {hrv_status}"

        return {
            'risk_level': risk,
            'metrics_summary': {
                'HEART_RATE': f"{hr:.1f} BPM",
                'RATE_STATUS': rate_status,
                'HRV_STATUS': hrv_status,
                'SDNN': f"{sdnn:.1f} ms",
                'RMSSD': f"{hrv_metrics.get('rmssd', 0):.1f} ms"
            },
            'diagnosis': diagnosis,
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
        }