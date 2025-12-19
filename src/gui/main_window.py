import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# Local imports
from src.data_loader import DataLoader
from src.preprocessing import SignalPreprocessor
from src.qrs_detector import QRSDetector
from src.hrv_analysis import HRVAnalyzer
from src.heart_failure_detector import HeartFailureDetector

class HeartMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardiac Health Monitor (MIT-BIH Optimized)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.data = None
        
        # Initialize modules
        self.data_loader = DataLoader()
        self.preprocessor = SignalPreprocessor()
        self.qrs_detector = QRSDetector()
        self.hf_detector = HeartFailureDetector()
        
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Left Panel: Controls & Results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 1. Data Loading Group
        load_group = QGroupBox("1. Data Loading")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load MIT-BIH File (.dat/.hea)")
        self.load_btn.clicked.connect(self.load_data)
        self.sample_btn = QPushButton("Use Sample Data")
        self.sample_btn.clicked.connect(self.use_sample_data)
        self.data_label = QLabel("No data loaded.")
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.sample_btn)
        load_layout.addWidget(self.data_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # 2. Analysis Control Group
        analyze_group = QGroupBox("2. Processing & Analysis")
        analyze_layout = QVBoxLayout()
        self.baseline_chk = QCheckBox("Remove Baseline Wander")
        self.baseline_chk.setChecked(True)
        self.denoise_chk = QCheckBox("Wavelet Denoising")
        self.denoise_chk.setChecked(True)
        self.analyze_btn = QPushButton("Run Analysis")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        analyze_layout.addWidget(self.baseline_chk)
        analyze_layout.addWidget(self.denoise_chk)
        analyze_layout.addWidget(self.analyze_btn)
        analyze_group.setLayout(analyze_layout)
        left_layout.addWidget(analyze_group)
        
        # 3. Interpretation & Risk Group
        result_group = QGroupBox("3. Interpretation & Risk")
        result_layout = QVBoxLayout()
        self.risk_label = QLabel("Status: Ready")
        self.risk_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #2980b9;")
        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.alerts_list = QListWidget()
        result_layout.addWidget(self.risk_label)
        result_layout.addWidget(QLabel("Clinical Summary:"))
        result_layout.addWidget(self.interpretation_text)
        result_layout.addWidget(QLabel("System Alerts:"))
        result_layout.addWidget(self.alerts_list)
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)
        
        main_layout.addWidget(left_panel, 1)
        
        # Right Panel: Visualization & Metrics
        right_panel = QTabWidget()
        
        # Tab 1: Signals
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        self.ecg_plot = pg.PlotWidget(title="ECG Signal")
        self.ecg_plot.setLabel('left', 'Amplitude', units='mV')
        self.ecg_plot.setLabel('bottom', 'Time', units='s')
        self.rr_plot = pg.PlotWidget(title="RR Intervals")
        self.rr_plot.setLabel('left', 'Interval', units='ms')
        self.rr_plot.setLabel('bottom', 'Beat Number')
        signal_layout.addWidget(self.ecg_plot)
        signal_layout.addWidget(self.rr_plot)
        right_panel.addTab(signal_tab, "Signals")
        
        # Tab 2: Metrics Table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        right_panel.addTab(self.metrics_table, "HRV Metrics")
        
        main_layout.addWidget(right_panel, 2)

    def load_data(self):
        """Load MIT-BIH records gracefully"""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open MIT-BIH Record", 
            "", 
            "WFDB Records (*.dat *.hea);;CSV Files (*.csv);;All Files (*.*)"
        )
        if path:
            new_data = self.data_loader.load_ecg_record(path)
            if new_data is not None and not new_data.empty:
                self.data = new_data
                self._on_data_loaded()
            else:
                QMessageBox.critical(self, "Load Error", "Failed to load the record. Check if .dat and .hea files are in the same folder.")

    def use_sample_data(self):
        """Load internal synthetic fallback"""
        self.data = self.data_loader.create_sample_data()
        self._on_data_loaded()
        
    def _on_data_loaded(self):
        """Prepare UI after successful data load"""
        if self.data is None:
            return
            
        fs = self.data['sampling_rate'].iloc[0]
        duration = self.data['time'].iloc[-1]
        self.data_label.setText(f"Loaded: {len(self.data)} samples\nFS: {fs}Hz, Duration: {duration:.1f}s")
        self.analyze_btn.setEnabled(True)
        
        # Initial Raw Plot
        self.ecg_plot.clear()
        self.ecg_plot.plot(self.data['time'].values, self.data['ecg'].values, pen='b')

    def run_analysis(self):
        """Run the full analysis pipeline"""
        if self.data is None: return
        
        try:
            fs = self.data['sampling_rate'].iloc[0]
            raw_signal = self.data['ecg'].values
            
            # 1. Preprocessing
            steps = []
            if self.baseline_chk.isChecked(): steps.append('baseline')
            if self.denoise_chk.isChecked(): steps.append('denoise')
            processed_signal = self.preprocessor.preprocess_pipeline(raw_signal, fs, steps)
            
            # 2. QRS Detection & RR Calculation
            r_peaks = self.qrs_detector.r_peaks_detector(processed_signal, fs)
            rr_intervals = self.qrs_detector.find_rr_intervals(r_peaks, fs)
            clean_rr = self.qrs_detector.clean_rr_intervals(rr_intervals)
            
            # 3. HRV Metrics
            analyzer = HRVAnalyzer(clean_rr)
            metrics = analyzer.comprehensive_analysis()
            
            # 4. Heart Failure / Rhythm Detection
            # Ensure metrics contains mean_hr for the detector
            if 'mean_hr' not in metrics and len(clean_rr) > 0:
                metrics['mean_hr'] = 60000 / np.mean(clean_rr)
                
            result = self.hf_detector.detect(metrics)
            
            # 5. UI Updates
            self._update_plots(processed_signal, r_peaks, clean_rr)
            self._update_results_ui(metrics, result)
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {str(e)}")

    def _update_plots(self, processed_signal, r_peaks, clean_rr):
        """Update signal visualizations"""
        # Update ECG Plot with Processed Signal and R-Peaks
        self.ecg_plot.clear()
        self.ecg_plot.plot(self.data['time'].values, processed_signal, pen='g', name='Processed')
        
        if len(r_peaks) > 0:
            peak_times = self.data['time'].iloc[r_peaks].values
            peak_vals = processed_signal[r_peaks]
            self.ecg_plot.plot(peak_times, peak_vals, pen=None, symbol='o', symbolBrush='r', name='R-Peaks')
        
        # Update RR Intervals Plot
        self.rr_plot.clear()
        if len(clean_rr) > 0:
            self.rr_plot.plot(clean_rr, pen='r', symbol='x')

    def _update_results_ui(self, metrics, result):
        """Display structured diagnosis and metrics summary"""
        self.risk_label.setText(f"ASSESSMENT: {result['risk_level']}")
        
        # Clinical Summary Formatting
        summary = result['metrics_summary']
        display_text = (
            f"CLINICAL DIAGNOSIS:\n{result['diagnosis']}\n"
            f"{'='*35}\n"
            f"METRICS SUMMARY:\n"
            f"• Heart Rate:   {summary['HEART_RATE']}\n"
            f"• Rate Status:  {summary['RATE_STATUS']}\n"
            f"• HRV Status:   {summary['HRV_STATUS']}\n"
            f"• SDNN Value:   {summary['SDNN']}\n"
            f"• RMSSD Value:  {summary['RMSSD']}\n"
            f"{'='*35}\n"
            f"ANALYSIS TIME:  {result['timestamp']}"
        )
        self.interpretation_text.setText(display_text)
        
        # System Alerts
        self.alerts_list.clear()
        if 'alerts' in result:
            self.alerts_list.addItems(result['alerts'])
        if len(self.data) < 3600: # Warning for samples shorter than 10s
            self.alerts_list.addItem("Caution: Ultra-short sample (<10s).")

        # Full Metrics Table
        self.metrics_table.setRowCount(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(key.upper()))
            val_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
            self.metrics_table.setItem(i, 1, QTableWidgetItem(val_str))