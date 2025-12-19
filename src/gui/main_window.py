import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from src.data_loader import DataLoader
from src.preprocessing import SignalPreprocessor
from src.qrs_detector import QRSDetector
from src.hrv_analysis import HRVAnalyzer
from src.heart_failure_detector import HeartFailureDetector

class HeartMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardiac Health Monitor (Spin-off)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.data = None
        self.analysis_results = None
        
        # Initialize modules
        self.data_loader = DataLoader()
        self.preprocessor = SignalPreprocessor()
        self.qrs_detector = QRSDetector()
        self.hf_detector = HeartFailureDetector()
        
        self.init_ui()

    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)
        
        # Left Panel: Controls & Results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Data Loading Control
        load_group = QGroupBox("1. Data Loading")
        load_layout = QVBoxLayout()
        self.load_btn = QPushButton("Load CSV/PTB-XL File")
        self.load_btn.clicked.connect(self.load_data)
        self.sample_btn = QPushButton("Use Sample Data")
        self.sample_btn.clicked.connect(self.use_sample_data)
        self.data_label = QLabel("No data loaded.")
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.sample_btn)
        load_layout.addWidget(self.data_label)
        load_group.setLayout(load_layout)
        left_layout.addWidget(load_group)
        
        # Analysis Control
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
        
        # Results & Interpretation
        result_group = QGroupBox("3. Interpretation & Risk")
        result_layout = QVBoxLayout()
        self.risk_label = QLabel("Risk Level: N/A")
        self.risk_label.setStyleSheet("font-weight: bold; font-size: 14pt;")
        self.interpretation_text = QTextEdit()
        self.interpretation_text.setReadOnly(True)
        self.alerts_list = QListWidget()
        result_layout.addWidget(self.risk_label)
        result_layout.addWidget(QLabel("Interpretation:"))
        result_layout.addWidget(self.interpretation_text)
        result_layout.addWidget(QLabel("Alerts:"))
        result_layout.addWidget(self.alerts_list)
        result_group.setLayout(result_layout)
        left_layout.addWidget(result_group)
        
        main_layout.addWidget(left_panel, 1)
        
        # Right Panel: Plots & Metrics
        right_panel = QTabWidget()
        
        # Tab 1: Signals
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        self.ecg_plot = pg.PlotWidget(title="ECG Signal")
        self.rr_plot = pg.PlotWidget(title="RR Intervals")
        signal_layout.addWidget(self.ecg_plot)
        signal_layout.addWidget(self.rr_plot)
        right_panel.addTab(signal_tab, "Signals")
        
        # Tab 2: Metrics Table
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        right_panel.addTab(self.metrics_table, "HRV Metrics")
        
        main_layout.addWidget(right_panel, 2)

    def load_data(self):
        # Updated filter to include WFDB formats
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Open ECG Data (CSV or PTB-XL)", 
            "", 
            "ECG Files (*.csv *.dat *.hea);;All Files (*.*)"
        )
        if path:
            self.data = self.data_loader.load_ecg_record(path)
            self._on_data_loaded()

    def use_sample_data(self):
        self.data = self.data_loader.create_sample_data()
        self._on_data_loaded()
        
    def _on_data_loaded(self):
        fs = self.data['sampling_rate'].iloc[0]
        duration = self.data['time'].iloc[-1]
        self.data_label.setText(f"Loaded: {len(self.data)} samples\nFS: {fs}Hz, Duration: {duration:.1f}s")
        self.analyze_btn.setEnabled(True)
        
        # Plot raw signal
        self.ecg_plot.clear()
        # Ensure we pass numpy values, not Series
        self.ecg_plot.plot(self.data['time'].values, self.data['ecg'].values, pen='b')

    def run_analysis(self):
        if self.data is None: return
        
        fs = self.data['sampling_rate'].iloc[0]
        signal = self.data['ecg'].values
        
        # 1. Preprocessing
        steps = []
        if self.baseline_chk.isChecked(): steps.append('baseline')
        if self.denoise_chk.isChecked(): steps.append('denoise')
        processed_signal = self.preprocessor.preprocess_pipeline(signal, fs, steps)
        
        # Plot processed signal (convert time to numpy values)
        self.ecg_plot.clear()
        self.ecg_plot.plot(self.data['time'].values, processed_signal, pen='g', name='Processed')
        
        # 2. QRS Detection
        r_peaks = self.qrs_detector.pan_tompkins_detector(processed_signal, fs)
        rr_intervals = self.qrs_detector.find_rr_intervals(r_peaks, fs)
        clean_rr = self.qrs_detector.clean_rr_intervals(rr_intervals)
        
        # Plot R-peaks and RR intervals
        # FIX IS HERE: Added .values to self.data['time'].iloc[r_peaks]
        if len(r_peaks) > 0:
            peak_times = self.data['time'].iloc[r_peaks].values
            peak_vals = processed_signal[r_peaks]
            self.ecg_plot.plot(peak_times, peak_vals, pen=None, symbol='o', symbolBrush='r')
        
        self.rr_plot.clear()
        if len(clean_rr) > 0:
            self.rr_plot.plot(clean_rr, pen='r', symbol='x')
        
        # 3. HRV Analysis
        analyzer = HRVAnalyzer(clean_rr)
        metrics = analyzer.comprehensive_analysis()
        
        # 4. Heart Failure Detection & Interpretation
        result = self.hf_detector.detect(metrics)
        
        # Update UI with results
        self._update_results_ui(metrics, result)
        QMessageBox.information(self, "Analysis Complete", f"Analysis finished.\nRisk Level: {result['risk_level']}")

    def _update_results_ui(self, metrics, result):
            # Clear previous alerts
            self.alerts_list.clear()
            
            # Update Risk and Diagnosis (Replacing prose text analysis)
            self.risk_label.setText(f"{result['risk_level']}")
            
            # Display simplified Metrics and Diagnosis in the text box
            summary = result['metrics_summary']
            display_text = (
                f"DIAGNOSIS: {result['diagnosis']}\n"
                f"{'-'*30}\n"
                f"HEART RATE:    {summary['HR']}\n"
                f"RATE STATUS:   {summary['Rate Status']}\n"
                f"HRV STATUS:    {summary['HRV Status']}\n"
                f"SDNN:          {summary['SDNN']}\n"
                f"TIMESTAMP:     {result['timestamp']}"
            )
            self.interpretation_text.setText(display_text)
            
            # Add a specific alert if the recording is short
            if self.data is not None and len(self.data) / self.data['sampling_rate'].iloc[0] < 30:
                self.alerts_list.addItem("Note: 10s sample used. Interpret with clinical caution.")
                
            # Update full metrics table as before
            self.metrics_table.setRowCount(len(metrics))
            for i, (key, value) in enumerate(metrics.items()):
                self.metrics_table.setItem(i, 0, QTableWidgetItem(key))
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.2f}"))