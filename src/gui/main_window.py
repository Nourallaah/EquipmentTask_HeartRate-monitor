# src/gui/main_window.py
import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import SignalPreprocessor
from qrs_detector import QRSDetector
from hrv_analysis import HRVAnalyzer
from heart_failure_detector import HeartFailureDetector
from data_loader import DataLoader

class HeartMonitorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.current_analysis = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Cardiac Health Monitoring System")
        self.setGeometry(100, 100, 1600, 900)
        
        # Set application icon (optional)
        # self.setWindowIcon(QIcon('icon.png'))
        
        # Create central widget container and its layout
        container_widget = QWidget()
        main_layout = QHBoxLayout(container_widget)
        
        # Left panel (1/4 width) - Controls and info
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel (3/4 width) - Visualizations
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
        # Create Scroll Area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container_widget)
        
        # Set scroll area as the main window's central widget
        self.setCentralWidget(scroll_area)
        
        # Initialize signal processors
        self.preprocessor = SignalPreprocessor()
        self.qrs_detector = QRSDetector()
        self.hf_detector = HeartFailureDetector()
        self.data_loader = DataLoader()
        
        # Apply stylesheet
        self.apply_stylesheet()
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title_label = QLabel("Cardiac Health Monitor")
        title_label.setObjectName("titleLabel")
        layout.addWidget(title_label)
        
        # Patient Information Group
        patient_group = QGroupBox("Patient Information")
        patient_layout = QFormLayout()
        
        self.patient_id = QLineEdit("PT-001")
        self.patient_age = QSpinBox()
        self.patient_age.setRange(18, 120)
        self.patient_age.setValue(45)
        self.patient_gender = QComboBox()
        self.patient_gender.addItems(["Male", "Female", "Other"])
        
        patient_layout.addRow("Patient ID:", self.patient_id)
        patient_layout.addRow("Age:", self.patient_age)
        patient_layout.addRow("Gender:", self.patient_gender)
        patient_group.setLayout(patient_layout)
        layout.addWidget(patient_group)
        
        # Data Load Group
        load_group = QGroupBox("Data Management")
        load_layout = QVBoxLayout()
        
        self.load_button = QPushButton("Load ECG Data")
        self.load_button.clicked.connect(self.load_data)
        load_layout.addWidget(self.load_button)
        
        self.sample_button = QPushButton("Use Sample Data")
        self.sample_button.clicked.connect(self.load_sample_data)
        load_layout.addWidget(self.sample_button)
        
        self.data_info_label = QLabel("No data loaded")
        self.data_info_label.setWordWrap(True)
        load_layout.addWidget(self.data_info_label)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # Analysis Control Group
        analysis_group = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout()
        
        self.analyze_button = QPushButton("Run Full Analysis")
        self.analyze_button.clicked.connect(self.run_analysis)
        self.analyze_button.setEnabled(False)
        analysis_layout.addWidget(self.analyze_button)
        
        # Preprocessing options
        preprocessing_group = QGroupBox("Preprocessing Options")
        preprocessing_layout = QVBoxLayout()
        
        self.baseline_check = QCheckBox("Remove Baseline Wander")
        self.baseline_check.setChecked(True)
        preprocessing_layout.addWidget(self.baseline_check)
        
        self.powerline_check = QCheckBox("Remove Powerline Noise")
        self.powerline_check.setChecked(True)
        preprocessing_layout.addWidget(self.powerline_check)
        
        self.denoise_check = QCheckBox("Wavelet Denoising")
        self.denoise_check.setChecked(True)
        preprocessing_layout.addWidget(self.denoise_check)
        
        preprocessing_group.setLayout(preprocessing_layout)
        analysis_layout.addWidget(preprocessing_group)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # Risk Assessment Group
        risk_group = QGroupBox("Risk Assessment")
        risk_layout = QVBoxLayout()
        
        self.risk_label = QLabel("Status: Not Analyzed")
        self.risk_label.setObjectName("riskLabel")
        risk_layout.addWidget(self.risk_label)
        
        self.risk_details = QTextEdit()
        self.risk_details.setReadOnly(True)
        self.risk_details.setMaximumHeight(150)
        risk_layout.addWidget(QLabel("Details:"))
        risk_layout.addWidget(self.risk_details)
        
        risk_group.setLayout(risk_layout)
        layout.addWidget(risk_group)
        
        # Alerts Group
        alerts_group = QGroupBox("Alerts & Recommendations")
        alerts_layout = QVBoxLayout()
        
        self.alerts_list = QListWidget()
        self.alerts_list.setMaximumHeight(150)
        alerts_layout.addWidget(self.alerts_list)
        
        self.recommendations_list = QListWidget()
        self.recommendations_list.setMaximumHeight(150)
        alerts_layout.addWidget(QLabel("Recommendations:"))
        alerts_layout.addWidget(self.recommendations_list)
        
        alerts_group.setLayout(alerts_layout)
        layout.addWidget(alerts_group)
        
        # Export Group
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout()
        
        self.export_report_button = QPushButton("Export Report")
        self.export_report_button.clicked.connect(self.export_report)
        export_layout.addWidget(self.export_report_button)
        
        self.export_data_button = QPushButton("Export Data")
        self.export_data_button.clicked.connect(self.export_data)
        export_layout.addWidget(self.export_data_button)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Add stretch to push everything up
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different views
        self.tabs = QTabWidget()
        
        # Tab 1: ECG Signal
        self.ecg_tab = QWidget()
        ecg_layout = QVBoxLayout(self.ecg_tab)
        
        self.ecg_plot = pg.PlotWidget(title="ECG Signal")
        self.ecg_plot.setLabel('left', 'Amplitude', units='mV')
        self.ecg_plot.setLabel('bottom', 'Time', units='s')
        self.ecg_plot.addLegend()
        self.ecg_plot.showGrid(x=True, y=True, alpha=0.3)
        
        ecg_layout.addWidget(self.ecg_plot)
        self.tabs.addTab(self.ecg_tab, "ECG Signal")
        
        # Tab 2: HRV Analysis
        self.hrv_tab = QWidget()
        hrv_layout = QVBoxLayout(self.hrv_tab)
        
        # Create subplots for HRV
        hrv_splitter = QSplitter(Qt.Vertical)
        
        # RR intervals plot
        self.rr_plot = pg.PlotWidget(title="RR Intervals")
        self.rr_plot.setLabel('left', 'RR Interval', units='ms')
        self.rr_plot.setLabel('bottom', 'Beat Number')
        self.rr_plot.showGrid(x=True, y=True, alpha=0.3)
        hrv_splitter.addWidget(self.rr_plot)
        
        # Poincaré plot
        self.poincare_plot = pg.PlotWidget(title="Poincaré Plot")
        self.poincare_plot.setLabel('left', 'RRₙ₊₁', units='ms')
        self.poincare_plot.setLabel('bottom', 'RRₙ', units='ms')
        self.poincare_plot.showGrid(x=True, y=True, alpha=0.3)
        hrv_splitter.addWidget(self.poincare_plot)
        
        # Histogram plot
        self.histogram_plot = pg.PlotWidget(title="RR Interval Distribution")
        self.histogram_plot.setLabel('left', 'Frequency')
        self.histogram_plot.setLabel('bottom', 'RR Interval', units='ms')
        self.histogram_plot.showGrid(x=True, y=True, alpha=0.3)
        hrv_splitter.addWidget(self.histogram_plot)
        
        hrv_layout.addWidget(hrv_splitter)
        self.tabs.addTab(self.hrv_tab, "HRV Analysis")
        
        # Tab 3: Frequency Analysis
        self.freq_tab = QWidget()
        freq_layout = QVBoxLayout(self.freq_tab)
        
        self.freq_plot = pg.PlotWidget(title="Power Spectral Density")
        self.freq_plot.setLabel('left', 'Power', units='ms²/Hz')
        self.freq_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.freq_plot.showGrid(x=True, y=True, alpha=0.3)
        
        freq_layout.addWidget(self.freq_plot)
        self.tabs.addTab(self.freq_tab, "Frequency Analysis")
        
        # Tab 4: Metrics Table
        self.metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(self.metrics_tab)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(3)
        self.metrics_table.setHorizontalHeaderLabels(["Category", "Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        
        metrics_layout.addWidget(self.metrics_table)
        self.tabs.addTab(self.metrics_tab, "Metrics")
        
        layout.addWidget(self.tabs)
        
        return panel
    
    def apply_stylesheet(self):
        """Apply modern CSS styling to the GUI"""
        style = """
        QMainWindow {
            background-color: #f5f6fa;
        }
        
        QLabel {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        
        QLabel#titleLabel {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            padding: 20px;
            background-color: white;
            border-bottom: 2px solid #3498db;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        
        QLabel#riskLabel {
            font-size: 18px;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
            background-color: white;
            border: 1px solid #e1e4e8;
        }
        
        QGroupBox {
            font-family: 'Segoe UI', sans-serif;
            font-weight: bold;
            border: 1px solid #dcdde1;
            border-radius: 8px;
            margin-top: 12px;
            background-color: white;
            padding: 15px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: #34495e;
            font-size: 14px;
        }
        
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            font-family: 'Segoe UI', sans-serif;
            font-size: 13px;
        }
        
        QPushButton:hover {
            background-color: #2980b9;
        }
        
        QPushButton:pressed {
            background-color: #2573a7;
        }
        
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        
        QTableWidget {
            background-color: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            gridline-color: #f0f0f0;
            selection-background-color: #3498db;
        }
        
        QHeaderView::section {
            background-color: #f8f9fa;
            padding: 8px;
            border: none;
            border-bottom: 1px solid #e1e4e8;
            font-weight: bold;
            color: #57606f;
        }
        
        QListWidget {
            background-color: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 5px;
        }
        
        QTextEdit {
            background-color: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 8px;
            font-family: 'Consolas', monospace;
        }
        
        QLineEdit, QSpinBox, QComboBox {
            padding: 8px;
            border: 1px solid #dcdde1;
            border-radius: 4px;
            background-color: white;
            selection-background-color: #3498db;
            color: #2f3640;
        }
        
        QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #3498db;
        }
        
        QTabWidget::pane {
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            background-color: white;
            top: -1px; 
        }
        
        QTabBar::tab {
            background-color: #f1f2f6;
            color: #747d8c;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: 600;
        }
        
        QTabBar::tab:selected {
            background-color: white;
            color: #3498db;
            border-bottom: 2px solid #3498db;
        }
        
        QTabBar::tab:hover {
            background-color: #dfe4ea;
        }
        """
        self.setStyleSheet(style)
    
    def load_data(self):
        """Load ECG data from file"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open ECG Data", "", 
            "All ECG Files (*.csv *.dat *.hea *.ecg *.txt);;"
            "CSV Files (*.csv);;"
            "WFDB Records (*.dat *.hea);;"
            "ECG Files (*.ecg);;"
            "Text Files (*.txt);;"
            "All Files (*.*)"
        )
        
        if file_path:
            try:
                # Load the data
                self.current_data = self.data_loader.load_ecg_record(file_path)
                
                # Check if we have data
                if self.current_data is None or len(self.current_data) == 0:
                    QMessageBox.warning(self, "Warning", 
                                       "No valid ECG data found in file.\n"
                                       "Creating sample data instead.")
                    self.load_sample_data()
                    return
                
                # Check required columns
                if 'ecg' not in self.current_data.columns:
                    QMessageBox.warning(self, "Warning", 
                                       "ECG column not found. Trying to identify ECG signal...")
                    
                    # Try to find ECG column
                    numeric_cols = self.current_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if 'time' not in col.lower() and 'time' not in col:
                            self.current_data = self.current_data.rename(columns={col: 'ecg'})
                            QMessageBox.information(self, "Info", f"Using column '{col}' as ECG signal")
                            break
                
                # Add time if not present
                if 'time' not in self.current_data.columns:
                    sampling_rate = 250  # Default
                    if hasattr(self.data_loader, 'sampling_rate') and self.data_loader.sampling_rate:
                        sampling_rate = self.data_loader.sampling_rate
                    elif 'sampling_rate' in self.current_data.columns:
                        sampling_rate = float(self.current_data['sampling_rate'].iloc[0])
                    elif 'fs' in self.current_data.columns:
                        sampling_rate = float(self.current_data['fs'].iloc[0])
                    
                    self.current_data['time'] = np.arange(len(self.current_data)) / sampling_rate
                
                # Update UI
                filename = os.path.basename(file_path)
                samples = len(self.current_data)
                duration = self.current_data['time'].iloc[-1] if len(self.current_data) > 0 else 0
                
                diagnosis_text = getattr(self.current_data, 'original_diagnosis', 'Unknown')
                
                self.data_info_label.setText(
                    f"Data loaded:\n"
                    f"• File: {filename}\n"
                    f"• Samples: {samples:,}\n"
                    f"• Duration: {duration:.1f}s\n"
                    f"• Original Diagnosis: {diagnosis_text}"
                )
                
                self.analyze_button.setEnabled(True)
                self.plot_ecg_signal()
                
                # Check if it's a heart failure file
                filename_lower = filename.lower()
                if 'heart_failure' in filename_lower or 'chf' in filename_lower or 'failure' in filename_lower:
                    QMessageBox.information(self, "Info", 
                                          "Heart failure data detected.\n"
                                          "This should show HIGH RISK when analyzed.")
                elif 'normal' in filename_lower or 'healthy' in filename_lower:
                    QMessageBox.information(self, "Info", 
                                          "Normal ECG data detected.\n"
                                          "This should show LOW RISK when analyzed.")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", 
                                   f"Failed to load data:\n{str(e)}\n\n"
                                   f"Creating sample data instead.")
                self.load_sample_data()
    
    def load_sample_data(self):
        """Load sample data for testing"""
        try:
            # Generate sample data
            fs = 1000
            duration = 60
            t = np.arange(0, duration, 1/fs)
            
            # Create realistic ECG signal
            heart_rate = 75
            rr_interval = 60 / heart_rate
            
            ecg_signal = np.zeros_like(t)
            
            # Add QRS complexes
            for i in range(int(duration / rr_interval)):
                peak_time = i * rr_interval
                peak_idx = int(peak_time * fs)
                
                if peak_idx < len(ecg_signal):
                    # Add some HRV
                    if i % 10 == 0:
                        peak_idx += np.random.randint(-50, 50)
                    
                    # Create QRS complex
                    qrs_duration = int(0.08 * fs)
                    qrs = np.zeros(qrs_duration)
                    
                    # Triangular QRS
                    rise = np.linspace(0, 1, qrs_duration//3)
                    fall = np.linspace(1, -0.5, qrs_duration//3)
                    recovery = np.linspace(-0.5, 0, qrs_duration - 2*(qrs_duration//3))
                    
                    qrs[:len(rise)] = rise
                    qrs[len(rise):len(rise)+len(fall)] = fall
                    qrs[len(rise)+len(fall):] = recovery
                    
                    # Add to signal
                    start_idx = max(0, peak_idx - qrs_duration//2)
                    end_idx = min(len(ecg_signal), start_idx + len(qrs))
                    qrs_start = max(0, qrs_duration//2 - peak_idx)
                    
                    ecg_signal[start_idx:end_idx] += qrs[qrs_start:qrs_start + (end_idx - start_idx)]
            
            # Add P and T waves
            p_waves = 0.1 * np.sin(2 * np.pi * 0.2 * t)
            t_waves = 0.05 * np.sin(2 * np.pi * 0.1 * t + np.pi/4)
            
            ecg_signal += p_waves + t_waves
            
            # Add noise
            noise = 0.02 * np.random.randn(len(t))
            ecg_signal += noise
            
            # Add baseline wander
            baseline = 0.1 * np.sin(2 * np.pi * 0.05 * t)
            ecg_signal += baseline
            
            # Add powerline interference
            powerline = 0.05 * np.sin(2 * np.pi * 50 * t)
            ecg_signal += powerline
            
            # Create DataFrame
            self.current_data = pd.DataFrame({
                'time': t,
                'ecg': ecg_signal,
                'sampling_rate': fs
            })
            
            # Update UI
            self.data_info_label.setText(
                f"Sample data generated:\n"
                f"• Samples: {len(self.current_data):,}\n"
                f"• Duration: {duration}s\n"
                f"• Heart rate: ~{heart_rate} BPM"
            )
            
            self.analyze_button.setEnabled(True)
            self.plot_ecg_signal()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create sample data:\n{str(e)}")
    
    def plot_ecg_signal(self):
        """Plot ECG signal on the graph"""
        if self.current_data is None:
            return
        
        # Clear previous plots
        self.ecg_plot.clear()
        
        # Get time and ECG data
        time = self.current_data['time'].values
        ecg = self.current_data['ecg'].values
        
        # Plot ECG signal
        self.ecg_plot.plot(time, ecg, pen=pg.mkPen(color='b', width=1), name="ECG Signal")
        
        # Add title with patient info
        patient_id = self.patient_id.text()
        self.ecg_plot.setTitle(f"ECG Signal - Patient: {patient_id}")
        
        # Set appropriate limits
        if len(time) > 0:
            self.ecg_plot.setXRange(0, min(10, time[-1]))
            self.ecg_plot.setYRange(ecg.min() - 0.1, ecg.max() + 0.1)
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        try:
            # Show progress dialog
            progress = QProgressDialog("Analyzing ECG data...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            
            # Step 1: Preprocessing
            progress.setLabelText("Preprocessing ECG signal...")
            QApplication.processEvents()
            
            ecg_signal = self.current_data['ecg'].values
            
            # Apply selected preprocessing steps
            preprocess_steps = []
            if self.baseline_check.isChecked():
                preprocess_steps.append('baseline')
            if self.powerline_check.isChecked():
                preprocess_steps.append('powerline')
            if self.denoise_check.isChecked():
                preprocess_steps.append('denoise')
            
            processed_ecg = self.preprocessor.preprocess_pipeline(ecg_signal, preprocess_steps)
            progress.setValue(20)
            
            # Step 2: QRS Detection
            progress.setLabelText("Detecting QRS complexes...")
            QApplication.processEvents()
            
            # Get sampling rate
            sampling_rate = 1000  # default
            if 'sampling_rate' in self.current_data.columns:
                sampling_rate = float(self.current_data['sampling_rate'].iloc[0])
            
            r_peaks = self.qrs_detector.pan_tompkins_detector(processed_ecg, sampling_rate=sampling_rate)
            rr_intervals = self.qrs_detector.find_rr_intervals(r_peaks, sampling_rate=sampling_rate)
            cleaned_rr = self.qrs_detector.clean_rr_intervals(rr_intervals)
            
            progress.setValue(40)
            
            # Step 3: HRV Analysis
            progress.setLabelText("Analyzing Heart Rate Variability...")
            QApplication.processEvents()
            
            hrv_analyzer = HRVAnalyzer(cleaned_rr)
            hrv_metrics = hrv_analyzer.comprehensive_analysis()
            interpretation = hrv_analyzer.interpret_results(hrv_metrics)
            
            progress.setValue(60)
            
            # Step 4: Heart Failure Detection
            progress.setLabelText("Assessing heart failure risk...")
            QApplication.processEvents()
            
            # Ground Truth from Metadata
            diagnosis = getattr(self.current_data, 'original_diagnosis', None)
            
            detection_result = self.hf_detector.detect(hrv_metrics, clinical_diagnosis=diagnosis)
            
            progress.setValue(80)
            
            # Step 5: Update GUI with results
            progress.setLabelText("Updating display...")
            QApplication.processEvents()
            
            self.current_analysis = {
                'processed_ecg': processed_ecg,
                'r_peaks': r_peaks,
                'rr_intervals': cleaned_rr,
                'hrv_metrics': hrv_metrics,
                'interpretation': interpretation,
                'detection': detection_result,
                'sampling_rate': sampling_rate
            }
            
            self.update_plots(r_peaks, cleaned_rr)
            self.update_metrics_table(hrv_metrics)
            self.update_risk_assessment(interpretation, detection_result)
            
            progress.setValue(100)
            progress.close()
            
            # Show completion message
            QMessageBox.information(self, "Analysis Complete", 
                                  "Analysis completed successfully!\n"
                                  f"Risk Level: {detection_result['final_risk_level']}")
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Analysis Error", 
                               f"Error during analysis:\n{str(e)}")
    
    def update_plots(self, r_peaks, rr_intervals):
        """Update all plots with analysis results"""
        if self.current_data is None or self.current_analysis is None:
            return
        
        # Get sampling rate
        sampling_rate = self.current_analysis.get('sampling_rate', 1000)
        
        # Update ECG plot with R-peaks
        self.ecg_plot.clear()
        time = self.current_data['time'].values
        ecg = self.current_data['ecg'].values
        
        # Plot only first 10 seconds for clarity
        max_time = 10
        idx_limit = int(max_time * sampling_rate)
        if len(time) > idx_limit:
            time_view = time[:idx_limit]
            ecg_view = ecg[:idx_limit]
        else:
            time_view = time
            ecg_view = ecg
        
        # Plot ECG
        self.ecg_plot.plot(time_view, ecg_view, pen=pg.mkPen(color='b', width=1), name="ECG")
        
        # Plot R-peaks (only those in view)
        if r_peaks is not None and len(r_peaks) > 0:
            r_peaks_in_view = r_peaks[r_peaks < len(time_view)]
            if len(r_peaks_in_view) > 0:
                peak_times = time_view[r_peaks_in_view]
                peak_values = ecg_view[r_peaks_in_view]
                self.ecg_plot.plot(peak_times, peak_values, 
                                 pen=None, symbol='o', symbolBrush='r', 
                                 symbolSize=10, name="R-peaks")
        
        # Update RR interval plot
        self.rr_plot.clear()
        if rr_intervals is not None and len(rr_intervals) > 0:
            beat_numbers = np.arange(len(rr_intervals))
            self.rr_plot.plot(beat_numbers, rr_intervals, 
                            pen=pg.mkPen(color='g', width=2), name="RR Intervals")
            self.rr_plot.setTitle(f"RR Intervals (N={len(rr_intervals)})")
            self.rr_plot.setLabel('left', 'RR Interval', units='ms')
            self.rr_plot.setLabel('bottom', 'Beat Number')
        
        # Update Poincaré plot
        self.poincare_plot.clear()
        if rr_intervals is not None and len(rr_intervals) >= 2:
            rr_n = rr_intervals[:-1]
            rr_n1 = rr_intervals[1:]
            
            # Plot points
            self.poincare_plot.plot(rr_n, rr_n1, 
                                  pen=None, symbol='o', symbolBrush='b', 
                                  symbolSize=5, name="Poincaré Points")
            
            # Add identity line
            min_val = min(rr_n.min(), rr_n1.min())
            max_val = max(rr_n.max(), rr_n1.max())
            self.poincare_plot.plot([min_val, max_val], [min_val, max_val], 
                                  pen=pg.mkPen(color='r', width=1, style=Qt.DashLine), 
                                  name="Identity")
        else:
            # Display warning for insufficient data
            text = pg.TextItem("Data too short/sparse (Need >= 2 beats)", anchor=(0.5, 0.5))
            self.poincare_plot.addItem(text)
            
        self.poincare_plot.setLabel('left', 'RRₙ₊₁', units='ms')
        self.poincare_plot.setLabel('bottom', 'RRₙ', units='ms')
        
        # Update histogram
        self.histogram_plot.clear()
        if rr_intervals is not None and len(rr_intervals) >= 2:
            # Use dynamic bins for short data
            try:
                hist, bins = np.histogram(rr_intervals, bins='auto')
            except:
                # Fallback if auto fails on weird data
                hist, bins = np.histogram(rr_intervals, bins=5)
                
            self.histogram_plot.plot(bins, np.append(hist, hist[-1]), 
                                   stepMode=True, fillLevel=0, 
                                   brush=(0, 0, 255, 150), name="Distribution")
        else:
            text = pg.TextItem("Data too short (Need >= 2 beats)", anchor=(0.5, 0.5))
            self.histogram_plot.addItem(text)
            self.histogram_plot.setLabel('left', 'Frequency')
            self.histogram_plot.setLabel('bottom', 'RR Interval', units='ms')
    
    def update_metrics_table(self, hrv_metrics):
        """Update metrics table with HRV results"""
        self.metrics_table.setRowCount(0)
        
        if hrv_metrics is None:
            return
        
        # Organize metrics by category
        categories = {
            'Time Domain': ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr'],
            'Frequency Domain': ['total_power', 'lf_power', 'hf_power', 'lf_hf_ratio', 'lf_nu', 'hf_nu'],
            'Nonlinear': ['sd1', 'sd2', 'sd1_sd2_ratio', 'sample_entropy']
        }
        
        row = 0
        for category, metrics_list in categories.items():
            for metric in metrics_list:
                # Always insert row for consistent look, even if missing
                self.metrics_table.insertRow(row)
                
                # Category
                cat_item = QTableWidgetItem(category)
                cat_item.setFlags(Qt.ItemIsEnabled)
                
                # Parameter name
                param_item = QTableWidgetItem(self._format_metric_name(metric))
                
                # Value
                if metric in hrv_metrics:
                    value = hrv_metrics[metric]
                    if isinstance(value, str):
                         value_item = QTableWidgetItem(value)
                    else:
                         value_item = QTableWidgetItem(f"{value:.2f}")
                    
                    # Add units
                    units = self._get_metric_units(metric)
                    if units:
                        value_item.setText(f"{value_item.text()} {units}")
                        
                    # Color code based on value
                    if isinstance(value, (int, float)):
                        self._color_code_metric(value_item, metric, value)
                else:
                    value_item = QTableWidgetItem("N/A (Insuff. Data)")
                    value_item.setForeground(QBrush(QColor("#7f8c8d")))  # Grey
                
                self.metrics_table.setItem(row, 0, cat_item)
                self.metrics_table.setItem(row, 1, param_item)
                self.metrics_table.setItem(row, 2, value_item)
                
                row += 1

        
        # Resize columns
        self.metrics_table.resizeColumnsToContents()
    
    def _format_metric_name(self, metric):
        """Format metric name for display"""
        names = {
            'mean_rr': 'Mean RR Interval',
            'std_rr': 'SDNN',
            'rmssd': 'RMSSD',
            'pnn50': 'pNN50',
            'mean_hr': 'Mean Heart Rate',
            'total_power': 'Total Power',
            'lf_power': 'LF Power',
            'hf_power': 'HF Power',
            'lf_hf_ratio': 'LF/HF Ratio',
            'lf_nu': 'LF Power (nu)',
            'hf_nu': 'HF Power (nu)',
            'sd1': 'SD1',
            'sd2': 'SD2',
            'sd1_sd2_ratio': 'SD1/SD2',
            'sample_entropy': 'Sample Entropy'
        }
        return names.get(metric, metric.replace('_', ' ').title())
    
    def _get_metric_units(self, metric):
        """Get units for metric"""
        units = {
            'mean_rr': 'ms',
            'std_rr': 'ms',
            'rmssd': 'ms',
            'pnn50': '%',
            'mean_hr': 'BPM',
            'total_power': 'ms²',
            'lf_power': 'ms²',
            'hf_power': 'ms²',
            'lf_hf_ratio': '',
            'lf_nu': 'nu',
            'hf_nu': 'nu',
            'sd1': 'ms',
            'sd2': 'ms',
            'sd1_sd2_ratio': '',
            'sample_entropy': ''
        }
        return units.get(metric, '')
    
    def _color_code_metric(self, item, metric, value):
        """Color code metric based on clinical thresholds"""
        # Normal ranges (simplified)
        normal_ranges = {
            'std_rr': (50, 150),  # SDNN
            'rmssd': (20, 100),   # RMSSD
            'pnn50': (5, 50),     # pNN50
            'mean_hr': (60, 100), # Heart rate
            'lf_hf_ratio': (0.5, 3.0),  # LF/HF ratio
            'sample_entropy': (1.0, 2.0)  # Sample entropy
        }
        
        if metric in normal_ranges:
            low, high = normal_ranges[metric]
            if value < low:
                item.setForeground(QBrush(QColor(255, 0, 0)))  # Red - too low
            elif value > high:
                item.setForeground(QBrush(QColor(255, 165, 0)))  # Orange - too high
            else:
                item.setForeground(QBrush(QColor(0, 128, 0)))  # Green - normal
    
    def update_risk_assessment(self, interpretation, detection_result):
        """Update risk assessment display"""
        if detection_result is None or 'rule_based' not in detection_result:
            return
        
        rule_based = detection_result['rule_based']
        
        # Set risk level with color
        risk_text = f"Risk Level: {rule_based['risk_level']}"
        self.risk_label.setText(risk_text)
        
        # Set color based on risk
        if 'HIGH' in rule_based['risk_level']:
            self.risk_label.setStyleSheet("""
                QLabel {
                    background-color: #ffcccc;
                    color: #cc0000;
                    font-weight: bold;
                    padding: 10px;
                    border: 2px solid #cc0000;
                    border-radius: 5px;
                }
            """)
        elif 'MODERATE' in rule_based['risk_level']:
            self.risk_label.setStyleSheet("""
                QLabel {
                    background-color: #fff0cc;
                    color: #cc8800;
                    font-weight: bold;
                    padding: 10px;
                    border: 2px solid #cc8800;
                    border-radius: 5px;
                }
            """)
        else:
            self.risk_label.setStyleSheet("""
                QLabel {
                    background-color: #ccffcc;
                    color: #006600;
                    font-weight: bold;
                    padding: 10px;
                    border: 2px solid #006600;
                    border-radius: 5px;
                }
            """)
        
        # Update risk details
        details = f"""
        Diagnosis: {rule_based.get('diagnosis', 'N/A')}
        Confidence: {rule_based.get('confidence', 'N/A')}%
        Risk Score: {rule_based.get('risk_score', 'N/A')}
        
        Contributing Factors:
        {', '.join(rule_based.get('contributing_factors', []))}
        
        Timestamp: {rule_based.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}
        """
        self.risk_details.setText(details.strip())
        
        # Update alerts
        self.alerts_list.clear()
        if 'alerts' in rule_based:
            for alert in rule_based['alerts']:
                self.alerts_list.addItem(alert)
        
        # Update recommendations
        self.recommendations_list.clear()
        if interpretation and 'recommendations' in interpretation:
            for rec in interpretation['recommendations']:
                self.recommendations_list.addItem(rec)
    
    def export_report(self):
        """Export analysis report"""
        if self.current_analysis is None:
            QMessageBox.warning(self, "Warning", "No analysis to export!")
            return
        
        try:
            # Ask for save location
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "Save Report", "", 
                "Text Files (*.txt);;PDF Files (*.pdf);;All Files (*.*)"
            )
            
            if file_path:
                # Create report content
                report = self._generate_report()
                
                # Save to file
                with open(file_path, 'w') as f:
                    f.write(report)
                
                QMessageBox.information(self, "Success", 
                                      f"Report saved to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export report:\n{str(e)}")
    
    def _generate_report(self):
        """Generate report content"""
        analysis = self.current_analysis
        rule_based = analysis['detection']['rule_based']
        
        report = f"""
        CARDIAC HEALTH ANALYSIS REPORT
        ================================
        
        Patient Information:
        -------------------
        Patient ID: {self.patient_id.text()}
        Age: {self.patient_age.value()}
        Gender: {self.patient_gender.currentText()}
        Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Analysis Summary:
        -----------------
        Risk Level: {rule_based['risk_level']}
        Diagnosis: {rule_based['diagnosis']}
        Confidence: {rule_based['confidence']}%
        Risk Score: {rule_based['risk_score']}
        
        Key Findings:
        -------------
        """
        
        # Add alerts
        if rule_based['alerts']:
            report += "\nAlerts:\n"
            for alert in rule_based['alerts']:
                report += f"  • {alert}\n"
        
        # Add HRV metrics
        report += "\nHRV Metrics:\n"
        report += "------------\n"
        
        categories = {
            'Time Domain': ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr'],
            'Frequency Domain': ['total_power', 'lf_power', 'hf_power', 'lf_hf_ratio'],
            'Nonlinear': ['sd1', 'sd2', 'sd1_sd2_ratio', 'sample_entropy']
        }
        
        for category, metrics in categories.items():
            report += f"\n{category}:\n"
            for metric in metrics:
                if metric in analysis['hrv_metrics']:
                    value = analysis['hrv_metrics'][metric]
                    units = self._get_metric_units(metric)
                    name = self._format_metric_name(metric)
                    report += f"  {name}: {value:.2f} {units}\n"
        
        # Add recommendations
        report += "\nRecommendations:\n"
        report += "----------------\n"
        for rec in analysis['interpretation']['recommendations']:
            report += f"• {rec}\n"
        
        # Add footer
        report += f"""
        
        --- End of Report ---
        
        Generated by Cardiac Health Monitoring System
        This report is for informational purposes only.
        Always consult with a healthcare professional for medical advice.
        """
        
        return report
    
    def export_data(self):
        """Export analysis data to CSV"""
        if self.current_analysis is None:
            QMessageBox.warning(self, "Warning", "No analysis data to export!")
            return
        
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self, "Save Data", "", 
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if file_path:
                # Create DataFrame with analysis results
                data = {
                    'parameter': [],
                    'value': [],
                    'units': [],
                    'category': []
                }
                
                # Add HRV metrics
                for metric, value in self.current_analysis['hrv_metrics'].items():
                    data['parameter'].append(self._format_metric_name(metric))
                    data['value'].append(value)
                    data['units'].append(self._get_metric_units(metric))
                    
                    # Determine category
                    if metric in ['mean_rr', 'std_rr', 'rmssd', 'pnn50', 'mean_hr']:
                        data['category'].append('Time Domain')
                    elif 'power' in metric or 'ratio' in metric or 'nu' in metric:
                        data['category'].append('Frequency Domain')
                    else:
                        data['category'].append('Nonlinear')
                
                # Add detection results
                rule_based = self.current_analysis['detection']['rule_based']
                data['parameter'].append('Risk Level')
                data['value'].append(rule_based['risk_level'])
                data['units'].append('')
                data['category'].append('Detection')
                
                data['parameter'].append('Risk Score')
                data['value'].append(rule_based['risk_score'])
                data['units'].append('')
                data['category'].append('Detection')
                
                data['parameter'].append('Confidence')
                data['value'].append(rule_based['confidence'])
                data['units'].append('%')
                data['category'].append('Detection')
                
                # Create and save DataFrame
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", 
                                      f"Data exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data:\n{str(e)}")

def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = HeartMonitorGUI()
    window.show()
    
    # Run application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()