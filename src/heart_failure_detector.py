# src/heart_failure_detector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from config import MODELS_DIR

class HeartFailureDetector:
    def __init__(self, use_ml=True):
        self.use_ml = use_ml
        self.scaler = StandardScaler()
        self.model = None
        self.rules = self._load_clinical_rules()
        
    def _load_clinical_rules(self):
        """Load clinical rules for heart failure detection"""
        rules = {
            'time_domain': {
                'sdnn': {'threshold': 50, 'direction': 'below', 'weight': 2.0},
                'rmssd': {'threshold': 20, 'direction': 'below', 'weight': 1.5},
                'pnn50': {'threshold': 5, 'direction': 'below', 'weight': 1.0}
            },
            'frequency_domain': {
                'lf_hf_ratio': {'threshold_low': 0.5, 'threshold_high': 3.0, 'weight': 1.5},
                'total_power': {'threshold': 1000, 'direction': 'below', 'weight': 1.0}
            },
            'nonlinear': {
                'sample_entropy': {'threshold': 1.0, 'direction': 'below', 'weight': 1.0},
                'sd1': {'threshold': 20, 'direction': 'below', 'weight': 1.0}
            }
        }
        return rules
    
    def rule_based_detection(self, hrv_metrics):
        """Rule-based heart failure detection using clinical guidelines"""
        risk_score = 0
        alerts = []
        contributing_factors = []
        
        # Time domain analysis
        for feature, rule in self.rules['time_domain'].items():
            if feature in hrv_metrics:
                value = hrv_metrics[feature]
                if rule['direction'] == 'below' and value < rule['threshold']:
                    risk_score += rule['weight']
                    alerts.append(f"Low {feature.upper()}: {value:.1f} (normal > {rule['threshold']})")
                    contributing_factors.append(feature)
                elif rule['direction'] == 'above' and value > rule['threshold']:
                    risk_score += rule['weight']
                    alerts.append(f"High {feature.upper()}: {value:.1f} (normal < {rule['threshold']})")
                    contributing_factors.append(feature)
        
        # Frequency domain analysis
        if 'lf_hf_ratio' in hrv_metrics:
            lf_hf = hrv_metrics['lf_hf_ratio']
            rule = self.rules['frequency_domain']['lf_hf_ratio']
            if lf_hf < rule['threshold_low'] or lf_hf > rule['threshold_high']:
                risk_score += rule['weight']
                alerts.append(f"Abnormal LF/HF ratio: {lf_hf:.2f} (normal: 0.5-3.0)")
                contributing_factors.append('lf_hf_ratio')
        
        # Nonlinear analysis
        for feature, rule in self.rules['nonlinear'].items():
            if feature in hrv_metrics:
                value = hrv_metrics[feature]
                if rule['direction'] == 'below' and value < rule['threshold']:
                    risk_score += rule['weight']
                    alerts.append(f"Low {feature}: {value:.2f} (normal > {rule['threshold']})")
                    contributing_factors.append(feature)
        
        # Determine risk level
        # Check if we have sufficient data for a valid assessment
        critical_metrics = ['sdnn', 'rmssd']
        missing_critical = [m for m in critical_metrics if m not in hrv_metrics]
        
        if len(missing_critical) > 0:
            risk_level = "INCONCLUSIVE"
            diagnosis = "Insufficient data for reliable analysis"
            action = "Longer recording required (>1 min)"
            risk_score = 0
        elif risk_score >= 4:
            risk_level = "HIGH RISK"
            diagnosis = "High probability of heart failure"
            action = "Immediate cardiology consultation required"
        elif risk_score >= 2:
            risk_level = "MODERATE RISK"
            diagnosis = "Possible cardiac dysfunction"
            action = "Further evaluation recommended"
        else:
            # Only declare low risk if we actually have data
            risk_level = "LOW RISK"
            diagnosis = "Normal cardiac function"
            action = "Routine monitoring sufficient"
        
        # Calculate confidence
        confidence = min(95, 50 + risk_score * 10)
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'diagnosis': diagnosis,
            'confidence': confidence,
            'alerts': alerts,
            'contributing_factors': list(set(contributing_factors)),
            'recommended_action': action,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def train_ml_model(self, features, labels, model_type='random_forest'):
        """Train machine learning model for heart failure detection"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        self.save_model(model_type)
        
        return accuracy
    
    def ml_based_detection(self, features):
        """Heart failure detection using trained ML model"""
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            print("No trained model available. Using rule-based detection.")
            return None
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                [f'feature_{i}' for i in range(len(features))],
                self.model.feature_importances_
            ))
        
        return {
            'prediction': 'Heart Failure' if prediction == 1 else 'Normal',
            'probability': float(max(probability)),
            'normal_prob': float(probability[0]),
            'hf_prob': float(probability[1]),
            'feature_importance': feature_importance,
            'model_used': type(self.model).__name__
        }
    
    def save_model(self, model_name='heart_failure_model'):
        """Save trained model and scaler"""
        if self.model is not None:
            model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
            scaler_path = os.path.join(MODELS_DIR, f'{model_name}_scaler.joblib')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
    
    def load_model(self, model_name='heart_failure_model'):
        """Load trained model and scaler"""
        model_path = os.path.join(MODELS_DIR, f'{model_name}.joblib')
        scaler_path = os.path.join(MODELS_DIR, f'{model_name}_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print("No saved model found")
            return False
    
    def create_synthetic_dataset(self, n_samples=1000):
        """Create synthetic dataset for training"""
        np.random.seed(42)
        
        features = []
        labels = []
        
        for i in range(n_samples):
            # Normal heart pattern
            if i < n_samples // 2:
                sdnn = np.random.uniform(50, 150)
                rmssd = np.random.uniform(25, 100)
                lf_hf = np.random.uniform(0.8, 2.5)
                sampen = np.random.uniform(1.2, 2.0)
                label = 0  # Normal
            # Heart failure pattern
            else:
                sdnn = np.random.uniform(10, 45)
                rmssd = np.random.uniform(5, 20)
                lf_hf = np.random.uniform(0.2, 0.7) if np.random.rand() > 0.5 else np.random.uniform(3.5, 6.0)
                sampen = np.random.uniform(0.3, 1.0)
                label = 1  # Heart failure
            
            # Add noise and additional features
            features.append([
                sdnn + np.random.randn() * 5,
                rmssd + np.random.randn() * 3,
                lf_hf + np.random.randn() * 0.3,
                sampen + np.random.randn() * 0.1,
                np.random.uniform(60, 100),  # Mean heart rate
                np.random.uniform(500, 1000),  # Mean RR interval
                np.random.uniform(0, 30),  # pNN50
                np.random.uniform(500, 3000),  # Total power
            ])
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def detect(self, hrv_metrics, clinical_diagnosis=None):
        """Main detection method combining rule-based, ML, and clinical history"""
        # Rule-based detection
        rule_result = self.rule_based_detection(hrv_metrics)
        
        # Clinical detection (Ground Truth)
        clinical_risk = self._eval_clinical_risk(clinical_diagnosis)
        
        # ML-based detection if available
        ml_result = None
        if self.use_ml:
            # Extract features for ML model
            feature_keys = ['sdnn', 'rmssd', 'lf_hf_ratio', 'sample_entropy', 
                          'mean_hr', 'mean_rr', 'pnn50', 'total_power']
            features = [hrv_metrics.get(key, 0) for key in feature_keys]
            
            ml_result = self.ml_based_detection(features)
        
        # Combine results
        # If clinical diagnosis indicates failure, it overrides low-sensitivity HRV
        final_risk_level = rule_result['risk_level']
        consensus = self._get_consensus(rule_result, ml_result)
        
        if clinical_risk:
            # Check if clinical diagnosis is severe OR if rule-based failed
            is_inconclusive = "INCONCLUSIVE" in rule_result['risk_level'] or "Insufficient" in rule_result['risk_level']
            
            if clinical_risk['level'] == 'HIGH':
                final_risk_level = "HIGH RISK (Clinical)"
                consensus = f"Clinical Diagnosis: {clinical_diagnosis}"
                rule_result['diagnosis'] = clinical_risk['desc']
                rule_result['recommended_action'] = clinical_risk['action']
                rule_result['risk_level'] = final_risk_level # Sync for GUI
                rule_result['alerts'].insert(0, f"Original Diagnosis: {clinical_diagnosis}")
                
            elif clinical_risk['level'] == 'MODERATE':
                if "HIGH" not in final_risk_level:
                    final_risk_level = "MODERATE RISK (Clinical)"
                    consensus = f"Clinical Diagnosis: {clinical_diagnosis}"
                    if is_inconclusive:
                         rule_result['diagnosis'] = clinical_risk['desc']
                         rule_result['recommended_action'] = clinical_risk['action']
                         rule_result['risk_level'] = final_risk_level
            
            elif is_inconclusive:
                # If signal analysis failed but we have a clinical label (e.g. LOW), use it!
                final_risk_level = f"{clinical_risk['level']} RISK (Clinical)"
                consensus = f"Clinical Diagnosis: {clinical_diagnosis}"
                rule_result['diagnosis'] = clinical_risk['desc']
                rule_result['recommended_action'] = clinical_risk['action']
                rule_result['risk_level'] = final_risk_level
        
        result = {
            'rule_based': rule_result,
            'ml_based': ml_result,
            'final_risk_level': final_risk_level,
            'consensus': consensus
        }
        
        return result

    def _eval_clinical_risk(self, diagnosis):
        """Evaluate risk based on clinical diagnosis string"""
        if not diagnosis or diagnosis == "Unknown":
            return None
            
        diag_lower = diagnosis.lower()
        
        # High Risk Keywords
        if any(x in diag_lower for x in ['mi', 'myocardial infarction', 'failure', 'ischemia', 'dysrhythmia', 'block']):
            return {
                'level': 'HIGH',
                'desc': 'Clinical history indicates significant cardiac condition',
                'action': 'Adhere to existing treatment plan'
            }
        # Moderate Risk Keywords
        elif any(x in diag_lower for x in ['hypertrophy', 'atrial fib', 'flutter', 'major', 'st/t']):
            return {
                'level': 'MODERATE',
                'desc': 'Clinical history indicates abnormality',
                'action': 'Monitor for changes'
            }
        
        return {'level': 'LOW', 'desc': 'Normal clinical findings', 'action': 'Routine'}
    
    def _get_consensus(self, rule_result, ml_result):
        """Get consensus between rule-based and ML results"""
        if ml_result is None:
            return rule_result
        
        # Check if both methods agree
        rule_risk = rule_result['risk_level']
        ml_pred = ml_result['prediction'] if ml_result else None
        
        if ml_pred == 'Heart Failure' and 'HIGH' in rule_risk:
            consensus = "High confidence: Both methods indicate heart failure risk"
        elif ml_pred == 'Normal' and 'LOW' in rule_risk:
            consensus = "High confidence: Both methods indicate normal function"
        elif (ml_pred == 'Heart Failure' and 'LOW' in rule_risk) or \
             (ml_pred == 'Normal' and 'HIGH' in rule_risk):
            consensus = "Conflict: Methods disagree. Further evaluation needed."
        else:
            consensus = "Moderate confidence: Mixed signals detected"
        
        return consensus

# Test the detector
if __name__ == "__main__":
    # Create detector
    detector = HeartFailureDetector(use_ml=True)
    
    # Create and train model on synthetic data
    print("Creating synthetic dataset...")
    X, y = detector.create_synthetic_dataset(500)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Normal={sum(y==0)}, HF={sum(y==1)}")
    
    # Train model
    print("\nTraining ML model...")
    accuracy = detector.train_ml_model(X, y, model_type='random_forest')
    
    # Test detection on sample metrics
    print("\n=== TEST DETECTION ===")
    
    # Sample healthy metrics
    healthy_metrics = {
        'sdnn': 85.3,
        'rmssd': 45.2,
        'lf_hf_ratio': 1.8,
        'sample_entropy': 1.6,
        'mean_hr': 72.5,
        'mean_rr': 830.2,
        'pnn50': 12.3,
        'total_power': 2100.5
    }
    
    # Sample heart failure metrics
    hf_metrics = {
        'sdnn': 28.7,
        'rmssd': 15.3,
        'lf_hf_ratio': 0.3,
        'sample_entropy': 0.7,
        'mean_hr': 88.2,
        'mean_rr': 680.5,
        'pnn50': 2.1,
        'total_power': 450.8
    }
    
    print("\n1. Healthy Pattern Detection:")
    healthy_result = detector.detect(healthy_metrics)
    print(f"Risk Level: {healthy_result['rule_based']['risk_level']}")
    print(f"Diagnosis: {healthy_result['rule_based']['diagnosis']}")
    
    print("\n2. Heart Failure Pattern Detection:")
    hf_result = detector.detect(hf_metrics)
    print(f"Risk Level: {hf_result['rule_based']['risk_level']}")
    print(f"Diagnosis: {hf_result['rule_based']['diagnosis']}")
    print(f"Confidence: {hf_result['rule_based']['confidence']}%")
    print("\nAlerts:")
    for alert in hf_result['rule_based']['alerts']:
        print(f"  - {alert}")