"""
Advanced machine learning models for behavioral analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import joblib
import os

class AdvancedAnomalyDetector:
    """
    Rule-based anomaly detection based on feature deviation from baseline.
    This version does not use TensorFlow or complex scikit-learn models.
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination # Not directly used in this rule-based model, but kept for compatibility
        self.scaler = StandardScaler()
        self.baseline_mean = None
        self.baseline_std = None
        self.feature_names = []

    def fit(self, X: np.ndarray, feature_names: List[str]):
        """Calculate baseline mean and standard deviation from training data."""
        self.feature_names = feature_names
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.baseline_mean = np.mean(X_scaled, axis=0)
        self.baseline_std = np.std(X_scaled, axis=0)
        
        # Avoid division by zero for std dev
        self.baseline_std[self.baseline_std == 0] = 1e-8 
        
        return self

    def predict(self, X: np.ndarray) -> Tuple[List[bool], List[float]]:
        """Predict anomalies based on deviation from baseline."""
        X_scaled = self.scaler.transform(X)
        
        # Calculate Z-scores for each feature
        z_scores = np.abs((X_scaled - self.baseline_mean) / self.baseline_std)
        
        # Combine Z-scores into a single risk score (e.g., max Z-score)
        risk_scores = np.max(z_scores, axis=1)
        
        # Normalize risk_scores to confidence (0-1)
        # Higher risk_score means higher confidence in anomaly
        # Using a simple sigmoid-like function or capping
        confidence = 1 / (1 + np.exp(-risk_scores + 3)) # Increased 2 to 3 for less aggressive confidence
        
        # Determine anomaly based on a threshold (e.g., max Z-score > 2.0)
        is_anomaly = risk_scores > 3.0 # Increased 2.0 to 3.0 for less sensitive anomaly detection
        
        return is_anomaly.tolist(), confidence.tolist()

class UserBehaviorClassifier:
    """Multi-user classifier for user identification"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        self.scaler = StandardScaler()
        self.users = []
        self.feature_names = []
        
    def fit(self, X: np.ndarray, y: List[str], feature_names: List[str]):
        """Train classifier to distinguish between users"""
        self.feature_names = feature_names
        self.users = list(set(y))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train classifier
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[List[str], List[float]]:
        """Predict user identity and confidence"""
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Get confidence as max probability
        confidences = np.max(probabilities, axis=1)
        
        return predictions.tolist(), confidences.tolist()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance for user classification"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return {}

class BehaviorEvolutionTracker:
    """Track how user behavior evolves over time"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.user_histories = {}
        
    def update_user_behavior(self, user_id: str, features: Dict):
        """Update user's behavioral history"""
        if user_id not in self.user_histories:
            self.user_histories[user_id] = []
        
        # Add new features with timestamp
        self.user_histories[user_id].append({
            'timestamp': pd.Timestamp.now(),
            'features': features
        })
        
        # Keep only recent history
        if len(self.user_histories[user_id]) > self.window_size:
            self.user_histories[user_id] = self.user_histories[user_id][-self.window_size:]
    
    def detect_behavior_drift(self, user_id: str, current_features: Dict) -> Tuple[bool, float]:
        """Detect if user behavior has significantly changed"""
        if user_id not in self.user_histories or len(self.user_histories[user_id]) < 3:
            return False, 0.0
        
        history = self.user_histories[user_id]
        
        # Extract historical features
        feature_names = ['wpm', 'accuracy', 'avg_flight_time', 'std_flight_time']
        historical_features = []
        
        for entry in history:
            feature_vector = [entry['features'].get(name, 0) for name in feature_names]
            historical_features.append(feature_vector)
        
        current_vector = [current_features.get(name, 0) for name in feature_names]
        
        # Calculate drift using statistical distance
        historical_mean = np.mean(historical_features, axis=0)
        historical_std = np.std(historical_features, axis=0)
        
        # Z-score based drift detection
        z_scores = np.abs((current_vector - historical_mean) / (historical_std + 1e-8))
        max_z_score = np.max(z_scores)
        
        # Drift detected if any feature has z-score > 2
        drift_detected = max_z_score > 2.0
        drift_magnitude = max_z_score / 2.0  # Normalize to 0-1 range
        
        return drift_detected, min(drift_magnitude, 1.0)

class AdaptiveLearningAgent:
    """Agent that continuously adapts to user behavior"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.user_models = {}
        self.update_counters = {}
        
    def adaptive_update(self, user_id: str, features: Dict, is_authentic: bool):
        """Adaptively update user model based on feedback"""
        if user_id not in self.user_models:
            return
        
        if user_id not in self.update_counters:
            self.update_counters[user_id] = 0
        
        # Only update if we're confident this is authentic behavior
        if is_authentic:
            self._update_baseline(user_id, features)
            self.update_counters[user_id] += 1
    
    def _update_baseline(self, user_id: str, features: Dict):
        """Update baseline statistics with new authentic sample"""
        if user_id not in self.user_models:
            return
        
        model_data = self.user_models[user_id]
        baseline_stats = model_data.get('baseline_stats', {})
        
        feature_names = model_data.get('feature_names', [])
        new_vector = [features.get(name, 0) for name in feature_names]
        
        # Exponential moving average update
        if 'mean' in baseline_stats:
            current_mean = baseline_stats['mean']
            updated_mean = (1 - self.adaptation_rate) * current_mean + self.adaptation_rate * np.array(new_vector)
            baseline_stats['mean'] = updated_mean
        
        model_data['baseline_stats'] = baseline_stats
        self.user_models[user_id] = model_data

class ModelPersistence:
    """Utilities for saving and loading models"""
    
    @staticmethod
    def save_model(model, filepath: str):
        """Save model to disk"""
        try:
            joblib.dump(model, filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    @staticmethod
    def load_model(filepath: str):
        """Load model from disk"""
        try:
            if os.path.exists(filepath):
                return joblib.load(filepath)
            return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def save_user_models(models_dict: Dict, directory: str):
        """Save all user models to directory"""
        os.makedirs(directory, exist_ok=True)
        
        for user_id, model_data in models_dict.items():
            filepath = os.path.join(directory, f"{user_id}_model.pkl")
            ModelPersistence.save_model(model_data, filepath)
    
    @staticmethod
    def load_user_models(directory: str) -> Dict:
        """Load all user models from directory"""
        models = {}
        
        if not os.path.exists(directory):
            return models
        
        for filename in os.listdir(directory):
            if filename.endswith('_model.pkl'):
                user_id = filename.replace('_model.pkl', '')
                filepath = os.path.join(directory, filename)
                model_data = ModelPersistence.load_model(filepath)
                
                if model_data:
                    models[user_id] = model_data
        
        return models

class RealTimeAnalyzer:
    """Real-time analysis of typing patterns"""
    
    def __init__(self, buffer_size: int = 50):
        self.buffer_size = buffer_size
        self.keystroke_buffer = []
        self.analysis_results = []
        
    def add_keystroke(self, char: str, timestamp: float, flight_time: float):
        """Add keystroke to real-time buffer"""
        self.keystroke_buffer.append({
            'char': char,
            'timestamp': timestamp,
            'flight_time': flight_time
        })
        
        # Keep buffer size manageable
        if len(self.keystroke_buffer) > self.buffer_size:
            self.keystroke_buffer.pop(0)
    
    def get_realtime_features(self) -> Dict:
        """Extract features from current buffer"""
        if len(self.keystroke_buffer) < 5:
            return {}
        
        # Recent flight times
        recent_flights = [ks['flight_time'] for ks in self.keystroke_buffer[-10:]]
        
        # Typing rhythm analysis
        rhythm_consistency = 1.0 / (1.0 + np.std(recent_flights))
        
        # Speed estimation
        total_time = self.keystroke_buffer[-1]['timestamp'] - self.keystroke_buffer[-10]['timestamp']
        estimated_wpm = (10 / 5) / (total_time / 60) if total_time > 0 else 0  # Rough estimation
        
        return {
            'realtime_wpm': estimated_wpm,
            'rhythm_consistency': rhythm_consistency,
            'avg_recent_flight': np.mean(recent_flights),
            'flight_variance': np.var(recent_flights),
            'buffer_size': len(self.keystroke_buffer)
        }
    
    def detect_realtime_anomalies(self, baseline_features: Dict) -> Tuple[bool, str]:
        """Detect anomalies in real-time"""
        current_features = self.get_realtime_features()
        
        if not current_features or not baseline_features:
            return False, "Insufficient data"
        
        # Simple threshold-based detection
        alerts = []
        
        # Speed check
        baseline_wpm = baseline_features.get('wpm', 50)
        current_wpm = current_features.get('realtime_wpm', 0)
        
        if abs(current_wpm - baseline_wpm) > baseline_wpm * 0.5:
            alerts.append("Speed anomaly")
        
        # Rhythm check
        baseline_rhythm = baseline_features.get('typing_rhythm_variance', 0.1)
        current_variance = current_features.get('flight_variance', 0)
        
        if current_variance > baseline_rhythm * 2:
            alerts.append("Rhythm anomaly")
        
        has_anomaly = len(alerts) > 0
        alert_message = "; ".join(alerts) if alerts else "Normal"
        
        return has_anomaly, alert_message
