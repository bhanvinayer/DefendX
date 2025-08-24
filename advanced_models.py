"""
Advanced machine learning models for behavioral analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import joblib
import os

class AdvancedAnomalyDetector:
    """Advanced anomaly detection with multiple algorithms"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.ensemble_weights = {}
        
    def fit(self, X: np.ndarray, feature_names: List[str]):
        """Train ensemble of anomaly detection models"""
        self.feature_names = feature_names
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        
        # Train multiple models
        self.models['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.models['isolation_forest'].fit(X_scaled)
        
        self.models['one_class_svm'] = OneClassSVM(
            nu=self.contamination,
            kernel='rbf',
            gamma='scale'
        )
        self.models['one_class_svm'].fit(X_scaled)
        
        # DBSCAN for density-based clustering
        self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = self.models['dbscan'].fit_predict(X_scaled)
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights(X_scaled)
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[List[bool], List[float]]:
        """Predict anomalies using ensemble approach"""
        X_scaled = self.scalers['standard'].transform(X)
        
        predictions = {}
        scores = {}
        
        # Isolation Forest
        if_pred = self.models['isolation_forest'].predict(X_scaled)
        if_score = self.models['isolation_forest'].decision_function(X_scaled)
        predictions['isolation_forest'] = if_pred == -1
        scores['isolation_forest'] = np.abs(if_score)
        
        # One-Class SVM
        svm_pred = self.models['one_class_svm'].predict(X_scaled)
        svm_score = self.models['one_class_svm'].decision_function(X_scaled)
        predictions['one_class_svm'] = svm_pred == -1
        scores['one_class_svm'] = np.abs(svm_score)
        
        # DBSCAN (outliers have label -1)
        dbscan_pred = self.models['dbscan'].fit_predict(X_scaled)
        predictions['dbscan'] = dbscan_pred == -1
        scores['dbscan'] = np.ones(len(X_scaled))  # Simplified score
        
        # Ensemble prediction
        ensemble_pred = []
        ensemble_scores = []
        
        for i in range(len(X_scaled)):
            weighted_score = 0
            anomaly_votes = 0
            
            for model_name in predictions:
                weight = self.ensemble_weights.get(model_name, 1.0)
                if predictions[model_name][i]:
                    anomaly_votes += weight
                weighted_score += scores[model_name][i] * weight
            
            ensemble_pred.append(anomaly_votes > sum(self.ensemble_weights.values()) / 2)
            ensemble_scores.append(weighted_score / sum(self.ensemble_weights.values()))
        
        return ensemble_pred, ensemble_scores
    
    def _calculate_ensemble_weights(self, X: np.ndarray):
        """Calculate weights for ensemble based on model performance"""
        # Simple equal weighting for now
        self.ensemble_weights = {
            'isolation_forest': 0.4,
            'one_class_svm': 0.4,
            'dbscan': 0.2
        }

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
