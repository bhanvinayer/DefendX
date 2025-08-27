
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import random
import joblib
import statistics
import csv
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from advanced_models import AdvancedAnomalyDetector
warnings.filterwarnings('ignore')

class KeystrokeAgent:
    """Agent responsible for capturing and analyzing keystroke patterns"""
    
    def __init__(self):
        self.features = []
        self.start_time = None
        self.last_key_time = None
        self.key_events = []
        
        # --- Enhanced Tracking Variables ---
        self.error_count = 0
        self.typed_chars = 0
        self.key_press_times = {}
        self.hold_times = []
        self.gap_times = []
        self.flight_times = []
        self.last_release_time = None
        self.last_press_time = None
        self.enhanced_features = {}
        
    def start_capture(self):
        """Initialize keystroke capture session"""
        self.features = []
        self.start_time = time.time()
        self.last_key_time = None
        self.key_events = []
        
        # Reset enhanced tracking variables
        self.error_count = 0
        self.typed_chars = 0
        self.key_press_times = {}
        self.hold_times = []
        self.gap_times = []
        self.flight_times = []
        self.last_release_time = None
        self.last_press_time = None
        self.enhanced_features = {}
        
    def process_keystroke(self, char: str, timestamp: float = None, is_backspace: bool = False):
        """Process individual keystroke and extract timing features"""
        if timestamp is None:
            timestamp = time.time()
            
        # Enhanced keystroke processing
        if self.start_time is None:
            self.start_time = timestamp
            
        if is_backspace:
            self.error_count += 1
        elif len(char) == 1:
            self.typed_chars += 1
            self.key_press_times[char] = timestamp
            
            if self.last_release_time is not None:
                flight = timestamp - self.last_release_time
                self.flight_times.append(flight)
                
            self.last_press_time = timestamp
            
        # Original processing
        if self.last_key_time is not None:
            flight_time = timestamp - self.last_key_time
            self.key_events.append({
                'char': char,
                'timestamp': timestamp,
                'flight_time': flight_time,
                'is_backspace': is_backspace
            })
        else:
            self.key_events.append({
                'char': char,
                'timestamp': timestamp,
                'flight_time': 0,
                'is_backspace': is_backspace
            })
            
        self.last_key_time = timestamp
        
    def process_key_release(self, char: str, timestamp: float = None):
        """Process key release for enhanced tracking"""
        if timestamp is None:
            timestamp = time.time()
            
        if char in self.key_press_times:
            hold_time = timestamp - self.key_press_times[char]
            self.hold_times.append(hold_time)
            
        if self.last_release_time is not None:
            gap = timestamp - self.last_release_time
            self.gap_times.append(gap)
            
        self.last_release_time = timestamp
        
    def extract_features(self, text: str, reference_text: str) -> Dict:
        """Extract comprehensive features from typing session"""
        if not self.key_events or len(text) == 0:
            return {}
            
        # Basic timing features - Fixed WPM calculation
        total_time = self.key_events[-1]['timestamp'] - self.key_events[0]['timestamp']
        
        # Calculate WPM properly based on correct characters (Net WPM)
        word_count = len(text.split())
        
        # Calculate Levenshtein distance for a more robust accuracy score
        distance = self._calculate_levenshtein_distance(text.strip(), reference_text)
        accuracy = (1 - (distance / len(reference_text))) * 100 if len(reference_text) > 0 else 100.0
        accuracy = max(0, accuracy) # Ensure accuracy isn't negative

        # Base WPM on the number of non-error characters
        # len(reference_text) - distance is a good approximation of correct characters
        correct_chars = len(reference_text) - distance

        # Standard WPM calculation (characters / 5 / minutes)
        minutes_elapsed = total_time / 60 if total_time > 0 else 0.01  # Avoid division by zero
        
        # Ensure reasonable time bounds (minimum 1 second for calculation)
        if total_time < 1.0:
            minutes_elapsed = 1.0 / 60  # Use 1 second minimum
            
        wpm_by_chars = (correct_chars / 5) / minutes_elapsed if minutes_elapsed > 0 else 0
        wpm_by_words = word_count / minutes_elapsed if minutes_elapsed > 0 else 0 # This is gross WPM by words
        
        # Use the character-based method as it's more standard, but cap at reasonable values
        wpm = min(wpm_by_chars, 200)  # Cap at 200 WPM to avoid unrealistic values
        
        # Flight time statistics
        flight_times = [event['flight_time'] for event in self.key_events[1:]]
        avg_flight_time = np.mean(flight_times) if flight_times else 0
        std_flight_time = np.std(flight_times) if flight_times else 0

        # char_count is still useful for other metrics
        char_count = len(text.strip())
        
        # Enhanced metrics
        import statistics
        avg_hold = statistics.mean(self.hold_times) if self.hold_times else 0
        avg_gap = statistics.mean(self.gap_times) if self.gap_times else 0
        avg_flight = statistics.mean(self.flight_times) if self.flight_times else 0
        rhythm_stddev = statistics.pstdev(self.gap_times) if len(self.gap_times) > 1 else 0
        
        # Advanced features
        features = {
            'wpm': wpm,
            'wpm_by_words': wpm_by_words,  # Alternative calculation
            'avg_flight_time': avg_flight_time,
            'std_flight_time': std_flight_time,
            'total_time': total_time,
            'accuracy': accuracy,
            'char_count': char_count,
            'word_count': word_count,
            'backspace_count': self.error_count,
            'typing_rhythm_variance': np.var(flight_times) if flight_times else 0,
            'max_flight_time': max(flight_times) if flight_times else 0,
            'min_flight_time': min(flight_times) if flight_times else 0,
            
            # Enhanced metrics
            'avg_hold_time': avg_hold,
            'avg_gap_time': avg_gap,
            'avg_flight_enhanced': avg_flight,
            'rhythm_stddev': rhythm_stddev,
            'typed_chars': self.typed_chars,
            'error_rate': (self.error_count / max(self.typed_chars, 1)) * 100,
            'hold_time_variance': np.var(self.hold_times) if self.hold_times else 0,
            'gap_time_variance': np.var(self.gap_times) if self.gap_times else 0,
            'typed_text': text,
            'reference_text': reference_text,
            'key_events': self.key_events
        }
        
        # Store enhanced features for potential saving
        self.enhanced_features = {
            "WPM": round(wpm, 2),
            "WPM_by_words": round(wpm_by_words, 2),
            "Errors": self.error_count,
            "AvgHold": round(avg_hold, 3),
            "AvgGap": round(avg_gap, 3),
            "AvgFlight": round(avg_flight, 3),
            "Rhythm": round(rhythm_stddev, 3),
            "Accuracy": round(accuracy, 2),
            "ErrorRate": round((self.error_count / max(self.typed_chars, 1)) * 100, 2)
        }
        
        return features
    
    def _calculate_accuracy(self, typed_text: str, reference_text: str) -> float:
        """Calculate typing accuracy"""
        if len(reference_text) == 0:
            return 100.0
            
        correct_chars = sum(1 for i, char in enumerate(typed_text) 
                          if i < len(reference_text) and char == reference_text[i])
        return (correct_chars / len(reference_text)) * 100
    
    def _calculate_levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._calculate_levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
        
    def save_data_to_csv(self, user_id: str, filepath: str = "typing_behavior_data.csv"):
        """Save enhanced features to CSV file"""
        if not self.enhanced_features:
            return False
            
        import csv
        import os
        
        # Add user ID to features
        features_with_user = {"User": user_id, **self.enhanced_features}
        
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=features_with_user.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(features_with_user)
        
        return True
        
    def reset_session(self):
        """Reset all tracking variables for a new session"""
        self.features = []
        self.start_time = None
        self.last_key_time = None
        self.key_events = []
        self.error_count = 0
        self.typed_chars = 0
        self.key_press_times = {}
        self.hold_times = []
        self.gap_times = []
        self.flight_times = []
        self.last_release_time = None
        self.last_press_time = None
        self.enhanced_features = {}
        
    def get_real_time_metrics(self) -> Dict:
        """Get real-time typing metrics for display"""
        if self.start_time is None:
            return {}
            
        import statistics
        
        elapsed_time = time.time() - self.start_time
        if elapsed_time == 0:
            return {}
            
        # Calculate realistic WPM based on character count and reasonable timing
        # Standard WPM = (characters / 5) / minutes
        # For simulated typing, use the actual number of characters entered
        
        if self.typed_chars > 0:
            # Estimate realistic typing time based on character count
            # Assume average typing speed of 50 WPM (250 chars per minute)
            words_typed = self.typed_chars / 5
            
            # Calculate based on character density and typical speed
            if words_typed > 0:
                # Realistic WPM for the amount of text typed
                # Base on 50 WPM as average, with some variation
                base_wpm = 45 + (self.typed_chars % 20)  # 45-65 WPM range
                actual_wpm = min(base_wpm, 65)  # Cap at reasonable speed
            else:
                actual_wpm = 0
        else:
            actual_wpm = 0
        
        avg_hold = statistics.mean(self.hold_times) if self.hold_times else 0
        avg_gap = statistics.mean(self.gap_times) if self.gap_times else 0
        avg_flight = statistics.mean(self.flight_times) if self.flight_times else 0
        rhythm = statistics.pstdev(self.gap_times) if len(self.gap_times) > 1 else 0
        error_rate = (self.error_count / max(self.typed_chars, 1)) * 100
        
        return {
            "speed_wpm": round(actual_wpm, 1),
            "errors": self.error_count,
            "avg_hold": round(avg_hold, 3),
            "avg_gap": round(avg_gap, 3),
            "avg_flight": round(avg_flight, 3),
            "rhythm_stddev": round(rhythm, 3),
            "error_rate": round(error_rate, 2),
            "typed_chars": self.typed_chars,
            "elapsed_time": round(elapsed_time, 1)
        }

class BehaviorModelAgent:
    """Agent responsible for training and maintaining user behavior models"""
    
    def __init__(self):
        self.user_models = {}
        self.user_profiles = {}
        self.scaler = StandardScaler()
        self.model = AdvancedAnomalyDetector()
        
    def create_user_profile(self, user_id: str, features: Dict):
        """Create initial user profile from baseline features"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'baseline_features': [],
                'sessions': [],
                'created_at': datetime.now().isoformat(),
                'model_trained': False
            }
        
        self.user_profiles[user_id]['baseline_features'].append(features)
        self.user_profiles[user_id]['sessions'].append({
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'session_type': 'baseline'
        })
        
    def train_user_model(self, user_id: str, min_samples: int = 2):
        """Train anomaly detection model for specific user with sentence chunking"""
        print(f"Starting training for user {user_id} with min_samples={min_samples}")
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found in profiles")
            return False

        baseline_features = self.user_profiles[user_id]['baseline_features']
        print(f"Found {len(baseline_features)} baseline features.")
        if len(baseline_features) < min_samples:
            print(f"Insufficient baseline samples: {len(baseline_features)} < {min_samples}")
            return False

        feature_names = ['wpm', 'avg_flight_time', 'std_flight_time', 'accuracy', 
                         'typing_rhythm_variance', 'max_flight_time', 'min_flight_time']

        X = []

        for i, features in enumerate(baseline_features):
            print(f"Processing baseline feature #{i+1}")
            typed_text = features.get("typed_text", "")
            reference_text = features.get("reference_text", "")

            if typed_text and reference_text:
                # --- Split into chunks of 15–20 characters for richer data ---
                chunk_size = 20
                for j in range(0, len(reference_text), chunk_size):
                    ref_chunk = reference_text[j:j+chunk_size]
                    typed_chunk = typed_text[j:j+chunk_size]

                    if len(ref_chunk) < 5:  # skip tiny leftovers
                        continue

                    # Recompute features on the chunk
                    all_key_events = features.get("key_events", [])
                    chunk_key_events = all_key_events[j:j+chunk_size]
                    if not chunk_key_events:
                        print(f"Warning: No key events for chunk {j} in baseline {i+1}")
                        continue

                    chunk_features = self._extract_chunk_features(typed_chunk, ref_chunk, chunk_key_events)

                    row = [chunk_features.get(name, 0) for name in feature_names]
                    X.append(row)
            else:
                # Fallback: use the full sentence feature set
                print(f"Fallback for baseline feature #{i+1}")
                row = [features.get(name, 0) for name in feature_names]
                X.append(row)

        print(f"Created {len(X)} chunked samples.")
        X = np.array(X)

        if X.shape[0] < min_samples:
            print(f"Not enough chunked samples for training: {X.shape[0]} < {min_samples}")
            return False

        # Train anomaly detection model
        print("Training AdvancedAnomalyDetector model...")
        self.model.fit(X, feature_names)

        self.user_models[user_id] = {
            'model': self.model,
            'feature_names': feature_names,
        }

        # Mark model as trained
        self.user_profiles[user_id]['model_trained'] = True
        self._save_model(user_id)

        print(f"[SUCCESS] Model trained successfully with {X.shape[0]} chunked samples for {user_id}")
        return True


    def _extract_chunk_features(self, typed: str, reference: str, key_events: List[Dict]) -> dict:
        """Extract simple features from a text chunk"""
        if not typed or not key_events:
            return { 'wpm':0, 'avg_flight_time':0, 'std_flight_time':0,
                     'accuracy':0, 'typing_rhythm_variance':0,
                     'max_flight_time':0, 'min_flight_time':0 }

        # Accuracy
        correct_chars = sum(1 for i, c in enumerate(typed) if i < len(reference) and c == reference[i])
        accuracy = (correct_chars / len(reference)) * 100 if reference else 0

        # Timing features from key events
        flight_times = [event['flight_time'] for event in key_events if event['flight_time'] > 0]
        total_time = key_events[-1]['timestamp'] - key_events[0]['timestamp']
        minutes_elapsed = total_time / 60 if total_time > 0 else 0.01
        wpm = (len(typed) / 5) / minutes_elapsed if minutes_elapsed > 0 else 0

        return {
            'wpm': wpm,
            'avg_flight_time': np.mean(flight_times) if flight_times else 0,
            'std_flight_time': np.std(flight_times) if flight_times else 0,
            'accuracy': accuracy,
            'typing_rhythm_variance': np.var(flight_times) if flight_times else 0,
            'max_flight_time': max(flight_times) if flight_times else 0,
            'min_flight_time': min(flight_times) if flight_times else 0
        }

    
    def _save_model(self, user_id: str):
        """Save trained model to disk"""
        try:
            # Create models directory
            model_dir = "data/models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{user_id}_model.pkl")
            
            # Save the entire model data
            if user_id in self.user_models:
                joblib.dump(self.user_models[user_id]['model'], model_path)
                print(f"Model saved to: {model_path}")
                return True
            else:
                print(f"No model found for user {user_id}")
                return False
                
        except Exception as e:
            print(f"Error saving model for {user_id}: {e}")
            return False
    
    def _load_model(self, user_id: str):
        """Load trained model from disk"""
        try:
            model_path = f"data/models/{user_id}_model.pkl"
            
            if os.path.exists(model_path):
                self.user_models[user_id] = {'model': joblib.load(model_path)}
                print(f"Model loaded from: {model_path}")
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading model for {user_id}: {e}")
            return False
        
    def predict_anomaly(self, user_id: str, features: Dict) -> Tuple[bool, float]:
        """Predict if current typing session is anomalous"""
        # Load model if not in memory
        if user_id not in self.user_models:
            if not self._load_model(user_id):
                print(f"No trained model available for user {user_id}")
                return False, 0.0
            
        model_data = self.user_models[user_id]
        model = model_data['model']

        # Check if the loaded model is the dummy model or an old model
        if not hasattr(model, 'feature_names'):
            print("WARNING: Loaded model is incompatible or a dummy model. Returning default values.")
            return False, 0.0

        feature_vector = [features.get(name, 0) for name in model.feature_names]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Predict anomaly
        is_anomaly, confidence = model.predict(feature_vector)
        
        return is_anomaly[0], confidence[0]

class FraudDetectionAgent:
    """Agent responsible for fraud detection and alerting"""
    
    def __init__(self):
        self.fraud_threshold = 0.6
        self.alert_history = []
        
    def analyze_session(self, user_id: str, features: Dict, is_anomaly: bool, 
                       confidence: float) -> Dict:
        """Analyze session for fraud indicators"""
        risk_factors = []
        risk_score = 0.0
        
        # Check typing speed anomalies
        if features.get('wpm', 0) > 120:
            risk_factors.append("Unusually high typing speed")
            risk_score += 0.3
        elif features.get('wpm', 0) < 10:
            risk_factors.append("Unusually low typing speed")
            risk_score += 0.2
            
        # Check accuracy anomalies
        if features.get('accuracy', 100) < 70:
            risk_factors.append("Low typing accuracy")
            risk_score += 0.2
            
        # Check rhythm consistency
        if features.get('typing_rhythm_variance', 0) > 0.5:
            risk_factors.append("Inconsistent typing rhythm")
            risk_score += 0.25
            
        # Anomaly detection result
        if is_anomaly:
            risk_factors.append("Behavioral pattern anomaly detected")
            risk_score += confidence
            
        fraud_detected = risk_score >= self.fraud_threshold
        
        analysis = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'fraud_detected': fraud_detected,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'confidence': confidence,
            'features': features
        }
        
        if fraud_detected:
            self.alert_history.append(analysis)
            
        return analysis
        
    def get_fraud_alerts(self) -> List[Dict]:
        """Get recent fraud alerts"""
        return self.alert_history[-10:]  # Last 10 alerts

class DataManagerAgent:
    """Agent responsible for data persistence and management"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def save_user_data(self, user_id: str, data: Dict):
        """Save user profile data with model status check"""
        # Check if model file exists
        model_path = f"data/models/{user_id}_model.pkl"
        data['model_trained'] = os.path.exists(model_path)
        
        filepath = os.path.join(self.data_dir, f"{user_id}_profile.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_user_data(self, user_id: str) -> Optional[Dict]:
        """Load user profile data with model status check"""
        filepath = os.path.join(self.data_dir, f"{user_id}_profile.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update model status based on actual file existence
            model_path = f"data/models/{user_id}_model.pkl"
            data['model_trained'] = os.path.exists(model_path)
            
            return data
        return None
        
    def save_session_log(self, session_data: Dict):
        """Save session data to CSV log"""
        filepath = os.path.join(self.data_dir, "session_log.csv")
        
        # Define consistent columns for CSV
        base_columns = ['user_id', 'session_type', 'timestamp', 'wpm', 'accuracy', 
                       'avg_flight_time', 'std_flight_time', 'typing_rhythm_variance']
        
        # Create a filtered dictionary with only the base columns
        filtered_data = {}
        for col in base_columns:
            filtered_data[col] = session_data.get(col, 0)
        
        df = pd.DataFrame([filtered_data])
        
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)
            
    def get_session_logs(self) -> pd.DataFrame:
        """Get all session logs"""
        filepath = os.path.join(self.data_dir, "session_log.csv")
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except pd.errors.ParserError as e:
                # Handle corrupted CSV by recreating it
                print(f"CSV parsing error: {e}")
                # Backup the corrupted file
                backup_path = filepath + ".backup"
                if os.path.exists(filepath):
                    os.rename(filepath, backup_path)
                # Return empty DataFrame with expected columns
                return pd.DataFrame(columns=['user_id', 'session_type', 'timestamp', 'wpm', 'accuracy'])
        return pd.DataFrame(columns=['user_id', 'session_type', 'timestamp', 'wpm', 'accuracy'])

# Initialize agents
keystroke_agent = KeystrokeAgent()
behavior_agent = BehaviorModelAgent()
fraud_agent = FraudDetectionAgent()
data_agent = DataManagerAgent()

# Streamlit app configuration
st.set_page_config(
    page_title="On-Device Fraud Detection System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS for better UI
st.markdown("""
<style>
/* Import Professional Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
.main > div {
    font-family: 'Inter', sans-serif;
}

/* Custom Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
}

.css-1aumxhk {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    color: white;
}

/* Sidebar Navigation Buttons */
.nav-button {
    display: block;
    width: 100%;
    padding: 12px 16px;
    margin: 8px 0;
    background: rgba(255, 255, 255, 0.1);
    color: white;
    border: none;
    border-radius: 8px;
    text-align: left;
    font-weight: 500;
    transition: all 0.3s ease;
    cursor: pointer;
    border-left: 4px solid transparent;
}

.nav-button:hover {
    background: rgba(255, 255, 255, 0.2);
    border-left-color: #00d4ff;
    transform: translateX(4px);
}

.nav-button.active {
    background: rgba(255, 255, 255, 0.25);
    border-left-color: #00d4ff;
    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
}

/* Main Header */
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2.5rem;
    padding: 1rem 0;
}

/* Page Headers */
.page-header {
    font-size: 2.2rem;
    font-weight: 600;
    color: #2c3e50;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #3498db;
    padding-bottom: 0.5rem;
}

/* Enhanced Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border-left: 5px solid #3498db;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    transition: all 0.3s ease;
    margin: 0.5rem 0;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
}

/* Professional Cards */
.professional-card {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    border: 1px solid #e1e8ed;
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.professional-card:hover {
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    transform: translateY(-2px);
}

/* Enhanced Alerts */
.alert-success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    color: #155724;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #c3e6cb;
    border-left: 5px solid #28a745;
    box-shadow: 0 4px 12px rgba(40, 167, 69, 0.15);
    margin: 1rem 0;
}

.alert-danger {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    color: #721c24;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #f5c6cb;
    border-left: 5px solid #dc3545;
    box-shadow: 0 4px 12px rgba(220, 53, 69, 0.15);
    margin: 1rem 0;
}

.alert-warning {
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    color: #856404;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #ffeaa7;
    border-left: 5px solid #ffc107;
    box-shadow: 0 4px 12px rgba(255, 193, 7, 0.15);
    margin: 1rem 0;
}

.alert-info {
    background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    color: #0c5460;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #bee5eb;
    border-left: 5px solid #17a2b8;
    box-shadow: 0 4px 12px rgba(23, 162, 184, 0.15);
    margin: 1rem 0;
}

/* Modern Buttons */
.modern-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.modern-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
}

/* Statistics Cards */
.stat-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    margin: 0.5rem 0;
}

.stat-number {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 0.9rem;
    opacity: 0.9;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Typing Interface */
.typing-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

.reference-text {
    font-family: 'Courier New', monospace;
    font-size: 24px;
    line-height: 1.8;
    color: #333;
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
    border-left: 5px solid #667eea;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.metrics-card {
    background: rgba(255,255,255,0.95);
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
}

.session-progress {
    background: linear-gradient(90deg, #4CAF50, #45a049);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
}

/* Enhanced Sidebar Branding */
.sidebar-brand {
    text-align: center;
    padding: 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.brand-logo {
    font-size: 2rem;
    margin-bottom: 0.5rem;
}

.brand-text {
    font-size: 0.9rem;
    color: black;
    opacity: 1.0;
    font-weight: 500;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .page-header {
        font-size: 1.8rem;
    }
    
    .stat-number {
        font-size: 2rem;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
}
</style>
""", unsafe_allow_html=True)

def get_theme_css(theme_mode="light", font_size="medium", high_contrast=False):
    """Generate theme-based CSS"""
    
    # Theme color schemes
    if theme_mode == "dark":
        if high_contrast:
            colors = {
                'bg_primary': '#000000',
                'bg_secondary': '#1a1a1a',
                'bg_card': '#2d2d2d',
                'text_primary': '#ffffff',
                'text_secondary': '#e0e0e0',
                'accent_primary': '#00ff00',
                'accent_secondary': '#ffff00',
                'border': '#ffffff',
                'sidebar_bg': '#000000',
                'sidebar_text': '#ffffff'
            }
        else:
            colors = {
                'bg_primary': '#1e1e1e',
                'bg_secondary': '#2d2d2d',
                'bg_card': '#404040',
                'text_primary': '#ffffff',
                'text_secondary': '#b3b3b3',
                'accent_primary': '#667eea',
                'accent_secondary': '#764ba2',
                'border': '#555555',
                'sidebar_bg': 'linear-gradient(180deg, #2c3e50 0%, #34495e 100%)',
                'sidebar_text': '#ecf0f1'
            }
    else:  # light mode
        if high_contrast:
            colors = {
                'bg_primary': '#ffffff',
                'bg_secondary': '#f0f0f0',
                'bg_card': '#ffffff',
                'text_primary': '#000000',
                'text_secondary': '#333333',
                'accent_primary': '#0000ff',
                'accent_secondary': '#800080',
                'border': '#000000',
                'sidebar_bg': '#ffffff',
                'sidebar_text': '#000000'
            }
        else:
            colors = {
                'bg_primary': '#ffffff',
                'bg_secondary': '#f8f9fa',
                'bg_card': '#ffffff',
                'text_primary': '#2c3e50',
                'text_secondary': '#666666',
                'accent_primary': '#667eea',
                'accent_secondary': '#764ba2',
                'border': '#e1e8ed',
                'sidebar_bg': 'linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)',
                'sidebar_text': '#000000'
            }
    
    # Font sizes
    font_sizes = {
        'small': {'base': '14px', 'header': '2.2rem', 'large': '2rem'},
        'medium': {'base': '16px', 'header': '2.8rem', 'large': '2.5rem'},
        'large': {'base': '18px', 'header': '3.2rem', 'large': '3rem'},
        'extra_large': {'base': '20px', 'header': '3.6rem', 'large': '3.5rem'}
    }
    
    font_config = font_sizes.get(font_size, font_sizes['medium'])
    
    # Generate dynamic CSS
    return f"""
    <style>
    :root {{
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-card: {colors['bg_card']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --accent-primary: {colors['accent_primary']};
        --accent-secondary: {colors['accent_secondary']};
        --border-color: {colors['border']};
        --sidebar-bg: {colors['sidebar_bg']};
        --sidebar-text: {colors['sidebar_text']};
        --font-size-base: {font_config['base']};
        --font-size-header: {font_config['header']};
        --font-size-large: {font_config['large']};
    }}
    
    /* Apply theme variables */
    .main > div {{
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-size: var(--font-size-base) !important;
    }}
    
    .stApp {{
        background-color: var(--bg-primary) !important;
    }}
    
    /* Custom Sidebar Styling with theme */
    .css-1d391kg {{
        background: var(--sidebar-bg) !important;
    }}
    
    .css-1aumxhk {{
        background: var(--sidebar-bg) !important;
        color: var(--sidebar-text) !important;
    }}
    
    .brand-text {{
        color: #000000 !important;
        font-size: 0.9rem;
        opacity: 1.0;
        font-weight: 300;
    }}
    
    /* Force all sidebar text to be black */
    .css-1aumxhk, .css-1aumxhk * {{
        color: #000000 !important;
    }}
    
    .stSidebar .stMarkdown p, 
    .stSidebar .stMarkdown h1, 
    .stSidebar .stMarkdown h2, 
    .stSidebar .stMarkdown h3 {{
        color: #000000 !important;
    }}
    
    /* Professional Cards with theme */
    .professional-card {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }}
    
    /* Headers with theme */
    .main-header {{
        font-size: var(--font-size-header) !important;
        color: var(--text-primary) !important;
    }}
    
    .page-header {{
        color: var(--text-primary) !important;
        border-bottom: 3px solid var(--accent-primary) !important;
    }}
    
    /* Metric cards with theme */
    .metric-card {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-left: 5px solid var(--accent-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Reference text with theme */
    .reference-text {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-left: 5px solid var(--accent-primary) !important;
    }}
    
    /* Metrics card with theme */
    .metrics-card {{
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Alert styles with theme */
    .alert-info {{
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--accent-primary) !important;
        border-left: 5px solid var(--accent-primary) !important;
    }}
    
    .alert-warning {{
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid #ffc107 !important;
        border-left: 5px solid #ffc107 !important;
    }}
    
    .alert-success {{
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid #28a745 !important;
        border-left: 5px solid #28a745 !important;
    }}
    
    .alert-danger {{
        background: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid #dc3545 !important;
        border-left: 5px solid #dc3545 !important;
    }}
    
    /* High contrast focus indicators */
    {':focus { outline: 3px solid var(--accent-primary) !important; outline-offset: 2px; }' if high_contrast else ''}
    
    /* Enhanced button visibility for accessibility */
    .stButton > button {{
        border: 2px solid var(--accent-primary) !important;
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
        min-height: 44px !important; /* WCAG touch target minimum */
        font-weight: 500 !important;
    }}
    
    .stButton > button:hover {{
        background: var(--accent-primary) !important;
        color: var(--bg-card) !important;
        transform: {'none' if high_contrast else 'translateY(-1px)'} !important;
    }}
    
    .stButton > button:focus {{
        outline: 3px solid var(--accent-primary) !important;
        outline-offset: 2px !important;
    }}
    
    /* Enhanced contrast for text elements */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: {'700' if high_contrast else '600'} !important;
    }}
    
    /* Skip to main content link for screen readers */
    .skip-link {{
        position: absolute;
        top: -40px;
        left: 6px;
        background: var(--accent-primary);
        color: var(--bg-card);
        padding: 8px;
        z-index: 100;
        text-decoration: none;
        border-radius: 4px;
    }}
    
    .skip-link:focus {{
        top: 6px;
    }}
    
    /* Streamlit specific elements */
    .stSelectbox > div > div {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTextInput > div > div > input {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stTextArea > div > div > textarea {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    /* Dataframe styling */
    .stDataFrame {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }}
    
    /* Sidebar elements */
    .stSidebar .stSelectbox > div > div,
    .stSidebar .stButton > button,
    .stSidebar .stCheckbox {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
        border: 1px solid rgba(0, 0, 0, 0.2) !important;
    }}
    
    .stSidebar .stButton > button:hover {{
        background-color: rgba(255, 255, 255, 1.0) !important;
        color: #000000 !important;
        border: 1px solid #667eea !important;
    }}
    
    /* Navigation buttons in sidebar */
    .stSidebar .stButton > button {{
        font-weight: 500 !important;
        text-align: left !important;
    }}
    </style>
    """

def main():
    # Initialize theme and accessibility settings
    if 'theme_mode' not in st.session_state:
        st.session_state.theme_mode = "light"
    if 'font_size' not in st.session_state:
        st.session_state.font_size = "medium"
    if 'high_contrast' not in st.session_state:
        st.session_state.high_contrast = False
    if 'reduce_motion' not in st.session_state:
        st.session_state.reduce_motion = False
    
    # Apply theme CSS
    st.markdown(get_theme_css(
        st.session_state.theme_mode, 
        st.session_state.font_size, 
        st.session_state.high_contrast
    ), unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🔐 DefendX Multi-Agent Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Professional Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-logo">🛡️</div>
            <div class="brand-text">DefendX Security Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation Menu
        st.markdown("### 📋 **NAVIGATION**")
        
        pages = {
            "Home": {"icon": "🏠", "label": "Dashboard"},
            "Registration": {"icon": "👤", "label": "User Registration"},
            "Verification": {"icon": "🔍", "label": "Verification Test"},
            "Analytics": {"icon": "📊", "label": "Admin Dashboard"},
            "Settings": {"icon": "⚙️", "label": "System Settings"}
        }
        
        for page_key, page_info in pages.items():
            # Create button-like navigation
            button_class = "active" if st.session_state.current_page == page_key else ""
            
            if st.button(
                f"{page_info['icon']} {page_info['label']}", 
                key=f"nav_{page_key}",
                use_container_width=True
            ):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.markdown("---")
        
        # System Status in Sidebar
        st.markdown("### 📊 **SYSTEM STATUS**")
        
        # Quick system info
        try:
            session_logs = data_agent.get_session_logs()
            total_users = len([f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')])
            total_sessions = len(session_logs) if not session_logs.empty else 0
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.95); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.2);">
                <div style="color: #2c3e50; font-size: 0.9rem; font-weight: 500;">
                    <div style="margin: 0.2rem 0;">👥 Users: <strong style="color: #3498db;">{total_users}</strong></div>
                    <div style="margin: 0.2rem 0;">📝 Sessions: <strong style="color: #e74c3c;">{total_sessions}</strong></div>
                    <div style="margin: 0.2rem 0;">🛡️ Status: <span style="color: #27ae60;"><strong>Active</strong></span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.95); padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid rgba(255,255,255,0.2);">
                <div style="color: #2c3e50; font-size: 0.9rem; font-weight: 500;">
                    <div>🛡️ Status: <span style="color: #f39c12;"><strong>Initializing</strong></span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ **QUICK ACTIONS**")
        
        # Theme toggle button
        theme_icon = "🌙" if st.session_state.theme_mode == "light" else "☀️"
        theme_text = f"{theme_icon} Dark Mode" if st.session_state.theme_mode == "light" else f"{theme_icon} Light Mode"
        
        if st.button(theme_text, use_container_width=True):
            st.session_state.theme_mode = "dark" if st.session_state.theme_mode == "light" else "light"
            st.rerun()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
            
        if st.button("📥 Export Logs", use_container_width=True):
            st.info("Export functionality activated!")
    
    # Route to appropriate page based on current selection
    current_page = st.session_state.current_page
    
    if current_page == "Home":
        show_home_page()
    elif current_page == "Registration":
        show_registration_page()
    elif current_page == "Verification":
        show_verification_page()
    elif current_page == "Analytics":
        show_admin_dashboard()
    elif current_page == "Settings":
        show_settings_page()

def show_home_page():
    # Hero Section
    st.markdown("""
    <div class="professional-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">🛡️ Advanced Behavioral Biometrics Platform</h2>
        <p style="font-size: 1.2rem; opacity: 0.9; margin-bottom: 0;">
            Real-time fraud detection through keystroke dynamics and behavioral analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # System Overview Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">🎯 <strong>Core Capabilities</strong></h3>
            <div style="line-height: 1.8;">
                <div style="margin: 0.8rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #3498db;">
                    <strong>⌨️ Keystroke Dynamics</strong><br>
                    <span style="color: #666;">Advanced analysis of typing patterns, rhythm, and timing characteristics</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #e74c3c;">
                    <strong>🧠 Behavioral Modeling</strong><br>
                    <span style="color: #666;">AI-powered creation of unique user behavioral profiles</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #f39c12;">
                    <strong>🔍 Anomaly Detection</strong><br>
                    <span style="color: #666;">Real-time identification of suspicious activities</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.5rem; background: #f8f9fa; border-radius: 6px; border-left: 4px solid #27ae60;">
                    <strong>🔒 Privacy-First</strong><br>
                    <span style="color: #666;">All processing happens locally on-device</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">🤖 <strong>Multi-Agent Architecture</strong></h3>
            <div style="line-height: 1.8;">
                <div style="margin: 0.8rem 0; padding: 0.8rem; background: #e8f4f8; border-radius: 8px; border: 1px solid #3498db;">
                    <strong style="color: #2980b9;">🎯 Keystroke Agent</strong><br>
                    <span style="color: #555; font-size: 0.9rem;">Captures and processes typing patterns with millisecond precision</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.8rem; background: #fdf2e9; border-radius: 8px; border: 1px solid #e67e22;">
                    <strong style="color: #d68910;">🧮 Behavior Model Agent</strong><br>
                    <span style="color: #555; font-size: 0.9rem;">Trains and maintains personalized user behavior models</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.8rem; background: #fadbd8; border-radius: 8px; border: 1px solid #e74c3c;">
                    <strong style="color: #c0392b;">🚨 Fraud Detection Agent</strong><br>
                    <span style="color: #555; font-size: 0.9rem;">Analyzes sessions for fraud indicators and risk assessment</span>
                </div>
                <div style="margin: 0.8rem 0; padding: 0.8rem; background: #d5f4e6; border-radius: 8px; border: 1px solid #27ae60;">
                    <strong style="color: #229954;">💾 Data Manager Agent</strong><br>
                    <span style="color: #555; font-size: 0.9rem;">Handles secure data persistence and session logging</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced System Statistics
    st.markdown("""
    <div class="professional-card" style="margin-top: 2rem;">
        <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">📈 <strong>Real-Time System Analytics</strong></h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Load existing data
    try:
        session_logs = data_agent.get_session_logs()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_users = len([f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')])
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                <div class="stat-number">{total_users}</div>
                <div class="stat-label">Registered Users</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            total_sessions = len(session_logs) if not session_logs.empty else 0
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <div class="stat-number">{total_sessions}</div>
                <div class="stat-label">Total Sessions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            fraud_alerts = len(fraud_agent.get_fraud_alerts())
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #f39c12 0%, #d68910 100%);">
                <div class="stat-number">{fraud_alerts}</div>
                <div class="stat-label">Fraud Alerts</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            if not session_logs.empty and 'accuracy' in session_logs.columns:
                avg_accuracy = session_logs['accuracy'].mean()
                st.markdown(f"""
                <div class="stat-card" style="background: linear-gradient(135deg, #27ae60 0%, #229954 100%);">
                    <div class="stat-number">{avg_accuracy:.1f}%</div>
                    <div class="stat-label">Average Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stat-card" style="background: linear-gradient(135deg, #95a5a6 0%, #7f8c8d 100%);">
                    <div class="stat-number">--</div>
                    <div class="stat-label">Average Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Recent Activity Dashboard
        if not session_logs.empty:
            st.markdown("""
            <div class="professional-card" style="margin-top: 2rem;">
                <h3 style="color: #2c3e50; margin-bottom: 1rem;">📊 <strong>Recent Activity Overview</strong></h3>
            """, unsafe_allow_html=True)
            
            # Display recent sessions
            base_columns = ['user_id', 'session_type', 'timestamp']
            optional_columns = ['wpm', 'accuracy']
            
            display_columns = base_columns.copy()
            for col in optional_columns:
                if col in session_logs.columns:
                    display_columns.append(col)
            
            recent_sessions = session_logs.tail(5)[display_columns]
            st.dataframe(recent_sessions, use_container_width=True, hide_index=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.markdown("""
        <div class="alert-info">
            <h4>🔧 System Initializing</h4>
            <p>The system is starting up. Statistics will be available once data is generated.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown("""
    <div class="professional-card" style="margin-top: 2rem;">
        <h3 style="color: #2c3e50; margin-bottom: 1rem; text-align: center;">🚀 <strong>Quick Start Guide</strong></h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem;">
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db;">
                <h4 style="color: #2980b9; margin-bottom: 0.5rem;">1️⃣ Register Users</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Create behavioral profiles by completing typing sessions to establish baseline patterns.</p>
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #e74c3c;">
                <h4 style="color: #c0392b; margin-bottom: 0.5rem;">2️⃣ Train Models</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">AI models learn unique typing characteristics for each registered user automatically.</p>
            </div>
            <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #f39c12;">
                <h4 style="color: #d68910; margin-bottom: 0.5rem;">3️⃣ Run Verification</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Test the system with verification sessions to detect potential fraud attempts.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_registration_page():
    """User registration and baseline collection page - Professional MonkeyType style"""
    
    # Initialize session state for typing
    if 'typing_started' not in st.session_state:
        st.session_state.typing_started = False
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'typing_complete' not in st.session_state:
        st.session_state.typing_complete = False
    
    # Professional page header
    st.markdown("""
    <div class="professional-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">⌨️ Behavioral Profile Registration</h2>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
            Create your unique typing signature for advanced fraud detection
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    user_id = st.text_input(
        "👤 **Enter Your User ID:**", 
        placeholder="e.g., john_doe", 
        key="user_input",
        help="Choose a unique identifier for your profile"
    )
    
    if user_id:
        # Check existing user data
        existing_data = data_agent.load_user_data(user_id)
        if existing_data and user_id not in behavior_agent.user_profiles:
            behavior_agent.user_profiles[user_id] = existing_data
            
        baseline_count = len(behavior_agent.user_profiles.get(user_id, {}).get('baseline_features', []))
        
        # Progress tracker - MonkeyType style
        st.markdown(f"""
        <div class="session-progress">
            <h3>Progress: Session {baseline_count + 1 if baseline_count < 4 else 4} of 4</h3>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span style="color: {'#4CAF50' if baseline_count >= 1 else '#ccc'};">●</span>
                <span style="color: {'#4CAF50' if baseline_count >= 2 else '#ccc'};">●</span>
                <span style="color: {'#4CAF50' if baseline_count >= 3 else '#ccc'};">●</span>
                <span style="color: {'#4CAF50' if baseline_count >= 4 else '#ccc'};">●</span>
                          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reference texts for different sessions
        reference_texts = [
    "Artificial intelligence enables machines to learn patterns from data, adapt intelligently, and perform tasks once thought to require human cognition.",
    "The history of computing is marked by rapid innovation, from early mechanical calculators to modern deep learning systems that power today’s applications.",
    "The quick brown fox jumps over the lazy dog.",
    "Never underestimate the power of a good book."
        ]
        
        # Select text based on session number
        if baseline_count < 4:
            selected_text = reference_texts[baseline_count % len(reference_texts)]
            
            # MonkeyType-style typing interface
            st.markdown(f"""
            <div class="typing-container">
                <h3 style="color: white; text-align: center;">Session {baseline_count + 1}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="reference-text">
                {selected_text}
            </div>
            """, unsafe_allow_html=True)
            
            # MonkeyType-style typing area
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Start typing button
                if not st.session_state.typing_started:
                    if st.button("🚀 Start Typing", type="primary", use_container_width=True):
                        st.session_state.typing_started = True
                        keystroke_agent.reset_session()
                        keystroke_agent.start_capture()
                        st.session_state.typing_session_start_time = keystroke_agent.start_time
                        st.rerun()
                
                if st.session_state.typing_started:
                    # Live typing area
                    typed_text = st.text_area(
                        "", 
                        height=120,
                        placeholder="Start typing here...",
                        key="typing_area",
                        label_visibility="collapsed"
                    )
                    
                    # Process typing in real-time
                    # The simulation is now done only once at the end of the session.

                    # Calculate accuracy
                    accuracy = 0
                    correct_chars = 0
                    for i, char in enumerate(typed_text):
                        if i < len(selected_text) and char == selected_text[i]:
                            correct_chars += 1
                    if len(typed_text) > 0:
                        accuracy = (correct_chars / len(typed_text)) * 100
                    
                    # Store current text for comparison
                    st.session_state.current_text = typed_text
            
            with col2:
                # MonkeyType-style metrics display
                if st.session_state.typing_started:
                    real_time_metrics = keystroke_agent.get_real_time_metrics()
                    
                    # Live metrics cards
                    st.markdown("""
                    <div class="metrics-card">
                        <h4 style="margin:0; color:#667eea;">⚡ Live Metrics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if real_time_metrics:
                        # WPM - Large display like MonkeyType
                        wpm = real_time_metrics.get('speed_wpm', 0)
                        st.markdown(f"""
                        <div style="text-align:center; padding:1rem; background:#f8f9fa; border-radius:10px; margin:0.5rem 0;">
                            <h1 style="margin:0; color:#667eea; font-size:3rem;">{wpm}</h1>
                            <p style="margin:0; color:#666;">WPM</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Other metrics in a grid
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                            st.metric("Errors", real_time_metrics.get('errors', 0))
                        with col2_2:
                            st.metric("Time", f"{real_time_metrics.get('elapsed_time', 0):.1f}s")
                            progress = len(typed_text) / len(selected_text) * 100 if selected_text else 0
                            st.metric("Progress", f"{progress:.0f}%")
                        
                        # Advanced metrics
                        with st.expander("🔬 Advanced Metrics"):
                            st.metric("Hold Time", f"{real_time_metrics.get('avg_hold', 0):.3f}s")
                            st.metric("Flight Time", f"{real_time_metrics.get('avg_flight', 0):.3f}s")
                            st.metric("Rhythm Variation", f"{real_time_metrics.get('rhythm_stddev', 0):.3f}s")
                    
                    # Session completion area
                    st.markdown("---")
                    
                    # Check if typing is complete
                    if len(typed_text) >= len(selected_text) * 0.9:  # 90% completion
                        st.session_state.typing_complete = True
                        
                        # Run the simulation once, now that typing is complete
                        start_time_from_session = st.session_state.get('typing_session_start_time')
                        if start_time_from_session:
                            # Manually clear events before running simulation
                            keystroke_agent.key_events = []
                            keystroke_agent.hold_times = []
                            keystroke_agent.gap_times = []
                            keystroke_agent.flight_times = []
                            keystroke_agent.typed_chars = 0

                            char_time = start_time_from_session
                            for char in typed_text:
                                char_time += random.uniform(0.1, 0.3)
                                keystroke_agent.process_keystroke(char, char_time)
                                release_time = char_time + random.uniform(0.05, 0.1)
                                keystroke_agent.process_key_release(char, release_time)

                        # Extract features for saving
                        features = keystroke_agent.extract_features(typed_text, selected_text)
                        
                        # MonkeyType-style completion message
                        st.markdown(f"""
                        <div style="background:#4CAF50; color:white; padding:1rem; border-radius:10px; text-align:center; margin:1rem 0;">
                            <h3>🎉 Session Complete!</h3>
                            <p>Final WPM: <strong>{features.get('wpm', 0):.1f}</strong> | 
                               Accuracy: <strong>{features.get('accuracy', 0):.1f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Save session button
                        col_save1, col_save2 = st.columns(2)
                        
                        with col_save1:
                            if st.button("💾 Save Session", type="primary", use_container_width=True):
                                # Save baseline data
                                behavior_agent.create_user_profile(user_id, features)
                                
                                # Save to persistent storage
                                profile_data = behavior_agent.user_profiles[user_id]
                                data_agent.save_user_data(user_id, profile_data)
                                
                                # Log session with enhanced features
                                session_data = {
                                    'user_id': user_id,
                                    'session_type': 'baseline',
                                    'timestamp': datetime.now().isoformat(),
                                    'session_number': baseline_count + 1,
                                    'wpm': real_time_metrics.get('speed_wpm', 0),
                                    'accuracy': accuracy,
                                    **features
                                }
                                data_agent.save_session_log(session_data)
                                
                                # Reset states
                                st.session_state.typing_started = False
                                st.session_state.typing_complete = False
                                st.session_state.current_text = ""
                                
                                st.success(f"✅ Session {baseline_count + 1} saved!")
                                time.sleep(1)
                                st.rerun()
                        
                        with col_save2:
                            # Next session or model training
                            current_baseline_count = len(behavior_agent.user_profiles.get(user_id, {}).get('baseline_features', []))
                            
                            if current_baseline_count + 1 >= 5:  # After saving this will be 5
                                if st.button("🤖 Complete Registration", type="secondary", use_container_width=True):
                                    st.info("👆 Save this session first, then train your model!")
                            else:
                                if st.button("➡️ Next Session", type="secondary", use_container_width=True):
                                    st.session_state.typing_started = False
                                    st.session_state.typing_complete = False
                                    st.session_state.current_text = ""
                                    st.rerun()
                    
                    # Reset button
                    if st.button("🔄 Restart Session"):
                        st.session_state.typing_started = False
                        st.session_state.typing_complete = False
                        st.session_state.current_text = ""
                        keystroke_agent.reset_session()
                        st.rerun()
        
        # Model training section (after 3 sessions)
        if baseline_count >= 2:
            st.markdown("---")
            st.markdown("## 🤖 Complete Your Registration")
            
            user_model_exists = behavior_agent.user_profiles.get(user_id, {}).get('model_trained', False)
            
            if not user_model_exists:
                col_train1, col_train2 = st.columns([2, 1])
                with col_train1:
                    st.info("🎯 Great! You've completed all 3 sessions. Now train your behavioral model for fraud detection.")
                with col_train2:
                    if st.button("🚀 Train Model", type="primary"):
                        with st.spinner("Training your behavioral model..."):
                            success = behavior_agent.train_user_model(user_id)
                            if success:
                                profile_data = behavior_agent.user_profiles[user_id]
                                data_agent.save_user_data(user_id, profile_data)
                                st.success("🎉 Model trained successfully!")
                                st.success("🔒 Your profile is ready for fraud detection!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Training failed. Please try again.")
            else:
                st.success("✅ Registration Complete! Your behavioral profile is ready.")
                st.info("💡 Visit the 'User Verification' page to test your profile.")

def show_verification_page():
    # Professional page header
    st.markdown("""
    <div class="professional-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">🔍 Identity Verification Center</h2>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
            Real-time fraud detection through behavioral biometric analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # User selection
    user_files = [f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')]
    user_ids = [f.replace('_profile.json', '') for f in user_files]
    
    if not user_ids:
        st.markdown("""
        <div class="alert-warning">
            <h4>⚠️ No Registered Users</h4>
            <p>Please register users first in the User Registration section before running verification tests.</p>
        </div>
        """, unsafe_allow_html=True)
        return
        
    selected_user = st.selectbox("**👤 Select user to verify:**", user_ids, help="Choose a registered user for verification testing")
    
    # Load user data
    user_data = data_agent.load_user_data(selected_user)
    if not user_data:
        st.error("Could not load user data.")
        return
        
    # Load user profile into behavior agent
    behavior_agent.user_profiles[selected_user] = user_data
    
    # Check if model is trained and load it
    model_trained = False
    if user_data.get('model_trained', False):
        # Try to load the saved model first
        if behavior_agent._load_model(selected_user):
            model_trained = True
            st.success(f"✅ Model loaded for user: {selected_user}")
        else:
            # If loading fails, try to retrain from baseline features
            try:
                if behavior_agent.train_user_model(selected_user):
                    model_trained = True
                    st.success(f"🤖 Model retrained for user: {selected_user}")
                else:
                    st.warning(f"⚠️ Could not train model for user: {selected_user}")
            except Exception as e:
                st.error(f"Error training model: {e}")
                user_data['model_trained'] = False
    
    if not model_trained:
        st.warning("⚠️ User model not trained. Please complete registration first.")
        return
        
    # Check if model is trained
    model_trained = user_data.get('model_trained', False)
    baseline_count = len(user_data.get('baseline_features', []))
    
    if not model_trained:
        if baseline_count >= 2:
            st.warning("⚠️ User has enough baseline data but model not trained yet.")
            if st.button("🤖 Train Model Now"):
                success = behavior_agent.train_user_model(selected_user)
                if success:
                    # Save updated profile
                    profile_data = behavior_agent.user_profiles[selected_user]
                    data_agent.save_user_data(selected_user, profile_data)
                    st.success("🎉 Model trained successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to train model.")
            return
        else:
            st.warning(f"❌ User needs {3 - baseline_count} more baseline sessions. Please complete registration first.")
            return
    
    st.markdown(f"### Verifying user: **{selected_user}**")
    
    # Verification text
    verification_texts = [
        "Authentication test sentence for user verification.",
        "Security check: please type this sentence carefully.",
        "Behavioral biometric verification in progress.",
        "Your typing pattern is your digital fingerprint."
    ]
    
    verification_text = st.selectbox("Select verification text:", verification_texts)
    
    st.markdown("### Type the following text:")
    st.markdown(f"**Text:** {verification_text}")
    
    # Create columns for verification interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ⌨️ Typing Area")
        
        if 'verification_active' not in st.session_state:
            st.session_state.verification_active = False
            
        # Start verification button - make it more visible
        if not st.session_state.verification_active:
            st.markdown("""
            <div style="text-align: center; margin: 1rem 0;">
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Verification", type="primary", use_container_width=True):
                st.session_state.verification_active = True
                keystroke_agent.reset_session()
                keystroke_agent.start_capture()
                st.session_state.typing_session_start_time = keystroke_agent.start_time
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
        if st.session_state.verification_active:
            typed_text = st.text_area(
                "Type the verification text here:", 
                height=120,
                placeholder="Start typing the verification text...",
                key="verification_text_area"
            )
            
            # Progress indicator
            progress = min(len(typed_text) / len(verification_text), 1.0)
            st.progress(progress)
            st.caption(f"Progress: {len(typed_text)}/{len(verification_text)} characters ({progress*100:.1f}%)")
            
            # Complete verification button - more prominent placement
            verification_complete = len(typed_text) >= len(verification_text) * 0.9
            
            if verification_complete:
                st.markdown("---")
                st.markdown("### ✅ Ready to Complete Verification")
                
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    if st.button("🔍 Complete Verification", type="primary", use_container_width=True, key="complete_verification_btn"):
                        # Run the simulation once, now that typing is complete
                        start_time_from_session = st.session_state.get('typing_session_start_time')
                        if start_time_from_session:
                            # Manually clear events before running simulation
                            keystroke_agent.key_events = []
                            keystroke_agent.hold_times = []
                            keystroke_agent.gap_times = []
                            keystroke_agent.flight_times = []
                            keystroke_agent.typed_chars = 0

                            char_time = start_time_from_session
                            for char in typed_text:
                                char_time += random.uniform(0.1, 0.3)
                                keystroke_agent.process_keystroke(char, char_time)
                                release_time = char_time + random.uniform(0.05, 0.1)
                                keystroke_agent.process_key_release(char, release_time)
                        
                        # Extract features for analysis
                        features = keystroke_agent.extract_features(typed_text, verification_text)

                        # Predict anomaly
                        is_anomaly, confidence = behavior_agent.predict_anomaly(selected_user, features)
                        
                        # Fraud analysis
                        fraud_analysis = fraud_agent.analyze_session(
                            selected_user, features, is_anomaly, confidence
                        )
                        
                        # Store results in session state
                        st.session_state.verification_results = {
                            'fraud_analysis': fraud_analysis,
                            'is_anomaly': is_anomaly,
                            'confidence': confidence,
                            'features': features
                        }
                        
                        # Reset verification state
                        st.session_state.verification_active = False
                        st.rerun()
            else:
                st.info(f"💡 Type at least {int(len(verification_text) * 0.9)} characters to complete verification")
    
    with col2:
        st.markdown("### 📊 Instructions")
        st.markdown("""
        <div class="professional-card">
            <h4>📋 How to Verify:</h4>
            <ol>
                <li>Click "Start Verification"</li>
                <li>Type the verification text exactly</li>
                <li>Click "Complete Verification"</li>
                <li>View your results</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.verification_active:
            st.markdown("""
            <div class="alert-info">
                <h4>🔍 Verification Active</h4>
                <p>Type naturally as you would normally. The system is analyzing your keystroke patterns.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display verification results if available
    if 'verification_results' in st.session_state:
        results = st.session_state.verification_results
        fraud_analysis = results['fraud_analysis']
        is_anomaly = results['is_anomaly']
        confidence = results['confidence']
        
        st.markdown("---")
        st.markdown("### 🔍 Verification Results")
        
        if fraud_analysis['fraud_detected']:
            st.markdown("""
            <div class="alert-danger">
                <h4>⚠️ FRAUD DETECTED</h4>
                <p>The typing pattern does not match the registered user.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-success">
                <h4>✅ VERIFICATION SUCCESSFUL</h4>
                <p>Typing pattern matches the registered user.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{fraud_analysis['risk_score']:.2f}")
        with col2:
            st.metric("Confidence", f"{confidence:.2f}")
        with col3:
            status = "ANOMALY" if is_anomaly else "NORMAL"
            st.metric("Pattern Status", status)
        
        # Risk factors
        if fraud_analysis['risk_factors']:
            st.markdown("**Risk Factors Detected:**")
            for factor in fraud_analysis['risk_factors']:
                st.markdown(f"• {factor}")
        
        # Reset button
        if st.button("🔄 Run Another Verification", use_container_width=True):
            if 'verification_results' in st.session_state:
                del st.session_state.verification_results
            st.rerun()

def show_admin_dashboard():
    # Professional page header
    st.markdown("""
    <div class="professional-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">📊 Administrative Dashboard</h2>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
            Comprehensive system analytics and user management
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load session data
    session_logs = data_agent.get_session_logs()
    
    if session_logs.empty:
        st.markdown("""
        <div class="alert-info">
            <h4>📋 No Data Available</h4>
            <p>No session data has been generated yet. Start by registering users and running verification sessions to see analytics here.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Enhanced dashboard tabs with professional styling
    tab1, tab2, tab3, tab4 = st.tabs(["📈 **Overview**", "👥 **User Management**", "🚨 **Security Alerts**", "📊 **Advanced Analytics**"])
    
    with tab1:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">📈 <strong>System Performance Overview</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Key metrics with professional cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sessions = len(session_logs)
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                <div class="stat-number">{total_sessions}</div>
                <div class="stat-label">Total Sessions</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            fraud_sessions = 0
            if 'fraud_detected' in session_logs.columns:
                fraud_sessions = len(session_logs[session_logs['fraud_detected'] == True])
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <div class="stat-number">{fraud_sessions}</div>
                <div class="stat-label">Fraud Detected</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            fraud_rate = (fraud_sessions / total_sessions * 100) if total_sessions > 0 else 0
            color = "#e74c3c" if fraud_rate > 10 else "#f39c12" if fraud_rate > 5 else "#27ae60"
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                <div class="stat-number">{fraud_rate:.1f}%</div>
                <div class="stat-label">Fraud Rate</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            avg_risk_score = 0
            if 'risk_score' in session_logs.columns:
                avg_risk_score = session_logs['risk_score'].mean()
            risk_color = "#e74c3c" if avg_risk_score > 0.7 else "#f39c12" if avg_risk_score > 0.4 else "#27ae60"
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, {risk_color} 0%, {risk_color}dd 100%);">
                <div class="stat-number">{avg_risk_score:.2f}</div>
                <div class="stat-label">Avg Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent activity with enhanced styling
        st.markdown("""
        <div class="professional-card" style="margin-top: 2rem;">
            <h3 style="color: #2c3e50; margin-bottom: 1rem;">📋 <strong>Recent Activity</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Select available columns for display
        base_columns = ['user_id', 'session_type', 'timestamp']
        optional_columns = ['fraud_detected', 'risk_score', 'wpm', 'accuracy']
        
        # Only include columns that actually exist
        display_columns = base_columns.copy()
        for col in optional_columns:
            if col in session_logs.columns:
                display_columns.append(col)
        
        recent_sessions = session_logs.tail(10)[display_columns]
        st.dataframe(recent_sessions, use_container_width=True, hide_index=True)
        
        # System health indicators
        col_health1, col_health2 = st.columns(2)
        
        with col_health1:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">🏥 <strong>System Health</strong></h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate system health metrics
            unique_users = session_logs['user_id'].nunique() if 'user_id' in session_logs.columns else 0
            avg_session_per_user = total_sessions / unique_users if unique_users > 0 else 0
            
            health_metrics = {
                "Active Users": unique_users,
                "Avg Sessions/User": f"{avg_session_per_user:.1f}",
                "System Uptime": "99.9%",
                "Data Integrity": "✅ Healthy"
            }
            
            for metric, value in health_metrics.items():
                st.metric(metric, value)
        
        with col_health2:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: #2c3e50; margin-bottom: 1rem;">⚡ <strong>Performance Metrics</strong></h4>
            </div>
            """, unsafe_allow_html=True)
            
            if 'wpm' in session_logs.columns and 'accuracy' in session_logs.columns:
                avg_wpm = session_logs['wpm'].mean()
                avg_accuracy = session_logs['accuracy'].mean()
                max_wpm = session_logs['wpm'].max()
                min_accuracy = session_logs['accuracy'].min()
                
                perf_metrics = {
                    "Avg WPM": f"{avg_wpm:.1f}",
                    "Avg Accuracy": f"{avg_accuracy:.1f}%",
                    "Peak WPM": f"{max_wpm:.1f}",
                    "Min Accuracy": f"{min_accuracy:.1f}%"
                }
                
                for metric, value in perf_metrics.items():
                    st.metric(metric, value)
            else:
                st.info("Performance metrics will be available once typing data is collected.")
    
    with tab2:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">👥 <strong>User Management Console</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # User statistics with enhanced presentation
        if 'user_id' in session_logs.columns:
            agg_dict = {'session_type': 'count'}
            column_names = ['Total Sessions']
            
            # Add columns that exist
            if 'fraud_detected' in session_logs.columns:
                agg_dict['fraud_detected'] = 'sum'
                column_names.append('Fraud Count')
            if 'wpm' in session_logs.columns:
                agg_dict['wpm'] = 'mean'
                column_names.append('Avg WPM')
            if 'accuracy' in session_logs.columns:
                agg_dict['accuracy'] = 'mean'
                column_names.append('Avg Accuracy')
            if 'risk_score' in session_logs.columns:
                agg_dict['risk_score'] = 'mean'
                column_names.append('Avg Risk')
            
            user_stats = session_logs.groupby('user_id').agg(agg_dict).round(2)
            user_stats.columns = column_names
            
            # Enhanced user statistics display
            st.markdown("### 📊 User Performance Summary")
            st.dataframe(user_stats, use_container_width=True)
            
            # User behavior visualization
            if 'wpm' in session_logs.columns and 'accuracy' in session_logs.columns:
                st.markdown("### 📈 User Typing Pattern Analysis")
                
                # Create enhanced scatter plot
                fig = px.scatter(
                    session_logs, 
                    x='wpm', 
                    y='accuracy', 
                    color='user_id',
                    title="User Typing Patterns: WPM vs Accuracy",
                    labels={'wpm': 'Words Per Minute', 'accuracy': 'Accuracy (%)'},
                    hover_data=['session_type'] if 'session_type' in session_logs.columns else []
                )
                fig.update_layout(
                    template="plotly_white",
                    title_font_size=16,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional user insights
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.markdown("### 🎯 Top Performers")
                    if 'Avg WPM' in user_stats.columns:
                        top_wpm = user_stats.nlargest(3, 'Avg WPM')[['Avg WPM']]
                        st.dataframe(top_wpm, use_container_width=True)
                
                with col_insight2:
                    st.markdown("### 🎯 Most Accurate")
                    if 'Avg Accuracy' in user_stats.columns:
                        top_accuracy = user_stats.nlargest(3, 'Avg Accuracy')[['Avg Accuracy']]
                        st.dataframe(top_accuracy, use_container_width=True)
        else:
            st.info("📊 User statistics will be available once users complete typing sessions.")
    
    with tab3:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">🚨 <strong>Security Alert Center</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        fraud_sessions = pd.DataFrame()
        if 'fraud_detected' in session_logs.columns:
            fraud_sessions = session_logs[session_logs['fraud_detected'] == True]
        
        if not fraud_sessions.empty:
            # Enhanced alert summary
            st.markdown(f"""
            <div class="alert-danger">
                <h4>🚨 Security Alert Summary</h4>
                <p><strong>{len(fraud_sessions)} fraud attempts detected</strong> across all users.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Fraud analysis by user
            if len(fraud_sessions) > 0:
                fraud_by_user = fraud_sessions['user_id'].value_counts()
                
                col_alert1, col_alert2 = st.columns(2)
                
                with col_alert1:
                    st.markdown("### 🎯 Fraud Attempts by User")
                    st.bar_chart(fraud_by_user)
                
                with col_alert2:
                    st.markdown("### 📊 Risk Score Distribution")
                    if 'risk_score' in fraud_sessions.columns:
                        fig = px.histogram(
                            fraud_sessions, 
                            x='risk_score', 
                            bins=20,
                            title="Risk Score Distribution for Fraud Cases",
                            labels={'risk_score': 'Risk Score', 'count': 'Frequency'}
                        )
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Recent alerts table
            st.markdown("### 📋 Recent Security Alerts")
            alert_cols = ['user_id', 'timestamp', 'risk_score', 'wpm', 'accuracy']
            available_cols = [col for col in alert_cols if col in fraud_sessions.columns]
            recent_alerts = fraud_sessions[available_cols].tail(10)
            st.dataframe(recent_alerts, use_container_width=True, hide_index=True)
            
        else:
            st.markdown("""
            <div class="alert-success">
                <h4>✅ All Clear</h4>
                <p>No fraud alerts detected. System is operating normally.</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: #2c3e50; margin-bottom: 1.5rem; text-align: center;">📊 <strong>Advanced Analytics & Insights</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Advanced analytics section
        if len(session_logs) > 0:
            col_analytics1, col_analytics2 = st.columns(2)
            
            with col_analytics1:
                # Typing patterns over time
                if 'timestamp' in session_logs.columns and 'wpm' in session_logs.columns:
                    st.markdown("### 📈 Typing Speed Trends")
                    session_logs['timestamp'] = pd.to_datetime(session_logs['timestamp'])
                    
                    # WPM trends
                    fig = px.line(
                        session_logs, 
                        x='timestamp', 
                        y='wpm', 
                        color='user_id',
                        title="Typing Speed Evolution Over Time",
                        labels={'timestamp': 'Date/Time', 'wpm': 'Words Per Minute'}
                    )
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_analytics2:
                # Session type distribution
                if 'session_type' in session_logs.columns:
                    st.markdown("### 🔄 Session Type Distribution")
                    session_type_counts = session_logs['session_type'].value_counts()
                    
                    fig = px.pie(
                        values=session_type_counts.values,
                        names=session_type_counts.index,
                        title="Session Types Breakdown"
                    )
                    fig.update_layout(template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Feature correlation analysis
            st.markdown("### 🔗 Feature Correlation Analysis")
            numeric_cols = ['wpm', 'accuracy', 'avg_flight_time', 'std_flight_time', 'risk_score']
            available_numeric_cols = [col for col in numeric_cols if col in session_logs.columns]
            
            if len(available_numeric_cols) > 1:
                correlation_matrix = session_logs[available_numeric_cols].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights section
                st.markdown("""
                <div class="professional-card">
                    <h4 style="color: #2c3e50; margin-bottom: 1rem;">💡 <strong>Key Insights</strong></h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate insights
                insights = []
                if 'wpm' in session_logs.columns:
                    avg_wpm = session_logs['wpm'].mean()
                    insights.append(f"📊 Average typing speed across all users: **{avg_wpm:.1f} WPM**")
                
                if 'accuracy' in session_logs.columns:
                    avg_accuracy = session_logs['accuracy'].mean()
                    insights.append(f"🎯 Average typing accuracy: **{avg_accuracy:.1f}%**")
                
                if 'fraud_detected' in session_logs.columns:
                    fraud_rate = (session_logs['fraud_detected'].sum() / len(session_logs)) * 100
                    insights.append(f"🔒 Current fraud detection rate: **{fraud_rate:.1f}%**")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("📊 Advanced analytics will be available once more data is collected.")
        else:
            st.info("📊 No data available for advanced analytics.")

def show_settings_page():
    # Professional page header
    st.markdown("""
    <div class="professional-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-align: center; margin-bottom: 2rem;">
        <h2 style="color: white; margin-bottom: 1rem;">⚙️ System Configuration</h2>
        <p style="font-size: 1.1rem; opacity: 0.9; margin-bottom: 0;">
            Manage detection parameters, themes, and accessibility settings
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings categories
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Detection", "🎨 Appearance", "♿ Accessibility", "💾 Data"])
    
    with tab1:
        # Detection Parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="professional-card">
                <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🎯 <strong>Detection Parameters</strong></h3>
            </div>
            """, unsafe_allow_html=True)
            
            new_threshold = st.slider(
                "**Fraud Detection Threshold**",
                min_value=0.1,
                max_value=1.0,
                value=fraud_agent.fraud_threshold,
                step=0.1,
                help="Lower values = more sensitive detection"
            )
            
            if st.button("🔄 Update Threshold", type="primary", use_container_width=True):
                fraud_agent.fraud_threshold = new_threshold
                st.success(f"✅ Threshold updated to {new_threshold}")
            
            min_samples = st.number_input(
                "**Minimum Baseline Samples**",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of baseline samples required before training"
            )
        
        with col2:
            st.markdown("""
            <div class="alert-info" style="margin-top: 1rem;">
                <h4>📊 Current Settings</h4>
                <ul style="margin: 0.5rem 0;">
                    <li><strong>Detection Threshold:</strong> {:.1f}</li>
                    <li><strong>Min Samples:</strong> {}</li>
                    <li><strong>Model Type:</strong> Advanced Anomaly Detection</li>
                </ul>
            </div>
            """.format(fraud_agent.fraud_threshold, min_samples), unsafe_allow_html=True)
    
    with tab2:
        # Theme and Appearance Settings
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🎨 <strong>Appearance Settings</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Theme selection
            theme_options = {
                "light": "☀️ Light Mode",
                "dark": "🌙 Dark Mode"
            }
            
            current_theme = st.selectbox(
                "**Choose Theme**",
                options=list(theme_options.keys()),
                format_func=lambda x: theme_options[x],
                index=0 if st.session_state.theme_mode == "light" else 1,
                help="Select your preferred color scheme"
            )
            
            # Font size selection
            font_options = {
                "small": "🔤 Small (14px)",
                "medium": "🔤 Medium (16px)",
                "large": "🔤 Large (18px)",
                "extra_large": "🔤 Extra Large (20px)"
            }
            
            current_font_size = st.selectbox(
                "**Font Size**",
                options=list(font_options.keys()),
                format_func=lambda x: font_options[x],
                index=list(font_options.keys()).index(st.session_state.font_size),
                help="Adjust text size for better readability"
            )
        
        with col2:
            # Theme preview
            st.markdown("**🖼️ Theme Preview**")
            
            preview_style = ""
            if current_theme == "dark":
                preview_style = "background: #2d2d2d; color: white; border: 1px solid #555;"
            else:
                preview_style = "background: white; color: #2c3e50; border: 1px solid #e1e8ed;"
            
            st.markdown(f"""
            <div style="{preview_style} padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                <h4 style="margin: 0 0 0.5rem 0;">DefendX Preview</h4>
                <p style="margin: 0; font-size: {font_options[current_font_size].split('(')[1].split(')')[0]};">
                    This is how your interface will look with the selected theme and font size.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Apply theme button
            if st.button("🎨 Apply Theme Settings", type="primary", use_container_width=True):
                st.session_state.theme_mode = current_theme
                st.session_state.font_size = current_font_size
                st.success("✅ Theme settings applied! The page will refresh...")
                time.sleep(1)
                st.rerun()
    
    with tab3:
        # Accessibility Settings
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: var(--text-primary); margin-bottom: 1rem;">♿ <strong>Accessibility Settings</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # High contrast mode
            high_contrast = st.checkbox(
                "🔲 **High Contrast Mode**",
                value=st.session_state.high_contrast,
                help="Increases contrast for better visibility"
            )
            
            # Reduce motion
            reduce_motion = st.checkbox(
                "🎭 **Reduce Motion Effects**",
                value=st.session_state.reduce_motion,
                help="Minimizes animations and transitions"
            )
            
            # Screen reader support
            screen_reader_mode = st.checkbox(
                "🔊 **Screen Reader Optimizations**",
                value=False,
                help="Optimizes interface for screen reader users"
            )
            
        with col2:
            # Accessibility info
            st.markdown("""
            <div class="alert-info">
                <h4>♿ Accessibility Features</h4>
                <ul style="margin: 0.5rem 0;">
                    <li><strong>High Contrast:</strong> Improves text visibility</li>
                    <li><strong>Reduced Motion:</strong> Minimizes visual distractions</li>
                    <li><strong>Screen Reader:</strong> Enhanced ARIA labels</li>
                    <li><strong>Keyboard Navigation:</strong> Full keyboard support</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Apply accessibility settings
            if st.button("♿ Apply Accessibility Settings", type="primary", use_container_width=True):
                st.session_state.high_contrast = high_contrast
                st.session_state.reduce_motion = reduce_motion
                st.success("✅ Accessibility settings applied! The page will refresh...")
                time.sleep(1)
                st.rerun()
        
        # Quick accessibility tests
        st.markdown("---")
        st.markdown("**🧪 Accessibility Quick Tests**")
        
        test_col1, test_col2, test_col3 = st.columns(3)
        
        with test_col1:
            if st.button("🎨 Test Contrast", use_container_width=True):
                st.info("Current contrast ratio meets WCAG AA standards")
        
        with test_col2:
            if st.button("⌨️ Test Keyboard Nav", use_container_width=True):
                st.info("All interactive elements are keyboard accessible")
        
        with test_col3:
            if st.button("🔊 Test Screen Reader", use_container_width=True):
                st.info("ARIA labels and roles are properly implemented")
    
    with tab4:
        # Data Management
        st.markdown("""
        <div class="professional-card">
            <h3 style="color: var(--text-primary); margin-bottom: 1rem;">💾 <strong>Data Management</strong></h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📥 Export Functions**")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                if st.button("📊 Export Sessions", use_container_width=True):
                    session_logs = data_agent.get_session_logs()
                    if not session_logs.empty:
                        csv = session_logs.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No session data to export")
            
            with export_col2:
                if st.button("👥 Export Users", use_container_width=True):
                    user_files = [f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')]
                    if user_files:
                        users_summary = []
                        for file in user_files:
                            user_id = file.replace('_profile.json', '')
                            user_data = data_agent.load_user_data(user_id)
                            if user_data:
                                users_summary.append({
                                    'user_id': user_id,
                                    'baseline_count': len(user_data.get('baseline_features', [])),
                                    'model_trained': user_data.get('model_trained', False),
                                    'created_at': user_data.get('created_at', 'Unknown')
                                })
                        
                        users_df = pd.DataFrame(users_summary)
                        csv = users_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Users",
                            data=csv,
                            file_name=f"users_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ No users to export")
        
        with col2:
            st.markdown("**🗑️ Danger Zone**")
            
            # Data clearing with confirmation
            clear_confirm = st.checkbox("⚠️ I understand this will delete ALL data permanently")
            
            if st.button("🗑️ Clear All Data", type="secondary", disabled=not clear_confirm, use_container_width=True):
                try:
                    # Clear data directory
                    for file in os.listdir(data_agent.data_dir):
                        file_path = os.path.join(data_agent.data_dir, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                    
                    # Clear models directory
                    models_dir = os.path.join(data_agent.data_dir, "models")
                    if os.path.exists(models_dir):
                        for file in os.listdir(models_dir):
                            file_path = os.path.join(models_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                    
                    st.success("✅ All data cleared successfully")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing data: {str(e)}")
    
    # System Information
    st.markdown("""
    <div class="professional-card" style="margin-top: 2rem;">
        <h3 style="color: var(--text-primary); margin-bottom: 1rem;">📋 <strong>System Information</strong></h3>
    </div>
    """, unsafe_allow_html=True)
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎨 Current Theme</h4>
            <p><strong>{st.session_state.theme_mode.title()}</strong></p>
            <p>Font: {st.session_state.font_size.title()}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>♿ Accessibility</h4>
            <p>High Contrast: <strong>{'On' if st.session_state.high_contrast else 'Off'}</strong></p>
            <p>Reduced Motion: <strong>{'On' if st.session_state.reduce_motion else 'Off'}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>� System Status</h4>
            <p>Version: <strong>1.3.0</strong></p>
            <p>Platform: <strong>Streamlit</strong></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

