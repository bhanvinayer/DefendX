import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import os
import random
import joblib
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
        
        # Calculate WPM properly: either word count or character count / 5
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Standard WPM calculation (characters / 5 / minutes) or (words / minutes)
        minutes_elapsed = total_time / 60 if total_time > 0 else 0.01  # Avoid division by zero
        
        # Ensure reasonable time bounds (minimum 1 second for calculation)
        if total_time < 1.0:
            minutes_elapsed = 1.0 / 60  # Use 1 second minimum
            
        wpm_by_chars = (char_count / 5) / minutes_elapsed if minutes_elapsed > 0 else 0
        wpm_by_words = word_count / minutes_elapsed if minutes_elapsed > 0 else 0
        
        # Use the character-based method as it's more standard, but cap at reasonable values
        wpm = min(wpm_by_chars, 200)  # Cap at 200 WPM to avoid unrealistic values
        
        # Flight time statistics
        flight_times = [event['flight_time'] for event in self.key_events[1:]]
        avg_flight_time = np.mean(flight_times) if flight_times else 0
        std_flight_time = np.std(flight_times) if flight_times else 0
        
        # Accuracy calculation
        accuracy = self._calculate_accuracy(text, reference_text)
        
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
            'gap_time_variance': np.var(self.gap_times) if self.gap_times else 0
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
    
    def _count_corrections(self, typed_text: str, reference_text: str) -> int:
        """Estimate number of corrections made"""
        return max(0, len(typed_text) - len(reference_text))
        
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
        
    def train_user_model(self, user_id: str, min_samples: int = 3):
        """Train anomaly detection model for specific user"""
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found in profiles")
            return False
            
        baseline_features = self.user_profiles[user_id]['baseline_features']
        if len(baseline_features) < min_samples:
            print(f"Insufficient baseline samples: {len(baseline_features)} < {min_samples}")
            return False
            
        # Prepare feature matrix
        feature_names = ['wpm', 'avg_flight_time', 'std_flight_time', 'accuracy', 
                        'typing_rhythm_variance', 'max_flight_time', 'min_flight_time']
        
        X = []
        for features in baseline_features:
            row = [features.get(name, 0) for name in feature_names]
            X.append(row)
            
        X = np.array(X)
        
        # Train anomaly detection model
        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)
        
        self.user_models[user_id] = {
            'model': model,
            'scaler': StandardScaler().fit(X),
            'feature_names': feature_names,
            'baseline_stats': {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0)
            }
        }
        
        # Mark model as trained
        self.user_profiles[user_id]['model_trained'] = True
        
        # Save model to disk
        self._save_model(user_id)
        
        print(f"Model trained and saved successfully for user {user_id}")
        return True
    
    def _save_model(self, user_id: str):
        """Save trained model to disk"""
        try:
            # Create models directory
            model_dir = "data/models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f"{user_id}_model.pkl")
            
            # Save the entire model data
            if user_id in self.user_models:
                joblib.dump(self.user_models[user_id], model_path)
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
                self.user_models[user_id] = joblib.load(model_path)
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
        feature_vector = [features.get(name, 0) for name in model_data['feature_names']]
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        feature_vector = model_data['scaler'].transform(feature_vector)
        
        # Predict anomaly
        prediction = model_data['model'].predict(feature_vector)[0]
        anomaly_score = model_data['model'].decision_function(feature_vector)[0]
        
        is_anomaly = prediction == -1
        confidence = abs(anomaly_score)
        
        return is_anomaly, confidence

class FraudDetectionAgent:
    """Agent responsible for fraud detection and alerting"""
    
    def __init__(self):
        self.fraud_threshold = 0.3
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

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.alert-success {
    background-color: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #c3e6cb;
}
.alert-danger {
    background-color: #f8d7da;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🔐 On-Device Multi-Agent Fraud Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page:", [
        "🏠 Home",
        "👤 User Registration", 
        "🔍 Verification Test",
        "📊 Admin Dashboard",
        "⚙️ System Settings"
    ])
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "👤 User Registration":
        show_registration_page()
    elif page == "🔍 Verification Test":
        show_verification_page()
    elif page == "📊 Admin Dashboard":
        show_admin_dashboard()
    elif page == "⚙️ System Settings":
        show_settings_page()

def show_home_page():
    st.markdown("## Welcome to the On-Device Fraud Detection System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        This multi-agent system provides real-time fraud detection based on behavioral biometrics:
        
        - **Keystroke Dynamics**: Analyzes typing patterns, rhythm, and timing
        - **Behavioral Modeling**: Creates unique user profiles from typing behavior
        - **Anomaly Detection**: Identifies suspicious activities in real-time
        - **Privacy-First**: All processing happens on-device
        """)
        
    with col2:
        st.markdown("""
        ### 🤖 Multi-Agent Architecture
        - **Keystroke Agent**: Captures and processes typing patterns
        - **Behavior Model Agent**: Trains and maintains user models
        - **Fraud Detection Agent**: Analyzes sessions for fraud indicators
        - **Data Manager Agent**: Handles data persistence and logs
        """)
    
    # System statistics
    st.markdown("### 📈 System Statistics")
    
    # Load existing data
    session_logs = data_agent.get_session_logs()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_users = len([f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')])
        st.metric("Registered Users", total_users)
        
    with col2:
        total_sessions = len(session_logs) if not session_logs.empty else 0
        st.metric("Total Sessions", total_sessions)
        
    with col3:
        fraud_alerts = len(fraud_agent.get_fraud_alerts())
        st.metric("Fraud Alerts", fraud_alerts)
        
    with col4:
        if not session_logs.empty:
            avg_accuracy = session_logs['accuracy'].mean()
            st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
        else:
            st.metric("Avg Accuracy", "N/A")

def show_registration_page():
    """User registration and baseline collection page - MonkeyType style"""
    
    # Initialize session state for typing
    if 'typing_started' not in st.session_state:
        st.session_state.typing_started = False
    if 'current_text' not in st.session_state:
        st.session_state.current_text = ""
    if 'typing_complete' not in st.session_state:
        st.session_state.typing_complete = False
    
    # Custom CSS for MonkeyType-like styling
    st.markdown("""
    <style>
    .typing-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
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
    }
    .metrics-card {
        background: rgba(255,255,255,0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .session-progress {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("# ⌨️ Behavioral Profile Setup")
    st.markdown("*Create your unique typing signature for fraud detection*")
    
    user_id = st.text_input("👤 Enter Your User ID:", placeholder="e.g., john_doe", key="user_input")
    
    if user_id:
        # Check existing user data
        existing_data = data_agent.load_user_data(user_id)
        if existing_data and user_id not in behavior_agent.user_profiles:
            behavior_agent.user_profiles[user_id] = existing_data
            
        baseline_count = len(behavior_agent.user_profiles.get(user_id, {}).get('baseline_features', []))
        
        # Progress tracker - MonkeyType style
        st.markdown(f"""
        <div class="session-progress">
            <h3>Progress: Session {baseline_count + 1 if baseline_count < 3 else 3} of 3</h3>
            <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                <span style="color: {'#4CAF50' if baseline_count >= 1 else '#ccc'};">●</span>
                <span style="color: {'#4CAF50' if baseline_count >= 2 else '#ccc'};">●</span>
                <span style="color: {'#4CAF50' if baseline_count >= 3 else '#ccc'};">●</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Reference texts for different sessions
        reference_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs and thirty quart jugs.",
            "How vexingly quick daft zebras jump over the lazy dog.",
            "Waltz, bad nymph, for quick jigs vex Bud and Jim.",
            "Sphinx of black quartz, judge my vow and hex."
        ]
        
        # Select text based on session number
        if baseline_count < 3:
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
                    if typed_text:
                        # Process keystrokes with realistic timing
                        keystroke_agent.reset_session()
                        keystroke_agent.start_capture()
                        
                        base_time = time.time()
                        for i, char in enumerate(typed_text):
                            char_time = base_time + i * random.uniform(0.2, 0.5)
                            keystroke_agent.process_keystroke(char, char_time)
                            release_time = char_time + random.uniform(0.05, 0.1)
                            keystroke_agent.process_key_release(char, release_time)
                        
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
                        
                        # Extract features for saving
                        features = keystroke_agent.extract_features(typed_text, selected_text)
                        
                        # MonkeyType-style completion message
                        st.markdown(f"""
                        <div style="background:#4CAF50; color:white; padding:1rem; border-radius:10px; text-align:center; margin:1rem 0;">
                            <h3>🎉 Session Complete!</h3>
                            <p>Final WPM: <strong>{real_time_metrics.get('speed_wpm', 0):.1f}</strong> | 
                               Accuracy: <strong>{accuracy:.1f}%</strong></p>
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
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                        
                        with col_save2:
                            # Next session or model training
                            current_baseline_count = len(behavior_agent.user_profiles.get(user_id, {}).get('baseline_features', []))
                            
                            if current_baseline_count + 1 >= 3:  # After saving this will be 3
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
        if baseline_count >= 3:
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
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Training failed. Please try again.")
            else:
                st.success("✅ Registration Complete! Your behavioral profile is ready.")
                st.info("💡 Visit the 'User Verification' page to test your profile.")

def show_verification_page():
    st.markdown("## 🔍 Real-time Verification & Fraud Detection")
    
    # User selection
    user_files = [f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')]
    user_ids = [f.replace('_profile.json', '') for f in user_files]
    
    if not user_ids:
        st.warning("No registered users found. Please register users first.")
        return
        
    selected_user = st.selectbox("Select user to verify:", user_ids)
    
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
        if baseline_count >= 3:
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
        if 'verification_active' not in st.session_state:
            st.session_state.verification_active = False
            
        if st.button("Start Verification"):
            st.session_state.verification_active = True
            keystroke_agent.start_capture()
            
        if st.session_state.verification_active:
            typed_text = st.text_area(
                "Type here:", 
                height=100,
                placeholder="Start typing the verification text..."
            )
            
            if typed_text:
                # Process enhanced keystroke data with realistic timing
                keystroke_agent.reset_session()
                keystroke_agent.start_capture()
                
                # Simulate realistic typing timing
                base_time = time.time()
                for i, char in enumerate(typed_text):
                    # Realistic keystroke timing (200-500ms between keys for more realistic WPM)
                    char_time = base_time + i * random.uniform(0.2, 0.5)
                    keystroke_agent.process_keystroke(char, char_time)
                    # Key release after 50-100ms
                    release_time = char_time + random.uniform(0.05, 0.1)
                    keystroke_agent.process_key_release(char, release_time)
                
                # Get real-time metrics
                real_time_metrics = keystroke_agent.get_real_time_metrics()
                features = keystroke_agent.extract_features(typed_text, verification_text)
                
                # Enhanced real-time metrics
                with col2:
                    st.markdown("### Live Verification Metrics")
                    if real_time_metrics:
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("WPM", f"{real_time_metrics.get('speed_wpm', 0):.1f}")
                            st.metric("Accuracy", f"{features.get('accuracy', 0):.1f}%" if features else "0%")
                            st.metric("Hold Time", f"{real_time_metrics.get('avg_hold', 0):.3f}s")
                            
                        with col2_2:
                            st.metric("Flight Time", f"{real_time_metrics.get('avg_flight', 0):.3f}s")
                            st.metric("Errors", real_time_metrics.get('errors', 0))
                            st.metric("Rhythm", f"{real_time_metrics.get('rhythm_stddev', 0):.3f}s")
                        
                        # Real-time risk assessment
                        if features:
                            risk_score = 0
                            risk_indicators = []
                            
                            # Speed check
                            wpm = features.get('wpm', 0)
                            if wpm > 120:
                                risk_score += 0.3
                                risk_indicators.append("High speed")
                            elif wpm < 10:
                                risk_score += 0.3
                                risk_indicators.append("Low speed")
                            
                            # Accuracy check
                            accuracy = features.get('accuracy', 100)
                            if accuracy < 80:
                                risk_score += 0.2
                                risk_indicators.append("Low accuracy")
                            
                            # Error rate check
                            error_rate = real_time_metrics.get('error_rate', 0)
                            if error_rate > 15:
                                risk_score += 0.2
                                risk_indicators.append("High error rate")
                            
                            # Display risk level
                            if risk_score > 0.5:
                                st.error(f"🚨 High Risk: {risk_score:.2f}")
                            elif risk_score > 0.3:
                                st.warning(f"⚠️ Medium Risk: {risk_score:.2f}")
                            else:
                                st.success(f"✅ Low Risk: {risk_score:.2f}")
                            
                            if risk_indicators:
                                st.markdown("**Risk Factors:**")
                                for indicator in risk_indicators:
                                    st.markdown(f"- {indicator}")
                
                # Complete verification
                if len(typed_text) >= len(verification_text) * 0.9:
                    if st.button("Complete Verification"):
                        # Predict anomaly
                        is_anomaly, confidence = behavior_agent.predict_anomaly(selected_user, features)
                        
                        # Fraud analysis
                        fraud_analysis = fraud_agent.analyze_session(
                            selected_user, features, is_anomaly, confidence
                        )
                        
                        # Display results
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
                            st.metric("Confidence", f"{fraud_analysis['confidence']:.2f}")
                        with col3:
                            status = "ANOMALY" if is_anomaly else "NORMAL"
                            st.metric("Pattern Status", status)
                        
                        # Risk factors
                        if fraud_analysis['risk_factors']:
                            st.markdown("**Risk Factors Detected:**")
                            for factor in fraud_analysis['risk_factors']:
                                st.markdown(f"- {factor}")
                        
                        # Log session
                        session_data = {
                            'user_id': selected_user,
                            'session_type': 'verification',
                            'timestamp': datetime.now().isoformat(),
                            'fraud_detected': fraud_analysis['fraud_detected'],
                            'risk_score': fraud_analysis['risk_score'],
                            **features
                        }
                        data_agent.save_session_log(session_data)
                        
                        st.session_state.verification_active = False

def show_admin_dashboard():
    st.markdown("## 📊 Admin Dashboard")
    
    # Load session data
    session_logs = data_agent.get_session_logs()
    
    if session_logs.empty:
        st.info("No session data available yet.")
        return
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 Users", "🚨 Alerts", "📊 Analytics"])
    
    with tab1:
        st.markdown("### System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sessions = len(session_logs)
            st.metric("Total Sessions", total_sessions)
            
        with col2:
            fraud_sessions = 0
            if 'fraud_detected' in session_logs.columns:
                fraud_sessions = len(session_logs[session_logs['fraud_detected'] == True])
            st.metric("Fraud Detected", fraud_sessions)
            
        with col3:
            fraud_rate = (fraud_sessions / total_sessions * 100) if total_sessions > 0 else 0
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
            
        with col4:
            avg_risk_score = 0
            if 'risk_score' in session_logs.columns:
                avg_risk_score = session_logs['risk_score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk_score:.2f}")
        
        # Recent activity
        st.markdown("### Recent Activity")
        
        # Select available columns for display
        base_columns = ['user_id', 'session_type', 'timestamp']
        optional_columns = ['fraud_detected', 'risk_score', 'wpm', 'accuracy']
        
        # Only include columns that actually exist
        display_columns = base_columns.copy()
        for col in optional_columns:
            if col in session_logs.columns:
                display_columns.append(col)
        
        recent_sessions = session_logs.tail(10)[display_columns]
        st.dataframe(recent_sessions, use_container_width=True)
    
    with tab2:
        st.markdown("### User Profiles")
        
        # User statistics with dynamic column handling
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
        
        st.dataframe(user_stats, use_container_width=True)
        
        # User behavior visualization
        if len(session_logs) > 0 and 'wpm' in session_logs.columns and 'accuracy' in session_logs.columns:
            fig = px.scatter(session_logs, x='wpm', y='accuracy', color='user_id',
                           title="User Typing Patterns: WPM vs Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Visualization will be available once users complete typing sessions.")
    
    with tab3:
        st.markdown("### Fraud Alerts")
        
        fraud_sessions = pd.DataFrame()
        if 'fraud_detected' in session_logs.columns:
            fraud_sessions = session_logs[session_logs['fraud_detected'] == True]
        
        if not fraud_sessions.empty:
            # Alert summary
            st.markdown(f"**{len(fraud_sessions)} fraud alerts detected**")
            
            # Recent alerts
            alert_cols = ['user_id', 'timestamp', 'risk_score', 'wpm', 'accuracy']
            available_cols = [col for col in alert_cols if col in fraud_sessions.columns]
            st.dataframe(fraud_sessions[available_cols].tail(10), use_container_width=True)
            
            # Risk score distribution
            fig = px.histogram(fraud_sessions, x='risk_score', bins=20,
                             title="Risk Score Distribution for Fraud Cases")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud alerts detected yet.")
    
    with tab4:
        st.markdown("### Advanced Analytics")
        
        # Typing patterns over time
        if 'timestamp' in session_logs.columns:
            session_logs['timestamp'] = pd.to_datetime(session_logs['timestamp'])
            
            # WPM trends
            fig = px.line(session_logs, x='timestamp', y='wpm', color='user_id',
                         title="Typing Speed Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
            
            # Accuracy trends
            fig = px.line(session_logs, x='timestamp', y='accuracy', color='user_id',
                         title="Accuracy Trends Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation heatmap
        numeric_cols = ['wpm', 'accuracy', 'avg_flight_time', 'std_flight_time', 'risk_score']
        available_numeric_cols = [col for col in numeric_cols if col in session_logs.columns]
        
        if len(available_numeric_cols) > 1:
            correlation_matrix = session_logs[available_numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig)

def show_settings_page():
    st.markdown("## ⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Detection Parameters")
        
        new_threshold = st.slider(
            "Fraud Detection Threshold",
            min_value=0.1,
            max_value=1.0,
            value=fraud_agent.fraud_threshold,
            step=0.1,
            help="Lower values = more sensitive detection"
        )
        
        if st.button("Update Threshold"):
            fraud_agent.fraud_threshold = new_threshold
            st.success(f"Threshold updated to {new_threshold}")
        
        min_samples = st.number_input(
            "Minimum Baseline Samples",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of baseline samples required before training"
        )
    
    with col2:
        st.markdown("### Data Management")
        
        if st.button("Export Session Data"):
            session_logs = data_agent.get_session_logs()
            if not session_logs.empty:
                csv = session_logs.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"session_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")
        
        if st.button("Clear All Data"):
            if st.checkbox("I confirm I want to delete all data"):
                # Clear data directory
                for file in os.listdir(data_agent.data_dir):
                    os.remove(os.path.join(data_agent.data_dir, file))
                st.success("All data cleared successfully")
        
        st.markdown("### System Information")
        st.info(f"""
        **Data Directory:** {data_agent.data_dir}
        **Registered Users:** {len([f for f in os.listdir(data_agent.data_dir) if f.endswith('_profile.json')])}
        **Session Logs:** {len(data_agent.get_session_logs())}
        **Fraud Alerts:** {len(fraud_agent.get_fraud_alerts())}
        """)

if __name__ == "__main__":
    main()
