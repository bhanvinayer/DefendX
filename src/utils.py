"""
Utility functions and helper classes for the fraud detection system
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import hashlib
from datetime import datetime, timedelta

class FeatureExtractor:
    """Advanced feature extraction utilities"""
    
    @staticmethod
    def extract_ngram_features(key_events: List[Dict], n: int = 2) -> Dict:
        """Extract n-gram timing features"""
        if len(key_events) < n:
            return {}
        
        ngram_times = []
        for i in range(len(key_events) - n + 1):
            ngram_time = sum(event['flight_time'] for event in key_events[i:i+n])
            ngram_times.append(ngram_time)
        
        return {
            f'ngram_{n}_mean': np.mean(ngram_times),
            f'ngram_{n}_std': np.std(ngram_times),
            f'ngram_{n}_min': np.min(ngram_times),
            f'ngram_{n}_max': np.max(ngram_times)
        }
    
    @staticmethod
    def extract_pressure_simulation(key_events: List[Dict]) -> Dict:
        """Simulate typing pressure from timing data"""
        if not key_events:
            return {}
        
        # Simulate pressure based on flight time variations
        flight_times = [event['flight_time'] for event in key_events[1:]]
        if not flight_times:
            return {}
        
        pressure_variance = np.var(flight_times)
        pressure_peaks = len([t for t in flight_times if t > np.mean(flight_times) + np.std(flight_times)])
        
        return {
            'pressure_variance': pressure_variance,
            'pressure_peaks': pressure_peaks,
            'pressure_consistency': 1.0 / (1.0 + pressure_variance)
        }

class SecurityUtils:
    """Security and privacy utilities"""
    
    @staticmethod
    def hash_user_id(user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    @staticmethod
    def anonymize_features(features: Dict) -> Dict:
        """Remove or hash sensitive features"""
        anonymized = features.copy()
        
        # Remove exact timing data that could be used for reconstruction
        sensitive_keys = ['exact_timestamps', 'raw_key_sequence']
        for key in sensitive_keys:
            anonymized.pop(key, None)
        
        return anonymized

class ModelUtils:
    """Machine learning utilities"""
    
    @staticmethod
    def calculate_feature_importance(model, feature_names: List[str]) -> Dict:
        """Calculate feature importance for anomaly detection models"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                # For models without built-in feature importance
                importance = np.ones(len(feature_names)) / len(feature_names)
            
            return dict(zip(feature_names, importance))
        except:
            return {}
    
    @staticmethod
    def adaptive_threshold(historical_scores: List[float], percentile: float = 95) -> float:
        """Calculate adaptive threshold based on historical data"""
        if not historical_scores:
            return 0.5
        
        return np.percentile(historical_scores, percentile)

class DataValidation:
    """Data validation and quality checks"""
    
    @staticmethod
    def validate_typing_session(features: Dict) -> Tuple[bool, List[str]]:
        """Validate typing session data quality"""
        errors = []
        
        # Check for required features
        required_features = ['wpm', 'accuracy', 'avg_flight_time']
        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")
        
        # Check for reasonable values
        if features.get('wpm', 0) < 0 or features.get('wpm', 0) > 300:
            errors.append("WPM value out of reasonable range (0-300)")
        
        if features.get('accuracy', 0) < 0 or features.get('accuracy', 0) > 100:
            errors.append("Accuracy value out of range (0-100)")
        
        if features.get('avg_flight_time', 0) < 0:
            errors.append("Negative flight time detected")
        
        return len(errors) == 0, errors

class ExportUtils:
    """Data export and reporting utilities"""
    
    @staticmethod
    def generate_user_report(user_id: str, sessions: List[Dict]) -> Dict:
        """Generate comprehensive user behavior report"""
        if not sessions:
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(sessions)
        
        report = {
            'user_id': user_id,
            'report_generated': datetime.now().isoformat(),
            'total_sessions': len(sessions),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df else None,
                'end': df['timestamp'].max() if 'timestamp' in df else None
            },
            'statistics': {
                'avg_wpm': df['wpm'].mean() if 'wpm' in df else 0,
                'avg_accuracy': df['accuracy'].mean() if 'accuracy' in df else 0,
                'fraud_incidents': df['fraud_detected'].sum() if 'fraud_detected' in df else 0
            },
            'trends': ExportUtils._calculate_trends(df),
            'risk_profile': ExportUtils._assess_risk_profile(df)
        }
        
        return report
    
    @staticmethod
    def _calculate_trends(df: pd.DataFrame) -> Dict:
        """Calculate behavioral trends"""
        if len(df) < 2:
            return {}
        
        trends = {}
        numeric_cols = ['wpm', 'accuracy', 'avg_flight_time']
        
        for col in numeric_cols:
            if col in df.columns:
                # Simple linear trend
                x = np.arange(len(df))
                y = df[col].values
                trend = np.polyfit(x, y, 1)[0]  # Slope
                trends[f'{col}_trend'] = trend
        
        return trends
    
    @staticmethod
    def _assess_risk_profile(df: pd.DataFrame) -> str:
        """Assess overall risk profile"""
        if 'fraud_detected' not in df.columns:
            return "Unknown"
        
        fraud_rate = df['fraud_detected'].mean()
        
        if fraud_rate == 0:
            return "Low Risk"
        elif fraud_rate < 0.1:
            return "Medium Risk"
        else:
            return "High Risk"

class VisualizationHelpers:
    """Helper functions for creating visualizations"""
    
    @staticmethod
    def prepare_timeline_data(sessions: List[Dict]) -> pd.DataFrame:
        """Prepare data for timeline visualizations"""
        df = pd.DataFrame(sessions)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        return df
    
    @staticmethod
    def create_risk_heatmap_data(sessions: List[Dict]) -> np.ndarray:
        """Create data for risk score heatmap"""
        df = pd.DataFrame(sessions)
        if 'risk_score' not in df.columns:
            return np.array([[0]])
        
        # Create hourly risk heatmap
        df = VisualizationHelpers.prepare_timeline_data(sessions)
        heatmap_data = df.pivot_table(
            values='risk_score',
            index='day_of_week',
            columns='hour',
            aggfunc='mean',
            fill_value=0
        )
        
        return heatmap_data.values

class SystemMonitor:
    """System monitoring and health checks"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.session_count = 0
        self.error_count = 0
    
    def log_session(self):
        """Log a new session"""
        self.session_count += 1
    
    def log_error(self):
        """Log an error"""
        self.error_count += 1
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        uptime = datetime.now() - self.start_time
        
        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime),
            'total_sessions': self.session_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.session_count, 1),
            'sessions_per_hour': self.session_count / max(uptime.total_seconds() / 3600, 1),
            'status': 'healthy' if self.error_count < self.session_count * 0.1 else 'degraded'
        }

# Global system monitor instance
system_monitor = SystemMonitor()
