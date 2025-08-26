"""
Configuration and constants for the fraud detection system
"""

import os
from typing import Dict, List

# Application Configuration
APP_CONFIG = {
    'title': 'On-Device Multi-Agent Fraud Detection System',
    'page_icon': '🔐',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Data Configuration
DATA_CONFIG = {
    'data_directory': 'data',
    'models_directory': 'models',
    'logs_directory': 'logs',
    'exports_directory': 'exports'
}

# Model Configuration
MODEL_CONFIG = {
    'min_baseline_samples': 3,
    'max_baseline_samples': 10,
    'contamination_rate': 0.1,
    'anomaly_threshold': 0.5,
    'fraud_threshold': 0.3,
    'adaptation_rate': 0.1,
    'ensemble_weights': {
        'isolation_forest': 0.4,
        'one_class_svm': 0.4,
        'dbscan': 0.2
    }
}

# Feature Configuration
FEATURE_CONFIG = {
    'core_features': [
        'wpm',
        'accuracy',
        'avg_flight_time',
        'std_flight_time',
        'total_time',
        'char_count',
        'word_count',
        'typing_rhythm_variance',
        'max_flight_time',
        'min_flight_time'
    ],
    'enhanced_features': [
        'avg_hold_time',
        'avg_gap_time',
        'avg_flight_enhanced',
        'rhythm_stddev',
        'typed_chars',
        'error_rate',
        'hold_time_variance',
        'gap_time_variance'
    ],
    'advanced_features': [
        'ngram_2_mean',
        'ngram_2_std',
        'pressure_variance',
        'pressure_peaks',
        'pressure_consistency'
    ],
    'realtime_features': [
        'realtime_wpm',
        'rhythm_consistency',
        'avg_recent_flight',
        'flight_variance'
    ]
}

# Risk Assessment Thresholds
RISK_THRESHOLDS = {
    'wpm': {
        'min_normal': 10,
        'max_normal': 120,
        'penalty_weight': 0.3
    },
    'accuracy': {
        'min_normal': 70,
        'penalty_weight': 0.2
    },
    'rhythm_variance': {
        'max_normal': 0.5,
        'penalty_weight': 0.25
    },
    'behavioral_anomaly': {
        'penalty_weight': 1.0
    }
}

# Reference Texts for Training and Testing
REFERENCE_TEXTS = {
    'training': [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "Waltz, bad nymph, for quick jigs vex.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "Five quacking zephyrs jolt my wax bed.",
        "The five boxing wizards jump quickly.",
        "Jackdaws love my big sphinx of quartz.",
        "Mr. Jock, TV quiz PhD., bags few lynx."
    ],
    'verification': [
        "Authentication test sentence for user verification.",
        "Security check: please type this sentence carefully.",
        "Behavioral biometric verification in progress.",
        "Your typing pattern is your digital fingerprint.",
        "Continuous authentication through keystroke dynamics.",
        "Fraud detection system monitoring user behavior.",
        "Real-time analysis of typing characteristics.",
        "Biometric security through behavioral patterns."
    ],
    'challenge': [
        "Advanced security protocol requires precise typing.",
        "Multi-factor authentication using behavioral biometrics.",
        "Sophisticated fraud prevention through pattern analysis.",
        "Continuous monitoring ensures account security."
    ]
}

# UI Configuration
UI_CONFIG = {
    'colors': {
        'primary': '#1f77b4',
        'success': '#28a745',
        'warning': '#ffc107',
        'danger': '#dc3545',
        'info': '#17a2b8'
    },
    'alert_styles': {
        'success': {
            'background': '#d4edda',
            'color': '#155724',
            'border': '#c3e6cb'
        },
        'danger': {
            'background': '#f8d7da',
            'color': '#721c24',
            'border': '#f5c6cb'
        },
        'warning': {
            'background': '#fff3cd',
            'color': '#856404',
            'border': '#ffeaa7'
        }
    },
    'metrics': {
        'refresh_interval': 1000,  # milliseconds
        'chart_height': 400,
        'max_points_display': 100
    }
}

# Security Configuration
SECURITY_CONFIG = {
    'hash_algorithm': 'sha256',
    'anonymization_enabled': True,
    'data_retention_days': 90,
    'max_failed_attempts': 5,
    'lockout_duration_minutes': 15,
    'encryption_enabled': False  # Can be enabled for production
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'max_concurrent_users': 100,
    'session_timeout_minutes': 30,
    'cache_size_mb': 50,
    'max_log_file_size_mb': 10,
    'realtime_buffer_size': 50,
    'analysis_interval_ms': 100
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_rotation': True,
    'max_file_size_mb': 10,
    'backup_count': 5
}

# Export Configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'json', 'excel'],
    'include_sensitive_data': False,
    'anonymize_exports': True,
    'max_export_records': 10000
}

# Validation Rules
VALIDATION_RULES = {
    'user_id': {
        'min_length': 3,
        'max_length': 50,
        'allowed_chars': 'alphanumeric_underscore'
    },
    'typing_session': {
        'min_characters': 10,
        'max_characters': 1000,
        'min_duration_seconds': 5,
        'max_duration_seconds': 300
    },
    'features': {
        'wpm_range': (0, 300),
        'accuracy_range': (0, 100),
        'flight_time_range': (0, 5.0)
    }
}

# Agent Configuration
AGENT_CONFIG = {
    'keystroke_agent': {
        'capture_precision_ms': 10,
        'feature_extraction_methods': ['basic', 'advanced'],
        'real_time_analysis': True
    },
    'behavior_agent': {
        'model_types': ['isolation_forest', 'one_class_svm', 'ensemble'],
        'auto_retrain': True,
        'retrain_threshold': 50  # sessions
    },
    'fraud_agent': {
        'alert_types': ['low', 'medium', 'high', 'critical'],
        'notification_methods': ['ui', 'log', 'export'],
        'escalation_rules': True
    },
    'data_agent': {
        'persistence_method': 'file_system',
        'backup_enabled': True,
        'compression_enabled': False
    }
}

# Feature Engineering Configuration
FEATURE_ENGINEERING = {
    'normalization_methods': ['standard', 'minmax', 'robust'],
    'dimensionality_reduction': ['pca', 'lda'],
    'feature_selection': ['variance', 'correlation', 'mutual_info'],
    'outlier_detection': ['iqr', 'zscore', 'isolation_forest']
}

# Model Training Configuration
TRAINING_CONFIG = {
    'cross_validation_folds': 5,
    'train_test_split': 0.8,
    'random_state': 42,
    'hyperparameter_tuning': {
        'method': 'grid_search',
        'cv_folds': 3,
        'scoring': 'f1'
    },
    'ensemble_methods': ['voting', 'stacking', 'boosting']
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'health_check_interval_minutes': 5,
    'performance_metrics': [
        'response_time',
        'memory_usage',
        'cpu_usage',
        'accuracy',
        'false_positive_rate'
    ],
    'alerting_thresholds': {
        'response_time_ms': 1000,
        'memory_usage_percent': 80,
        'cpu_usage_percent': 70,
        'accuracy_drop_percent': 10
    }
}

# API Configuration (for future expansion)
API_CONFIG = {
    'enabled': False,
    'host': '127.0.0.1',
    'port': 8000,
    'authentication': 'api_key',
    'rate_limiting': {
        'requests_per_minute': 60,
        'requests_per_hour': 1000
    },
    'versioning': 'v1'
}

# Development and Testing Configuration
DEV_CONFIG = {
    'debug_mode': False,
    'mock_data_enabled': False,
    'test_users': ['test_user_1', 'test_user_2', 'test_user_3'],
    'performance_profiling': False,
    'verbose_logging': False
}

def get_config(config_name: str) -> Dict:
    """Get configuration by name"""
    config_map = {
        'app': APP_CONFIG,
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'feature': FEATURE_CONFIG,
        'risk': RISK_THRESHOLDS,
        'ui': UI_CONFIG,
        'security': SECURITY_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'logging': LOGGING_CONFIG,
        'export': EXPORT_CONFIG,
        'validation': VALIDATION_RULES,
        'agent': AGENT_CONFIG,
        'training': TRAINING_CONFIG,
        'monitoring': MONITORING_CONFIG,
        'api': API_CONFIG,
        'dev': DEV_CONFIG
    }
    
    return config_map.get(config_name, {})

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check data directories
    for directory in DATA_CONFIG.values():
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
    
    # Validate thresholds
    if MODEL_CONFIG['fraud_threshold'] <= 0 or MODEL_CONFIG['fraud_threshold'] >= 1:
        errors.append("Fraud threshold must be between 0 and 1")
    
    # Validate feature configuration
    required_features = FEATURE_CONFIG['core_features']
    if len(required_features) < 3:
        errors.append("At least 3 core features required")
    
    return errors

# Initialize directories on import
try:
    for directory in DATA_CONFIG.values():
        os.makedirs(directory, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create directories: {e}")

# Environment-specific overrides
if os.getenv('ENVIRONMENT') == 'production':
    MODEL_CONFIG['min_baseline_samples'] = 5
    SECURITY_CONFIG['anonymization_enabled'] = True
    DEV_CONFIG['debug_mode'] = False
elif os.getenv('ENVIRONMENT') == 'development':
    DEV_CONFIG['debug_mode'] = True
    DEV_CONFIG['verbose_logging'] = True
