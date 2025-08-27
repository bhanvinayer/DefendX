# DefendX API Reference

## Overview

This document provides comprehensive API reference for the DefendX Multi-Agent Fraud Detection System. The system is implemented as a Streamlit application with four main agent classes and supporting functions.

## Core Agent Classes

### KeystrokeAgent

Captures and analyzes keystroke dynamics to create behavioral biometric profiles.

#### Class Definition
```python
class KeystrokeAgent:
    def __init__(self)
```

#### Methods

##### `start_capture()`
Initializes the keystroke capture session.

**Returns**: None

**Usage**:
```python
agent = KeystrokeAgent()
agent.start_capture()
```

##### `process_keystroke(char: str, timestamp: float = None, is_backspace: bool = False)`
Processes individual keystroke events and calculates timing metrics.

**Parameters**:
- `char` (str): The character that was typed
- `timestamp` (float, optional): Unix timestamp of the event
- `is_backspace` (bool): Whether this is a backspace event

**Returns**: None

**Usage**:
```python
agent.process_keystroke('a', time.time())
agent.process_keystroke('\b', time.time(), is_backspace=True)
```

##### `process_key_release(char: str, timestamp: float = None)`
Processes key release events to calculate dwell times.

**Parameters**:
- `char` (str): The character that was released
- `timestamp` (float, optional): Unix timestamp of the release

**Returns**: None

##### `extract_features(text: str, reference_text: str) -> Dict`
Extracts comprehensive behavioral features from typing session.

**Parameters**:
- `text` (str): The text that was typed by the user
- `reference_text` (str): The reference text that should have been typed

**Returns**: 
- `Dict`: Dictionary containing extracted features

**Feature Dictionary Structure**:
```python
{
    'typing_speed_wpm': float,           # Words per minute
    'typing_speed_cpm': float,           # Characters per minute
    'avg_dwell_time': float,             # Average key press duration
    'std_dwell_time': float,             # Standard deviation of dwell times
    'avg_flight_time': float,            # Average time between keystrokes
    'std_flight_time': float,            # Standard deviation of flight times
    'pause_count': int,                  # Number of significant pauses
    'avg_pause_duration': float,         # Average duration of pauses
    'backspace_count': int,              # Number of corrections made
    'accuracy': float,                   # Typing accuracy percentage
    'total_time': float,                 # Total session duration
    'rhythm_consistency': float,         # Consistency of typing rhythm
    'common_bigram_times': Dict,         # Timing for common letter pairs
    'key_pressure_variation': float,     # Variation in key press durations
    'error_correction_pattern': float    # Pattern analysis of corrections
}
```

##### `save_data_to_csv(user_id: str, filepath: str = "typing_behavior_data.csv")`
Saves collected data to CSV file.

**Parameters**:
- `user_id` (str): Unique identifier for the user
- `filepath` (str): Path to save the CSV file

**Returns**: None

##### `reset_session()`
Resets all session data for a new capture session.

**Returns**: None

##### `get_real_time_metrics() -> Dict`
Provides real-time typing metrics during active session.

**Returns**:
```python
{
    'current_wpm': float,
    'current_accuracy': float,
    'current_errors': int,
    'session_duration': float,
    'characters_typed': int
}
```

### BehaviorModelAgent

Creates and manages user behavioral profiles using machine learning algorithms.

#### Class Definition
```python
class BehaviorModelAgent:
    def __init__(self)
```

#### Methods

##### `create_user_profile(user_id: str, features: Dict)`
Creates a new user profile with initial behavioral features.

**Parameters**:
- `user_id` (str): Unique identifier for the user
- `features` (Dict): Feature dictionary from KeystrokeAgent

**Returns**: None

##### `train_user_model(user_id: str, min_samples: int = 2)`
Trains machine learning models for user behavior analysis.

**Parameters**:
- `user_id` (str): User identifier
- `min_samples` (int): Minimum samples required for training (default: 2)

**Returns**: None

**Model Details**:
- Uses ensemble approach with Isolation Forest and One-Class SVM
- Features are standardized using StandardScaler
- Models are saved automatically using joblib

##### `predict_anomaly(user_id: str, features: Dict) -> Tuple[bool, float]`
Predicts whether current session shows anomalous behavior.

**Parameters**:
- `user_id` (str): User identifier
- `features` (Dict): Current session features

**Returns**:
- `Tuple[bool, float]`: (is_anomaly, confidence_score)

**Usage**:
```python
is_anomaly, confidence = agent.predict_anomaly("user123", features)
if is_anomaly:
    print(f"Anomaly detected with {confidence:.2f} confidence")
```

##### `_save_model(user_id: str)` / `_load_model(user_id: str)`
Private methods for model persistence.

**Parameters**:
- `user_id` (str): User identifier

**File Structure**:
```
data/
├── models/
│   ├── {user_id}_model.pkl
│   └── {user_id}_scaler.pkl
```

### FraudDetectionAgent

Analyzes sessions for potential fraud and generates security alerts.

#### Class Definition
```python
class FraudDetectionAgent:
    def __init__(self)
```

#### Methods

##### `analyze_session(user_id: str, features: Dict, is_anomaly: bool, confidence_score: float, additional_context: Dict = None)`
Performs comprehensive session analysis for fraud detection.

**Parameters**:
- `user_id` (str): User identifier
- `features` (Dict): Session features
- `is_anomaly` (bool): Anomaly detection result
- `confidence_score` (float): Confidence level of anomaly detection
- `additional_context` (Dict, optional): Additional contextual information

**Returns**: None

**Analysis Components**:
- Risk scoring based on behavioral deviation
- Context analysis (time, location, device)
- Historical pattern comparison
- Threat level classification

##### `get_fraud_alerts() -> List[Dict]`
Retrieves current active fraud alerts.

**Returns**:
```python
[
    {
        'user_id': str,
        'timestamp': datetime,
        'threat_level': str,  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
        'confidence': float,
        'description': str,
        'recommended_action': str
    }
]
```

### DataManagerAgent

Handles all data persistence and management operations.

#### Class Definition
```python
class DataManagerAgent:
    def __init__(self, data_dir: str = "data")
```

#### Methods

##### `save_user_data(user_id: str, data: Dict)`
Saves user profile data to JSON format.

**Parameters**:
- `user_id` (str): User identifier
- `data` (Dict): User profile data

**File Structure**:
```
data/
├── users/
│   └── {user_id}_profile.json
```

##### `load_user_data(user_id: str) -> Optional[Dict]`
Loads user profile data from storage.

**Parameters**:
- `user_id` (str): User identifier

**Returns**:
- `Optional[Dict]`: User profile data or None if not found

##### `save_session_log(session_data: Dict)`
Saves session data to CSV log file.

**Parameters**:
- `session_data` (Dict): Session information

**CSV Schema**:
```python
{
    'timestamp': datetime,
    'user_id': str,
    'session_duration': float,
    'typing_speed_wpm': float,
    'accuracy': float,
    'anomaly_detected': bool,
    'confidence_score': float,
    'threat_level': str
}
```

##### `get_session_logs() -> pd.DataFrame`
Retrieves all session logs as pandas DataFrame.

**Returns**:
- `pd.DataFrame`: Complete session history

## UI Functions

### Theme System

##### `get_theme_css(theme_mode="light", font_size="medium", high_contrast=False)`
Generates dynamic CSS based on theme settings.

**Parameters**:
- `theme_mode` (str): "light" or "dark"
- `font_size` (str): "small", "medium", "large", "extra-large"
- `high_contrast` (bool): Enable high contrast mode

**Returns**:
- `str`: CSS styles for the application

### Page Functions

##### `show_home_page()`
Displays the main dashboard with system overview and statistics.

##### `show_registration_page()`
Handles user registration and initial profile creation.

##### `show_verification_page()`
Manages user authentication and behavioral verification.

##### `show_admin_dashboard()`
Provides administrative interface for system monitoring.

##### `show_settings_page()`
Manages application settings including theme and accessibility options.

## Configuration

### Session State Variables

The application uses Streamlit session state to maintain application state:

```python
st.session_state = {
    'current_page': str,           # Current active page
    'theme_mode': str,             # Current theme (light/dark)
    'font_size': str,              # Current font size setting
    'high_contrast': bool,         # High contrast mode status
    'reduce_motion': bool,         # Reduced motion preference
    'keystroke_agent': KeystrokeAgent,
    'behavior_agent': BehaviorModelAgent,
    'fraud_agent': FraudDetectionAgent,
    'data_manager': DataManagerAgent
}
```

### Data Directories

```
data/
├── models/                    # ML model storage
│   ├── {user_id}_model.pkl   # Trained anomaly detection models
│   └── {user_id}_scaler.pkl  # Feature scalers
├── users/                    # User profile storage
│   └── {user_id}_profile.json
├── session_log.csv          # Main session log
└── session_log.csv.backup   # Backup session log
```

## Error Handling

### Common Exceptions

- **FileNotFoundError**: When user models or profiles don't exist
- **ValueError**: When invalid parameters are passed to methods
- **ModelNotTrainedException**: When attempting to predict without trained model

### Error Responses

Most methods handle errors gracefully and return appropriate default values:

```python
# Example error handling in predict_anomaly
try:
    prediction = model.predict(features)
    return bool(prediction[0] == -1), confidence_score
except Exception as e:
    print(f"Error in anomaly prediction: {e}")
    return False, 0.0  # Default safe values
```

## Usage Examples

### Complete User Registration Flow

```python
# Initialize agents
keystroke_agent = KeystrokeAgent()
behavior_agent = BehaviorModelAgent()
data_manager = DataManagerAgent()

# Start capture session
keystroke_agent.start_capture()

# Process user typing (simulated)
for char in "Hello World":
    keystroke_agent.process_keystroke(char, time.time())

# Extract features
features = keystroke_agent.extract_features("Hello World", "Hello World")

# Create user profile
user_id = "user123"
behavior_agent.create_user_profile(user_id, features)

# Save user data
data_manager.save_user_data(user_id, {
    'profile_created': datetime.now(),
    'features': features
})

# Train model (after multiple sessions)
behavior_agent.train_user_model(user_id)
```

### Fraud Detection Flow

```python
# Load existing user
user_data = data_manager.load_user_data("user123")

# Capture new session
keystroke_agent.reset_session()
keystroke_agent.start_capture()

# ... typing session ...

# Extract features and detect anomaly
features = keystroke_agent.extract_features(typed_text, reference_text)
is_anomaly, confidence = behavior_agent.predict_anomaly("user123", features)

# Analyze for fraud
fraud_agent.analyze_session("user123", features, is_anomaly, confidence)

# Check for alerts
alerts = fraud_agent.get_fraud_alerts()
if alerts:
    print(f"Security alert: {alerts[0]['description']}")
```

This API reference provides comprehensive documentation for integrating with and extending the DefendX fraud detection system.
