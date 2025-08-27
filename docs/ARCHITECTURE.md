# DefendX System Architecture

## Overview

DefendX follows a multi-agent architecture pattern where specialized agents collaborate to provide comprehensive fraud detection capabilities. The system is built as a monolithic Streamlit application with clear separation of concerns through agent-based design.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │    Home     │ │Registration │ │Verification │ │ Admin  │ │
│  │    Page     │ │    Page     │ │    Page     │ │Dashboard│ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Theme & UI System                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │Light/Dark   │ │Accessibility│ │    Professional CSS    │ │
│  │Theme System │ │  Features   │ │      Styling           │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Multi-Agent Core                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │ Keystroke   │ │ Behavior    │ │   Fraud     │ │  Data  │ │
│  │   Agent     │ │ Model Agent │ │ Detection   │ │Manager │ │
│  │             │ │             │ │   Agent     │ │ Agent  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Machine Learning Layer                     │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │ Isolation   │ │ One-Class   │ │    Feature Extraction   │ │
│  │   Forest    │ │    SVM      │ │    & Preprocessing      │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Data Persistence                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐ │
│  │  CSV Files  │ │ Model Files │ │   JSON Configuration   │ │
│  │(Session Log)│ │  (joblib)   │ │     (Profiles)         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Agent Architecture

### 1. KeystrokeAgent

**Purpose**: Captures and analyzes keystroke dynamics in real-time.

**Key Components**:
- Key event capture and timing analysis
- Dwell time and flight time calculation
- Typing rhythm and pattern detection
- Feature extraction from keystroke sequences

**Methods**:
- `start_capture()`: Initializes keystroke monitoring
- `process_keystroke()`: Handles individual key events
- `extract_features()`: Extracts behavioral features
- `get_real_time_metrics()`: Provides live typing statistics

**Features Extracted**:
- Dwell times (key press duration)
- Flight times (time between key presses)
- Typing speed (WPM, CPM)
- Error patterns and corrections
- Rhythm analysis and pause patterns

### 2. BehaviorModelAgent

**Purpose**: Creates and manages user behavioral profiles using machine learning.

**Key Components**:
- User profile creation and management
- Machine learning model training
- Behavioral pattern analysis
- Anomaly detection model management

**Methods**:
- `create_user_profile()`: Establishes new user profiles
- `train_user_model()`: Trains ML models on user data
- `predict_anomaly()`: Detects deviations from normal behavior
- `_save_model()` / `_load_model()`: Model persistence

**Machine Learning Pipeline**:
- Feature standardization using StandardScaler
- Ensemble approach with Isolation Forest and One-Class SVM
- Model persistence with joblib
- Continuous learning from user sessions

### 3. FraudDetectionAgent

**Purpose**: Analyzes sessions for potential fraud and security threats.

**Key Components**:
- Real-time threat assessment
- Risk scoring and classification
- Alert generation and management
- Historical fraud pattern analysis

**Methods**:
- `analyze_session()`: Comprehensive session analysis
- `get_fraud_alerts()`: Retrieves active security alerts

**Detection Capabilities**:
- Behavioral anomaly detection
- Session-based risk scoring
- Real-time threat classification
- Historical pattern matching

### 4. DataManagerAgent

**Purpose**: Handles all data persistence and management operations.

**Key Components**:
- User data storage and retrieval
- Session logging and management
- Data format standardization
- File system operations

**Methods**:
- `save_user_data()` / `load_user_data()`: User profile management
- `save_session_log()`: Session data persistence
- `get_session_logs()`: Historical data retrieval

**Data Storage**:
- CSV format for session logs
- JSON format for user profiles
- Pickle format for ML models
- Structured directory organization

## Theme and UI System

### Theme Architecture

The application implements a sophisticated theme system with:

**Theme Modes**:
- Light mode (default)
- Dark mode
- High contrast mode (accessibility)

**Accessibility Features**:
- Font size adjustment (small, medium, large, extra-large)
- Reduced motion options
- High contrast color schemes
- WCAG 2.1 compliance considerations

**CSS Generation**:
- Dynamic CSS generation based on theme settings
- Professional styling with gradient backgrounds
- Responsive design patterns
- Custom component styling

### UI Components

**Navigation**:
- Sidebar-based navigation with brand identity
- Page-based routing system
- Theme toggle integration
- Settings panel access

**Professional Styling**:
- Custom CSS with professional color schemes
- Card-based layout system
- Interactive visualizations with Plotly
- Responsive design patterns

## Data Flow

### User Registration Flow
```
User Input → KeystrokeAgent → Feature Extraction → BehaviorModelAgent → 
Model Training → DataManagerAgent → Profile Storage
```

### Verification Flow
```
User Input → KeystrokeAgent → Feature Extraction → BehaviorModelAgent → 
Anomaly Detection → FraudDetectionAgent → Risk Assessment → Alert Generation
```

### Session Management Flow
```
Session Start → Data Collection → Real-time Analysis → 
Fraud Detection → Logging → DataManagerAgent → Persistence
```

## Security Considerations

### Data Protection
- Local data storage (no external dependencies)
- User data isolation
- Session-based security
- Model encryption capabilities

### Privacy Features
- No external data transmission
- Local processing only
- User data ownership
- Configurable data retention

## Scalability Design

### Modular Architecture
- Agent-based separation of concerns
- Pluggable component design
- Easy feature extension
- Independent agent scaling

### Performance Optimization
- Efficient feature extraction algorithms
- Optimized ML model training
- Lazy loading of components
- Memory-efficient data structures

## Technology Integration

### Streamlit Framework
- Component-based UI development
- Session state management
- Real-time updates
- Interactive widgets

### Machine Learning Stack
- scikit-learn for ML algorithms
- pandas for data manipulation
- numpy for numerical operations
- plotly for visualizations

### Data Persistence
- CSV for structured data
- JSON for configuration
- joblib for model serialization
- File-based storage system

## Deployment Architecture

### Single Application Deployment
- Monolithic Streamlit application
- Self-contained execution
- No external service dependencies
- Cross-platform compatibility

### Configuration Management
- Environment-based settings
- Runtime configuration
- Theme and accessibility preferences
- User-specific configurations

This architecture provides a robust, scalable, and maintainable foundation for the DefendX fraud detection system while maintaining simplicity and ease of deployment.
