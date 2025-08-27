# Quick Start Guide

## 🚀 Getting Started

### 1. Installation
```bash
# Clone or download the project
cd userbehavior

# Run the startup script (Windows)
start_app.bat

# Or manually start:
pip install -r requirements.txt
streamlit run app.py
```

### 2. First Time Setup
1. Open http://localhost:8501 in your browser
2. Navigate to "User Registration" 
3. Create your first user profile
4. Type the reference sentences 3-5 times for baseline
5. Train the user model

### 3. Testing the System
1. Go to "Verification Test"
2. Select your registered user
3. Type a verification sentence
4. See real-time fraud detection results

## 🎯 Key Features Demo

### User Registration
- **Purpose**: Establish behavioral baseline
- **Process**: Type reference sentences multiple times
- **Metrics**: WPM, accuracy, keystroke timing
- **Outcome**: Trained user model

### Real-time Verification  
- **Purpose**: Authenticate user identity
- **Process**: Type verification text
- **Analysis**: Compare against baseline
- **Result**: Fraud score and risk assessment

### Admin Dashboard
- **Users**: View all registered users
- **Analytics**: Behavioral trends and patterns
- **Alerts**: Fraud detection history
- **Export**: Download session data

## 🔧 System Configuration

### Fraud Detection Sensitivity
- Navigate to "System Settings"
- Adjust fraud threshold (0.1 = sensitive, 1.0 = relaxed)
- Set minimum baseline samples
- Configure risk parameters

### Data Management
- View system statistics
- Export session logs
- Clear system data
- Monitor performance

## 🎮 Demo Scenarios

The system includes built-in demo scenarios:

1. **Normal Behavior**: Consistent typing patterns
2. **Speed Anomaly**: Unusually fast/slow typing
3. **Accuracy Drop**: Significant errors increase  
4. **Impersonation**: Multiple behavioral mismatches
5. **Behavioral Drift**: Gradual pattern changes
6. **Multi-User**: Comparative analysis

## 🔍 Understanding Results

### Risk Indicators
- **Green (0.0-0.3)**: Normal behavior
- **Yellow (0.3-0.7)**: Suspicious activity  
- **Red (0.7-1.0)**: High fraud risk

### Common Fraud Patterns
- Speed significantly faster/slower than baseline
- Accuracy drop below normal range
- Inconsistent keystroke timing
- Multiple simultaneous anomalies

## 📊 Metrics Explained

### Core Features
- **WPM**: Words per minute typing speed
- **Accuracy**: Percentage of correct characters
- **Flight Time**: Time between keystrokes
- **Rhythm Variance**: Consistency in timing

### Risk Factors
- Speed deviation > 50% from baseline
- Accuracy drop > 20% from baseline  
- High timing inconsistency
- Behavioral model anomaly detection

## 🛠️ Troubleshooting

### Common Issues
1. **No users found**: Complete user registration first
2. **Model not trained**: Need 3+ baseline samples
3. **Poor detection**: Adjust sensitivity threshold
4. **Performance issues**: Clear old session data

### Tips for Best Results
- Type naturally during registration
- Use consistent keyboard/device
- Complete 5+ baseline samples
- Regular system monitoring

## 🔐 Privacy & Security

### Data Protection
- All processing happens locally
- No external data transmission
- Encrypted local storage option
- User data anonymization

### Compliance
- GDPR-ready data handling
- User consent management
- Data retention policies
- Audit trail logging

## 📈 Advanced Usage

### Integration Options
- API endpoints for external systems
- Batch processing capabilities
- Custom model training
- Enterprise deployment

### Customization
- Custom reference texts
- Adjustable risk thresholds
- Feature engineering options
- Model parameter tuning

---

**Need Help?**
- Check the Admin Dashboard for system status
- Review session logs for patterns
- Adjust settings for your use case
- Contact support for advanced features
