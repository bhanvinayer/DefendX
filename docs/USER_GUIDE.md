# DefendX User Guide

## Getting Started

Welcome to DefendX, the advanced multi-agent fraud detection system. This guide will help you understand and effectively use all features of the system.

## Overview

DefendX analyzes your typing behavior to create a unique biometric profile that can detect when someone else is using your account. The system learns your typing patterns and alerts administrators when suspicious activity is detected.

## First Time Setup

### 1. Access the Application

1. Open your web browser
2. Navigate to `http://localhost:8501` (or the URL provided by your administrator)
3. You'll see the DefendX Security Platform homepage

### 2. Theme and Accessibility Settings

Before starting, you may want to customize your experience:

1. Click on **Settings** in the sidebar
2. Choose your preferred theme:
   - **Light Mode**: Standard bright interface
   - **Dark Mode**: Dark background for reduced eye strain
3. Adjust accessibility options:
   - **Font Size**: Small, Medium, Large, or Extra Large
   - **High Contrast**: Enhanced contrast for better visibility
   - **Reduce Motion**: Minimizes animations

## User Registration

### Step 1: Create Your Profile

1. Click **Registration** in the sidebar
2. Enter a unique **User ID** (e.g., your employee ID or username)
3. The system will display sample text for you to type

### Step 2: Complete Training Sessions

**Important**: You need to complete multiple training sessions for accurate detection.

1. **First Session**:
   - Type the displayed text exactly as shown
   - Don't worry about speed - focus on accuracy
   - The system captures your natural typing rhythm

2. **Subsequent Sessions**:
   - Complete at least 3-5 training sessions
   - Try to type naturally each time
   - Sessions should be spread over different times if possible

3. **Training Tips**:
   - Type at your normal speed
   - Don't try to type perfectly - natural errors help build your profile
   - Take breaks between sessions
   - Use your usual typing posture and environment

### Step 3: View Your Progress

After each session, you'll see:
- **Typing Speed**: Your words per minute (WPM)
- **Accuracy**: Percentage of correctly typed characters
- **Training Status**: Progress toward completing your profile
- **Behavioral Metrics**: Advanced typing pattern analysis

## Daily Usage (Verification)

### Accessing Verification

1. Click **Verification** in the sidebar
2. Enter your User ID
3. Type the provided text sample

### Understanding Results

The system provides real-time feedback:

#### Normal Session (✅ Verified)
- **Green indicators**: Your typing matches your profile
- **Low risk score**: Typically below 0.3
- **Status**: "User Verified" or "Normal Behavior"

#### Suspicious Session (⚠️ Anomaly Detected)
- **Yellow/Red indicators**: Typing patterns differ from your profile
- **High risk score**: Above 0.7
- **Possible causes**:
  - You're unusually tired or stressed
  - Different keyboard or environment
  - You've injured your hand
  - Someone else is typing

#### What to Do If Flagged
1. **Don't panic** - false positives can occur
2. **Retry** the verification with normal typing
3. **Contact your administrator** if problems persist
4. **Note any changes** in your environment (new keyboard, injury, etc.)

## Features and Pages

### Home Page

The main dashboard shows:
- **System Status**: Overall health of the fraud detection system
- **Recent Activity**: Your recent verification sessions
- **Security Alerts**: Any system-wide security notifications
- **Quick Stats**: Total users, sessions, and system metrics

### Registration Page

Complete registration and training features:
- **New User Registration**: Create your initial profile
- **Additional Training**: Add more training data to improve accuracy
- **Profile Management**: View your training progress
- **Session History**: Review your past training sessions

### Verification Page

Daily authentication features:
- **Quick Verification**: Fast typing test for routine access
- **Detailed Analysis**: Comprehensive behavioral analysis
- **Risk Assessment**: Current threat level evaluation
- **Session Feedback**: Performance metrics and improvement tips

### Settings Page

Customization and preferences:
- **Theme Selection**: Light/Dark mode toggle
- **Accessibility Options**: Font size, contrast, motion settings
- **Profile Settings**: Personal preferences and configurations
- **System Information**: Version info and technical details

### Admin Dashboard (Administrators Only)

System monitoring and management:
- **User Management**: View all registered users
- **Security Monitoring**: Real-time threat detection
- **System Analytics**: Performance metrics and usage statistics
- **Alert Management**: Review and respond to security alerts

## Understanding Your Typing Profile

### Behavioral Metrics Explained

**Dwell Time**: How long you hold each key down
- Unique to each person like a fingerprint
- Affected by finger strength and typing style

**Flight Time**: Time between releasing one key and pressing the next
- Shows your typing rhythm and coordination
- Reflects muscle memory and familiarity

**Typing Speed**: Words per minute (WPM)
- Your average typing speed
- Can vary throughout the day

**Accuracy**: Percentage of correct characters
- How often you make typos
- Pattern of errors is unique to each person

**Pause Patterns**: When and how long you pause while typing
- Reflects thinking patterns
- Shows comfort level with different words

### Factors That Affect Your Profile

**Normal Variations**:
- Time of day (morning vs. evening)
- Fatigue level
- Stress or mood
- Recent caffeine consumption

**Environmental Factors**:
- Different keyboards
- Lighting conditions
- Sitting position
- Noise distractions

**Physical Factors**:
- Hand injuries or pain
- Nail length changes
- Temperature (cold hands type differently)

## Best Practices

### For Accurate Detection

1. **Consistent Environment**:
   - Use the same keyboard when possible
   - Maintain similar lighting
   - Keep consistent posture

2. **Regular Training**:
   - Complete training sessions regularly
   - Add training data if your typing changes
   - Update profile after major changes (injury, new keyboard)

3. **Natural Typing**:
   - Don't try to type "perfectly"
   - Maintain your natural rhythm
   - Don't rush or go unusually slow

### Security Tips

1. **Keep Your User ID Private**: Don't share your ID with others
2. **Report Suspicious Activity**: Contact administrators if you see unexpected alerts
3. **Update Your Profile**: Retrain if your typing significantly changes
4. **Monitor Results**: Pay attention to your verification scores

## Troubleshooting

### Common Issues

#### "Anomaly Detected" When It's Really You

**Possible Causes**:
- Insufficient training data
- Changed typing environment
- Unusual stress or fatigue
- Physical changes (injury, long nails)

**Solutions**:
1. Complete additional training sessions
2. Retry verification after a break
3. Contact administrator for profile review
4. Note any environmental changes

#### Can't Complete Registration

**Possible Issues**:
- Typing too fast or slow
- Not typing the text exactly as shown
- Technical browser issues

**Solutions**:
1. Clear browser cache and cookies
2. Try a different browser
3. Type at a moderate, comfortable pace
4. Ensure you're typing the exact text shown

#### Verification Takes Too Long

**Possible Causes**:
- System processing large amounts of data
- Network connectivity issues
- High system load

**Solutions**:
1. Wait for the system to complete processing
2. Check your internet connection
3. Try again during off-peak hours
4. Contact technical support if persistent

### Getting Help

#### Self-Service Options
1. **Settings Page**: Check system status and configuration
2. **Retry**: Most issues resolve with a second attempt
3. **Different Environment**: Try from a different location/computer

#### Administrator Support
Contact your system administrator for:
- Persistent verification failures
- Profile reset requests
- Technical system issues
- Security concerns

## Privacy and Security

### Data Protection

**What We Collect**:
- Keystroke timing data (not the actual content you type)
- Typing pattern metrics
- Session timestamps and results

**What We Don't Collect**:
- Personal information beyond User ID
- Actual text content you type
- Information from other applications
- Browsing history or personal files

### Data Storage

- All data is stored locally on your organization's servers
- No data is transmitted to external services
- Your typing patterns are encrypted and secured
- Data retention follows your organization's policies

### Your Rights

- **Access**: View your own profile and session data
- **Correction**: Request updates to incorrect information
- **Deletion**: Request removal of your profile (contact administrator)
- **Portability**: Export your data if needed

## Advanced Features

### Real-Time Monitoring

The system provides live feedback during typing:
- **Speed Indicator**: Current WPM as you type
- **Accuracy Meter**: Real-time accuracy percentage
- **Rhythm Analysis**: Visual feedback on typing consistency
- **Error Detection**: Immediate feedback on unusual patterns

### Adaptive Learning

Your profile continuously improves:
- **Automatic Updates**: Profile adapts to gradual changes in typing
- **Seasonal Adjustments**: Accounts for normal variations over time
- **Context Awareness**: Considers time of day and session length
- **Performance Optimization**: System learns optimal detection thresholds

### Integration Features

**Single Sign-On (SSO)**: May integrate with your organization's authentication system
**API Access**: Programmatic access for developers (administrator-controlled)
**Reporting**: Detailed analytics for security teams
**Alerts**: Automated notifications for security events

This user guide provides comprehensive information for effectively using the DefendX fraud detection system. For additional support or advanced configuration options, contact your system administrator.
