# DefendX Android App Architecture

This document describes the actual architecture of the DefendX Android app, strictly based on the code.

## Overview
- **Language:** Kotlin
- **UI:** Jetpack Compose
- **Navigation:** Jetpack Compose NavController
- **Model Inference:** TensorFlow Lite (autoencoder)
- **Local Storage:** Android SharedPreferences
- **Notifications:** Android Notification APIs

## Main Components
- **MainActivity.kt**: Sets up navigation, handles notification intents, applies app theme.
- **ui/HomeScreen.kt**: Home dashboard, navigation to all main features.
- **ui/SecurityScreen.kt**: Security question setup, stores answer in SharedPreferences.
- **ui/TestScreen.kt**: Behavioral biometric authentication test. Captures real keystroke events, extracts features, runs TFLite model (if present), shows result and confidence, triggers fallback security question if needed.
- **ui/VerifyScreen.kt**: Standalone security question verification.
- **utils/TFLiteModel.kt**: Loads and runs TFLite model. Extracts features from real keystroke data, computes confidence, classifies as Normal/Suspicious/Critical.
- **utils/NotificationHelper.kt**: Shows notifications, can deep-link to verification.
- **AndroidManifest.xml**: Declares permissions, main activity.
- **res/values/strings.xml**: App name and string resources.

## Data Flow
1. **User types in TestScreen** → Keystroke events captured (press/release times)
2. **Feature extraction** → Dwell time, flight time, WPM, etc.
3. **TFLiteModel** → Model inference on features, returns result and confidence
4. **Result** → If "Suspicious"/"Critical", prompt for security question
5. **Security answer** → Checked against SharedPreferences
6. **Notifications** → Can trigger direct navigation to verification

## Security
- Security answer is stored locally (not sent to server)
- Keystroke data is used only for local inference
- No backend or cloud integration in current code

## No extra layers, APIs, or features are present beyond the above actual code.