# DefendX Android App Overview

This document describes the actual code and features present in the `android_app` folder of the DefendX project.

## Main Components

- **MainActivity.kt**: Entry point. Sets up navigation and theme. Handles notification intent for direct verification.
- **ui/HomeScreen.kt**: Home dashboard. Shows system status and navigation to security, test, and verification screens.
- **ui/SecurityScreen.kt**: Lets user set a security answer (stored in SharedPreferences).
- **ui/TestScreen.kt**: Behavioral biometric authentication test. Uses a TFLite model. If suspicious/critical, prompts for security answer.
- **ui/VerifyScreen.kt**: Standalone security answer verification screen.
- **utils/TFLiteModel.kt**: Loads and runs a TensorFlow Lite model for behavioral analysis. 
- **utils/NotificationHelper.kt**: Shows notifications, optionally deep-linking to verification.
- **AndroidManifest.xml**: Declares permissions and main activity.
- **res/values/strings.xml**: App name and string resources.

## App Flow

1. **HomeScreen**: Entry dashboard. Navigate to security setup, test, or verification.
2. **SecurityScreen**: User sets a security answer (used as fallback verification).
3. **TestScreen**: User types a test sentence. Behavioral features are extracted and analyzed (TFLite). If result is suspicious/critical, user must answer their security question.
4. **VerifyScreen**: User answers their security question for identity verification.

## Data & Security

- Security answer is stored locally in SharedPreferences.
- Behavioral features are extracted from typing input.
- TFLite model is used.
- Notifications can trigger direct navigation to verification.

## Technologies Used

- Kotlin, Jetpack Compose for UI
- Android SharedPreferences for local storage
- Optional TensorFlow Lite for behavioral analysis
- Android notification APIs


