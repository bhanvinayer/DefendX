# DefendX Android App Features

This list is strictly based on the current code.

## Core Features
- Home dashboard with navigation to all main functions
- Security question setup and local storage (SharedPreferences)
- Behavioral biometric authentication test (fixed sentence)
- Real keystroke timing capture during typing (not simulated)
- Feature extraction from actual keystroke events (dwell time, flight time, WPM, etc.)
- TensorFlow Lite model for behavioral analysis (autoencoder, optional)
- Model inference uses extracted features for confidence-based classification (Normal, Suspicious, Critical)
- Confidence score and detailed result shown to user
- Fallback logic if TFLite model is missing
- Security fallback: answer security question if test is suspicious/critical
- Standalone security question verification screen
- Local notification support (deep-link to verification)
- Static sample workspace table (shown after successful test)

## Security & Privacy
- Security answer is stored locally, not sent to server
- Keystroke data is used only for local inference

## No extra features, APIs, or integrations are present beyond the above actual code.