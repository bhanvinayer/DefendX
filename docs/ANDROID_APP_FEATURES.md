# DefendX Android App Features (Code-Accurate)

## Core Features
- Home dashboard with navigation to all main functions
- Security question setup and local storage (SharedPreferences)
- Behavioral biometric authentication test (fixed sentence)
- **Real keystroke timing capture** during typing (not simulated)
- Feature extraction from actual keystroke events (dwell time, flight time, etc.)
- TensorFlow Lite model for behavioral analysis (autoencoder)
- Model inference uses extracted features for confidence-based classification (Normal, Suspicious, Critical)
- Confidence score and detailed result shown to user
- Fallback logic if TFLite model is missing
- Security fallback: answer security question if test is suspicious/critical
- Standalone security question verification screen
- Local notification support (deep-link to verification)
- Static sample workspace table (shown after successful test)

## Improvements
- More accurate behavioral analysis using real keystroke data
- Confidence-based feedback for user transparency

## All features above are directly reflected in the code. No extra features are documented.