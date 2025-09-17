# DefendX Android App Flow (Based on Actual Code)

## Navigation
- **MainActivity** sets up navigation with four screens:
  - `home` → HomeScreen
  - `security` → SecurityScreen
  - `test` → TestScreen
  - `verify` → VerifyScreen
- Navigation is handled using Jetpack Compose's NavController.
- App theme is set with `DEFEND_XTheme`.

## HomeScreen
- Shows app name, system status, and three main actions:
  - Setup Security Question (navigates to SecurityScreen)
  - Run Authentication Test (navigates to TestScreen)
  - Identity Verification (navigates to VerifyScreen)

## SecurityScreen
- User enters and saves a security answer (stored in SharedPreferences).
- Used as fallback authentication if behavioral test is suspicious/critical.

## TestScreen
- User types a fixed sentence.
- **Real keystroke events are captured** (press/release times for each key).
- Features are extracted from actual keystroke timings (dwell time, flight time, etc.).
- TFLite model runs inference on extracted features
- Model returns a result (Normal, Suspicious, Critical) with a confidence score.
- If result is "Suspicious" or "Critical":
  - Prompts user to answer their security question.
  - If correct, shows "Verified"; else, "Locked out".
- If result is "Normal" or "Verified":
  - Shows a sample workspace table (static data).

## VerifyScreen
- User answers their security question.
- If correct, navigates to HomeScreen and shows "Verified".
- If incorrect, shows "Wrong Answer".

## NotificationHelper
- Can show notifications that deep-link to the VerifyScreen.

## Data Handling
- Security answer is stored in SharedPreferences under key `security_answer`.
- Keystroke events are only used for local feature extraction and model inference.

## Summary
- All flows and features are based strictly on the code in the `android_app` folder, with improved model inference and real keystroke capture.

