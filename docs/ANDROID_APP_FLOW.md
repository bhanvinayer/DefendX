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
- Features are extracted (simulated keystroke timings).
- If TFLite model is present, runs inference; else, uses fallback logic.
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


## Summary
- All flows and features are based strictly on the code in the `android_app` folder.

