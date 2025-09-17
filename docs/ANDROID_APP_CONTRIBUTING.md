# Contributing to DefendX Android App

This guide is based strictly on the current Android app codebase.

## How to Contribute

1. **Clone the repository**
   ```bash
   git clone https://github.com/bhanvinayer/DefendX.git
   ```
2. **Open in Android Studio**
   - Open the `src/android_app` folder as a project.
3. **Build and Run**
   - Use Android Studio to build and run the app on an emulator or device.

## Code Structure
- All code is in Kotlin using Jetpack Compose.
- Main files:
  - `MainActivity.kt`: Navigation and theme
  - `ui/`: All screens (Home, Security, Test, Verify)
  - `utils/`: TFLiteModel and NotificationHelper

## Adding Features
- Follow the existing pattern for new screens (Composable functions, navigation route in MainActivity).
- For model improvements, update `TFLiteModel.kt` (feature extraction, inference, thresholds).
- For UI changes, use Jetpack Compose best practices.

## Code Style
- Use Kotlin idioms and Jetpack Compose conventions.
- Keep functions small and focused.
- Use clear, descriptive names for variables and functions.
- Add comments for non-obvious logic, especially in model inference and feature extraction.

## Testing
- Manual testing: Use the app on emulator/device, check all flows (test, fallback, verification).
- No automated tests are present in the current codebase.

## Submitting Changes
1. Commit your changes with a clear message.
2. Push to your fork/branch.
3. Open a pull request on GitHub.

## What Not to Do
- Do not add backend/cloud integration unless discussed.
- Do not add features not present in the current code without approval.

## Questions?
- Review the code in `src/android_app` for examples.
- Open an issue or discussion in the repository if you need help.
