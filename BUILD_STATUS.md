# SketchAI build / test status (2026-10-30)

## Validated in this environment

- Python training completed: 9,000 synthetic train / 1,500 synthetic holdout.
- Held-out synthetic score: 1498/1500 (99.87%) — **NOT human-drawing accuracy**.
- `.tflite` verified as TFL3 schema version 3, with two FULLY_CONNECTED
  operators and one SOFTMAX; expected tensor shapes, saved weights and
  reference NumPy predictions matched.
- Local Java compilation passed against **API stubs only**, NOT Android SDK.
- XML files parse; source tree references and ZIP contents checked.
- LiteRT 1.4.2 and AGP 8.7.3 selected for 16 KB compatibility precautions.

## Not yet validated

- Gradle sync or APK build using full Android SDK.
- Interpreter execution on an actual Android emulator/device.
- Native ELF / APK 16 KB alignment of an actual APK.
- Real finger-drawing classification and full UI behavior.
- Original instructor repository clone/build/run and screenshot evidence.

## Recommended next action

Open Android Studio on Windows 11 → select project root → Sync Gradle →
Run on emulator or device → record results and screenshot → Analyze APK for 16 KB.

Any error should be repaired before describing the assignment as submission-ready.
