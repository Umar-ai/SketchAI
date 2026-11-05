# SketchAI — Offline AI Drawing Classifier (Java + LiteRT)

A complete **Android Studio mini application** for **Assignment No. 2**. Draw a
**circle, square, triangle, star, or heart** with your finger; a trained neural
network inside the app returns a label and five model scores. Works offline.

## Project details

- Student: **Umer Farooq Abbasi**
- Registration number: **242201066**
- Submitted to: **Sir Uzair**
- Language: **Java 17**; UI: **Android XML**
- ML runtime: **LiteRT 1.4.2** (`org.tensorflow.lite.Interpreter` Java API)
- Android build: **Gradle 8.9 / Android Gradle Plugin 8.7.3 / compileSdk 35**
- Minimum SDK: 26; target SDK: 34
- Original instructor repository (to clone and analyze independently):
  https://github.com/Uzair390-Del/NumberGuessingAppTflite.git

## What you get

- `MainActivity.java` — UI, model startup, worker-thread prediction and results.
- `DrawingView.java` — finger drawing, multi-stroke, Undo/Clear, ink-only capture.
- `SketchPreprocessor.java` — crop, resize to 32×32, grayscale, normalize.
- `SketchClassifier.java` — model loading, tensor checks, real Interpreter.run().
- `app/src/main/assets/sketchai_shapes.tflite` — actual float32 neural network.
- `app/src/main/assets/labels.txt` — five prediction labels in output order.
- `app/src/main/res/layout/activity_main.xml` — complete screen, no Compose.
- `ml/train_export.py` — procedural training data + PyTorch network training +
  TFLite FlatBuffer exporter.
- `ml/tensorflow_conversion.py` — **optional standard TensorFlow/Keras to
  TensorFlow Lite conversion** from saved weights (requires TensorFlow).
- `ml/validate_model.py` and `ml/check_java_sources.py` — local static/model tests.
- `docs/` — architecture, source repo study, tests, screenshots, 16 KB check, GitHub.

## Run on Windows 11 / Android Studio

1. Right-click the ZIP file → **Extract All**. Do not open ZIP directly.
2. Start Android Studio → **File → Open** → select the directory containing
   `settings.gradle` (the **SketchAI_Android_Project** root).
3. Allow Android Studio to download dependencies. Internet is needed for this
   initial Gradle setup, but **not** for predictions after installation.
4. Android Studio → **Settings → Build, Execution, Deployment → Build Tools →
   Gradle → Gradle JDK** → select **JDK 17**, if Gradle reports a version error.
5. Install Android SDK Platform 35 in **Tools → SDK Manager** if requested.
6. Select **File → Sync Project with Gradle Files** (or Ctrl+Shift+A to search).
7. Select your emulator or connected Android phone (USB debugging enabled).
8. Select **app** run configuration and press **▶ Run** (Shift+F10).
9. Wait until the status label shows **AI ready**. Draw one shape, tap
   **Recognize**. Try **Undo**, **Clear**, and drawing several times.

**If virtualization is unavailable:** You can use a physical Android device with
USB debugging instead of an emulator.

### Troubleshooting

- **Design editor unavailable until next Gradle sync:** Sync from the *root*
  project; open **Build Output** for the actual error. Make sure Gradle has
  network access the first time.
- **Black screen or crash:** Check **View → Tool Windows → Logcat** and filter
  for `package:com.umer.sketchai` and `FATAL EXCEPTION`. This app calls
  `setContentView()` before initializing LiteRT, and reports load errors in UI.
- **Model could not load:** Send the error text and relevant Logcat lines;
  confirm the asset `.tflite` file exists and dependencies downloaded.
- **Could not find com.google.ai.edge.litert:** Check internet and Maven Central/
  Google Maven repositories, then retry Gradle sync.
- **16 KB page alignment warning:** Follow `docs/16KB_ALIGNMENT.md` to inspect
  affected native `.so` and verify the APK. Do not assume settings alone prove
  device compatibility.
- **Random/unfamiliar drawing:** The model supports **only five shapes**, one at
  a time. It was trained with synthetic data; odd drawings can be wrong.

## How the REAL ML inference works

1. Draw strokes on the touch canvas. No image is uploaded anywhere.
2. Capture only user ink (not the visible decorative grid).
3. Crop the alpha mask to the ink bounding box; scale the longest side to **26
   pixels**, center in **32×32** and normalize grayscale alpha to **[0,1]**.
4. Flatten to a float32 tensor **[1, 1024]**.
5. `org.tensorflow.lite.Interpreter.run(input, output)` runs locally.
6. Output float32 **[1,5]** gives softmax scores: Circle, Square, Triangle, Star,
   Heart. Highest score determines the label; scores are *not calibrated odds*.
7. If the highest score is under 60%, the screen says **Not sure**.

### Model architecture, training and limits

- Educational fully-connected neural network: **1024 → 96 ReLU → 5 Softmax**.
- Python PyTorch was used to train on **9,000 procedural synthetic drawings**
  (1,800 examples per class). A separate synthetic test set has **1,500** images.
- Synthetic held-out accuracy in this environment: **1498/1500 = 99.87%**.
  **This is NOT accuracy on actual hand-drawn sketches.** Phone strokes, unusual
  drawing styles, circles that look like ovals, and unknown shapes can fail.
- Exported to the standard TFLite `.tflite` FlatBuffer model with built-in
  `FULLY_CONNECTED`, `RELU`, and `SOFTMAX` operators. Its contents were checked
  against the trained weight arrays, but a device Interpreter run is pending.
- The output softmax uses beta=0.25 (temperature scaling) to avoid extremely
  saturated scores on training-like drawings. **Do not interpret the percentage
  as a statistically validated confidence estimate.**
- Rebuild model locally: `pip install numpy pillow torch`, then
  `python ml/train_export.py`. Check: `python ml/validate_model.py`.
- Standard TensorFlow model conversion demonstration is available separately:
  install TensorFlow and run `python ml/tensorflow_conversion.py`. The included
  prebuilt model was NOT exported with that optional script.

## 16 KB page-size policy

The app uses LiteRT **1.4.2** instead of legacy TensorFlow Lite **2.17.0**, and
Android Gradle Plugin **8.7.3** for native library packaging. These versions are
chosen to avoid the earlier `LOAD segments not aligned at 16 KB boundaries`
warning. **The built APK must still be analyzed and tested**, see
`docs/16KB_ALIGNMENT.md`.

## What has / has not been tested

Locally checked: Python neural training and evaluation, saved model binary
structure, TFLite tensor shapes/weights, reference NumPy inference, XML syntax,
Java source syntax against stub APIs, project file completeness.

Not checked here: Android SDK Gradle sync/build, execution on emulator or device,
actual LiteRT Interpreter run in Android, ELF/APK 16 KB alignment. These MUST be
verified on your Windows 11 Android Studio environment before submission.

## Assignment instructions

This project alone is **not the full course submission**. The instructor also
requires: cloning/running and analyzing the supplied reference repo, genuine
application screenshots, your own GitHub URL, a theory report PDF, an
implementation report PDF, code snippets PDF, and complete ZIP. See
`docs/ASSIGNMENT_REQUIREMENTS.md`.

## Sources and credit

- Instructor reference: https://github.com/Uzair390-Del/NumberGuessingAppTflite
- Google LiteRT: https://developers.google.com/edge/litert/android
- Android 16 KB guidance: https://developer.android.com/guide/practices/page-sizes
- Training drawings: procedurally generated by the provided code; no external
  images, Quick Draw dataset, or proprietary pretrained model is bundled.

## Reproducibility notes

The bundled Gradle wrapper JAR is a **documented lightweight bootstrap class**,
not Gradle's official signed wrapper binary, because the official JAR was not
available in the build environment. `gradle/wrapper/GradleWrapperMain.java`
is included for inspection; it downloads official Gradle 8.9 on first use.
For production-quality publishing, replace it using `gradle wrapper --gradle-version
8.9` from an installed trusted Gradle distribution, or generate the wrapper in
Android Studio. Do not confuse this bootstrap with the Android application.
