# Android 16 KB native library alignment — verification

The earlier DigitAI app showed a native ELF `LOAD segments not aligned at 16 KB`
warning. This app avoids the old TensorFlow Lite 2.17 runtime and instead uses
Google LiteRT 1.4.2 plus Android Gradle Plugin 8.7.3. These are prevention
steps, **not proof the APK has been verified**.

## Step 1 — Build APK

In Android Studio: **Build → Build APK(s)**; find
`app/build/outputs/apk/debug/app-debug.apk`.

## Step 2 — Inspect native files

Android Studio: **Build → Analyze APK**, select APK → `lib/` → `.so` files.
Check the ELF segment alignment for each `.so` and copy the filename of any
problematic library. If a warning appears, do not ignore it.

## Step 3 — Verify zip alignment (Windows PowerShell)

Find the installed Android SDK path in **Settings → Languages & Frameworks →
Android SDK**. Replace the path below with your real SDK path/build-tools.

```
& "C:\Users\YOUR_USER\AppData\Local\Android\Sdk\build-tools\35.0.0\zipalign.exe" -v -c -P 16 4 "app\build\outputs\apk\debug\app-debug.apk"
```

A passing zipalign command checks APK packaging; it does **not** prove all
native ELF LOAD segments have 16 KB alignment.

## Step 4 — Test using a 16 KB emulator/device

Use a 16 KB Android system image (where offered in SDK Manager). Run:

```
adb shell getconf PAGE_SIZE
```

Expected: `16384` for an actual 16 KB environment. Launch app; draw and
classify a circle/heart and check Logcat.

## If the warning persists

Send the expanded Android Studio message listing exact `.so` filename(s).
Update or replace that specific dependency with a verified compatible version.
Do not suppress a warning via manifest flags instead of fixing the binary.

Official documentation: https://developer.android.com/guide/practices/page-sizes
