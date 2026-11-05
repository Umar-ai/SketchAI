# GitHub submission steps (after verifying a working Android build)

1. Create a **public** GitHub repository, e.g. `SketchAI-Java-TFLite`.
2. Open Android Studio Terminal at root `SketchAI_Android_Project`.
3. Run:

```
git init
git add .
git commit -m "Build SketchAI on-device drawing classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/SketchAI-Java-TFLite.git
git push -u origin main
```

4. Replace `YOUR_USERNAME` with your GitHub username. Verify README and
   `app/src/main/assets/sketchai_shapes.tflite` exist on GitHub.
5. Do not commit `local.properties`, credentials, `build/`, `.gradle/`, or
   private student data; `.gitignore` already excludes local settings.
6. Include the real GitHub link in both practical and theory PDFs.
7. Credit the instructor's repo and the LiteRT runtime, and describe the
   synthetic dataset truthfully.
