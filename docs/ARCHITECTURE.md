# SketchAI architecture and workflow

```
Finger touches
    ↓
DrawingView.java — Paths, Undo, Clear
    ↓ (ink-only ARGB_8888 Bitmap)
SketchPreprocessor.java — bbox, fit 26 px, center 32×32, alpha/255
    ↓ (float[1][1024])
SketchClassifier.java — ByteBuffer asset, LiteRT Interpreter.run()
    ↓ (float[1][5])
MainActivity.java — best class, label, bar chart, uncertainty notice
```

## Why this is an AI app

`Interpreter.run()` executes learned matrix weights: 1024 pixels → dense layer
96 units with ReLU → dense layer 5 units → softmax. The `if` statements in the
Activity only decide what label and UI to show; they do **not** classify shapes.

## Threading and safety

- Activity renders layout immediately so model initialization doesn't block UI.
- A single background executor owns the Interpreter. This keeps AI operations
  off the main UI thread and serializes inference.
- UI captures user ink, preprocesses it, then passes immutable float arrays.
- Model tensor shape and data type are checked at startup; failures show on UI.
- No INTERNET permission or user-data access is requested.
- Interpreter is closed during Activity destruction.

## Preprocessing example

1. Find all pixels with alpha > 8; crop bounding box.
2. Resize crop preserving aspect ratio, fitting in 26×26.
3. Place at center of 32×32 black image.
4. Read alpha intensity into array (white strokes=1, black background=0).
5. Flatten row-major into model input `[1,1024]`.

Training images use the same white-on-black convention and the same
crop/fit/center operation. The Android implementation interpolates with Android
Bitmap and the generator uses Pillow bilinear interpolation; edges may differ.

## Supported shapes and limitations

Circle, Square, Triangle, Star, Heart. One recognizable shape at a time.
Synthetic procedural training is a demonstration, NOT a model trained with a
real human drawing dataset. Test with different people's sketches before making
any real-world accuracy statements.
