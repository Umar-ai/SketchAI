# Testing checklist — COMPLETE THIS ON AN ACTUAL ANDROID DEVICE/EMULATOR

All rows are **PENDING** until the student records actual results. Do not fill in
"Pass" automatically; include real screenshots in reports.

| ID | Test | Expected result | Actual outcome |
|----|------|-----------------|----------------|
| T01 | First opening after Gradle sync | App launches, no black screen, status AI ready | PENDING |
| T02 | Draw single large circle | Predicted shape and all 5 model scores appear | PENDING |
| T03 | Draw square | Scores update after Predict | PENDING |
| T04 | Draw triangle | Scores update after Predict | PENDING |
| T05 | Draw star | Scores update after Predict | PENDING |
| T06 | Draw heart | Scores update after Predict | PENDING |
| T07 | Tap Predict on empty board | Helpful message, no crash | PENDING |
| T08 | Draw 2 strokes, tap Undo | Removes the last stroke only | PENDING |
| T09 | Tap Clear | Removes all strokes and hides old prediction | PENDING |
| T10 | Disable internet after initial install | Inference still functions | PENDING |
| T11 | Rotate phone / recreate Activity | App reopens and model initializes | PENDING |
| T12 | Inspect APK in Analyze APK | No 16 KB LOAD alignment warning | PENDING |
| T13 | Test with 16 KB device image | `getconf PAGE_SIZE`=16384; app runs | PENDING |
| T14 | Draw unrelated scribble | Model may still pick a label; note limitation | PENDING |

Record: date, device name, Android version, model page size, predicted label,
score breakdown, actual result and screenshot file names.

Screenshot suggestions:
- `01_home_ready.png`: empty board + AI ready.
- `02_circle_prediction.png`: circle and score panel.
- `03_star_prediction.png`: star and score panel.
- `04_empty_canvas_validation.png`: toast on empty input.
- `05_gradle_sync.png`: successful sync.
- `06_analyze_apk_16kb.png`: verify alignment after APK build.

**Do not fabricate screenshots.**
