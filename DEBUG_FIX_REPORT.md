# Deep-Live-Cam debugging and fixes — 8 October 2026

## 1. Root cause
The model itself can swap faces correctly on CPU. The GUI did not consume the saved result; Preview independently recomputed the target on the GUI thread. Start only ran file processing, while Live was a separate action. Core copied the original target to the output and reported success based on the input being an image, even if the swapper returned early. This could leave an unchanged target labelled successful.

Windows capture used constructor property parameters that may be unsupported by the backend. Capture failures closed a partially initialized preview whose stop event and workers did not exist. Model initialization/camera setup also blocked the GUI, and worker exceptions and shutdown were not handled safely. Detection reused stale face geometry between processed frames.

CUDA was a separate environment issue: Python 3.11.9, onnxruntime-gpu 1.26.0, Intel UHD Graphics (driver 31.0.101.5522), no NVIDIA GPU reported, no nvidia-smi/nvcc/CUDA_PATH, and missing NVIDIA driver/CUDA/cuDNN DLLs. A compiled CUDA provider listing does not demonstrate working acceleration.

## 2. Files changed
- modules/ui.py
- modules/core.py
- modules/video_capture.py
- modules/face_analyser.py
- modules/processors/frame/face_swapper.py
- modules/processors/frame/_onnx_enhancer.py
- modules/runtime.py (new)
- modules/platform_info.py
- run.py
- run-cpu.bat (new)
- .gitignore (allow the CPU launcher)
- tests/verify_pipeline.py (new integration check)
- tests/verify_camera.py (new opt-in physical webcam check)
- DEBUG_FIX_REPORT.md

Generated verification images are in diagnostics/. The existing root output.png was preserved.

## 3. Exact changes
- Move file processing, preview inference and target-map analysis into QThread workers. Send arrays/results to Qt through signals and preserve the enabled state of controls.
- Send the actual verified saved image to Preview. Reopening Preview shows that saved result.
- Validate source/target faces and reject unchanged results with a clear status. Log paths, shapes, detected source count, selected bounding boxes, target count, active providers and changed channel values.
- Stage image processing in a temporary file, only replace the requested output on success, and preserve previous output on errors.
- Start launches the selected camera when no file target is selected, or after explicitly selecting a camera. Selecting a target returns Start to file mode. Live explicitly selects camera mode.
- Open DirectShow without unsupported constructor parameters, then set requested properties. Match the physical device path before trying an MSMF fallback. Do not reuse a DirectShow index blindly for MSMF.
- Camera open/read/release all belong to the capture worker. Use queues of size one to retain recent frames and cache the source face. Detect faces on each processed frame so geometry matches that frame.
- Prevent duplicate camera windows/capture instances. Use the current live window for Preview. Stop hides the live window immediately and retains workers until their current calls return, then releases resources and restores controls. Closing the main window waits asynchronously for workers. File Destroy waits for the current file operation to finish.
- Keep live face-mapping separate from file target state, so live processing no longer clears the selected target path. Lock mapping and detection-size changes until live capture stops.
- Bound ONNX session CPU pools to at most four intra-op threads and one inter-op thread. Reuse analyser/swapper sessions. Retain Windows DLL directory handles for the process lifetime.
- Validate CUDA once, preload compatible installed DLLs through ONNX Runtime when a driver is present, probe provider initialization, and fall back to CPU. Never require CUDA. Startup reports compiled providers; models report active providers.
- Existing model files, dependencies, UI design, enhancement/refinement controls and face-selection semantics remain in use. No packages or CUDA toolkits were installed.

## 4. Commands to run
From PowerShell in the project directory:

~~~powershell
.\venv\Scripts\python.exe run.py --execution-provider cpu
~~~

Or double-click run-cpu.bat.

Regression checks:

~~~powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -q
.\venv\Scripts\python.exe tests\verify_pipeline.py
~~~

Opt-in physical camera check (uses diagnostics/source.png, opens camera 0, displays GUI for 25 seconds, stops and reopens the camera):

~~~powershell
.\venv\Scripts\python.exe tests\verify_camera.py
~~~

Set DLC_DEBUG=1 before launch for live frame shape/face count/pixel-change diagnostics.

## 5. How to test image mode
Select a face, then Select a target. Keep opacity above zero. Click Start and choose output.png. The verified result opens automatically in Preview. The leftmost face is used in single-face mode; Many faces swaps all detected faces; Map faces uses the existing mapping dialog. No-face or unchanged-result errors leave an earlier output intact. You can compare diagnostics/target.png and diagnostics/output.png for the automated example (the last result exercises explicit mapping).

## 6. How to test live mode
Select a source face. Select Integrated Camera in the dropdown (even if it is already the current item), then click Start, or use Live. Begin at 640x480; detection size 320 can reduce detector cost for nearby faces. The live window displays processed frames. Preview brings that same window forward. Stop closes the live stream without closing the main app. Wait until Start/Live are enabled again, then restart to verify release. Closing the main app also stops capture safely.

## 7. CUDA status
Unavailable on the inspected Intel-only system. Installing CUDA libraries would not add an NVIDIA GPU. CUDA requests now produce one fallback diagnostic and continue on CPU. ONNX Runtime documents the NVIDIA CUDA/cuDNN requirements and preload_dlls API at https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html.

## 8. CPU fallback and test evidence
- All 13 existing unittest tests pass; compilation and git diff whitespace checks pass.
- Actual run.py GUI opened and closed successfully outside the sandbox.
- Integration: saved output has different pixels, renders in Qt Preview, and GUI timers keep firing during processing.
- Single-face, many-face and explicit mapping paths pass with the real models. No-face target preserves previous output.
- Simulated live frames are continuously swapped; duplicate starts use one capture; source/analyser/swapper objects are reused; failure cleanup passes.
- Physical Integrated Camera opens on DirectShow index 0 at 640x480/30 fps. A processed frame detected one face and changed 90,354 channel values; a subsequent frame changed 125,115. The GUI timer fired 753 times during the 25-second check. These changes were measured against each frame before swapping, without FPS text.
- Stop finished both workers and the same camera reopened successfully. No webcam images were saved.
- Final bundled-media check: cold single-image pipeline 8.87s; warm two-face pipeline 5.76s; warm mapped single-face pipeline 2.31s. These are sample-specific measurements, not a direct benchmark against the reported 41-second user image.

## 9. Remaining warnings and limits
Continuous live processing works, but measured CPU swapping takes roughly 2–3 seconds per face on this environment. This does not meet smooth 30-fps face-swapping performance. The camera's 30-fps capture rate is different from processed output FPS. Bounded queues avoid accumulated delay; the GUI remains responsive.

ONNX can emit an unused-initializer warning for buff2fs in the existing model. It is non-fatal. A third-party Albumentations update-check warning appeared in restricted tests. CUDA diagnostics are expected if CUDA is requested on this machine; use the CPU command for ordinary operation.

Camera access is blocked inside the restricted execution sandbox. Physical webcam tests succeeded outside it, so run the app normally on Windows.
