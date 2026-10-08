# Running Deep-Live-Cam on Windows

## Launch this existing installation

Open PowerShell in the Deep-Live-Cam folder, then run:

```powershell
.\run-cpu.bat
```

For an NVIDIA GPU with a working CUDA driver:

```powershell
.\run-cuda.bat
```

These launchers use the project's virtual environment. Equivalent commands:

```powershell
.\venv\Scripts\python.exe run.py --execution-provider cpu
.\venv\Scripts\python.exe run.py --execution-provider cuda
```

## Start the live face swap

1. Select a clear source image with a visible face using the left Source face control.
2. Choose your physical webcam, such as Integrated Camera, from the camera list.
3. Set Transparency to 100% for a fully swapped face. Leave Mouth Mask at 0 initially.
4. Leave face enhancers off initially to reduce processing time.
5. Click Live (or Start in the updated UI).
6. Keep your face visible and well lit. The live preview should show the selected source face on your camera face.
7. Read the LIVE status for measured FPS and the actual execution provider.
8. Click Stop or close the live preview to release the camera.

The webcam supplies the live target frames; no static target image is needed. In the updated local pipeline, the source face is extracted once and cached until the selected source path changes. Target faces are detected on each processed webcam frame. File face mappings are ignored during live mode.

## Setup on another Windows computer

Use Python 3.11 and open PowerShell in the repository folder:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Keep model files in the models folder. The app's model downloader checks the required files; first launch may require internet access. Follow README.md for the full installation prerequisites and model setup. Installing InsightFace may require Microsoft C++ build tools if pip needs to build it locally.

## CUDA checks

CUDA requires an NVIDIA GPU, its working Windows display driver, and a GPU-enabled ONNX Runtime installation. A listed provider alone does not prove that CUDA initialized.

```powershell
nvidia-smi
.\venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
.\venv\Scripts\python.exe tests\verify_cuda.py
```

Use the last command when that verification script is included in your checkout. On a working CUDA setup, the live model's actual provider should be CUDAExecutionProvider. If the status says CPU, read the terminal's CUDA initialization error.

If nvcuda.dll is missing, repair or install the NVIDIA display driver for your GPU. Python packages cannot replace the NVIDIA driver. If the computer has no NVIDIA GPU, use the CPU launcher.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Camera cannot open | Close other camera apps, check Windows camera permissions for desktop apps, and confirm the selected device. |
| No target face detected | Face the camera, improve lighting, and move closer. Original camera frames are allowed while no face is detected. |
| No face in source | Select a clear image with a larger, unobstructed face. |
| Original face remains | Check Transparency is 100%; inspect the terminal for source embedding, model, inference, or unchanged-frame errors. |
| Very low FPS | Check the actual provider. Disable enhancers and try 640×480 capture and 320 detection size. Measure again with a face visible. |
| Live processing fails | Read the full exception in the terminal; correct it and restart live mode. |

The first processed frame in the updated pipeline logs:

```text
source face detected = YES/NO
target face detected = YES/NO
swap applied = YES/NO
```

## Verification scripts in the updated local checkout

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_live_swap.py
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_swap_failures.py
.\venv\Scripts\python.exe -m unittest discover -s tests -p test_runtime.py
.\venv\Scripts\python.exe tests\verify_live.py
.\venv\Scripts\python.exe tests\verify_camera.py
```

verify_live.py uses controlled camera input with real models. verify_camera.py opens physical camera 0 and checks swapped frames and FPS; it uses diagnostics/source.png as its source and stops after verification.

Latest local results: 10 regression tests passed. The controlled-input live GUI test passed at 0.35 FPS on CPU. Physical webcam verification could not open camera 0, and CUDA initialization failed because nvcuda.dll was missing in the test environment. Physical webcam replacement and CUDA FPS have not yet been verified.

These instructions describe the updated local checkout. Uploading this file alone does not install the app fixes or verification scripts in another repository.
