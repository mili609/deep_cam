# Deep-Live-Cam: Run Instructions

This guide covers Windows, macOS, and Linux. Run commands from the repository root. Python 3.11 is the documented baseline and the version used by CI.

## 1. Prerequisites

- Git and Python 3.11 with pip and virtual-environment support.
- A graphical desktop and a webcam for live mode.
- FFmpeg and ffprobe on PATH for video-file processing.
- Internet access for dependencies and initial model downloads; allow several GB of disk space.
- If InsightFace needs to build from source: Microsoft C++ Build Tools on Windows, Xcode Command Line Tools on macOS, or a C/C++ compiler and Python development headers on Linux.

## 2. Clone the repository

The following commands work in PowerShell, bash, and zsh:

```sh
git clone https://github.com/mili609/deep_cam.git
cd deep_cam
```

## 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash or zsh)**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

After activation, use `python` in the commands below. Confirm the interpreter:

```sh
python --version
```

Without activation, use the platform's virtual-environment interpreter:

| Platform | Interpreter path relative to repository root |
| --- | --- |
| Windows | `venv\Scripts\python.exe` |
| macOS / Linux | `./venv/bin/python` |

If PowerShell blocks activation, invoke its interpreter directly; no execution-policy change is required.

## 4. Install dependencies

With the virtual environment activated, on any platform:

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the repository's dependency versions. Its environment markers select `onnxruntime-gpu` on Windows/Linux and `onnxruntime` on macOS, with separate Apple Silicon and Intel Mac versions. Avoid installing competing ONNX Runtime distributions into the same environment. Torch is optional in the current code and is not required by this installation command.

## 5. Model setup

The app checks and downloads missing models when their processors initialize:

- Face swapping: `models/inswapper_128.onnx` or `models/inswapper_128_fp16.onnx`.
- Face analysis: the `buffalo_l` pack in `~/.insightface/models/buffalo_l` (`~` means your home directory on each platform).
- Optional enhancers: their corresponding ONNX models in `models/`.

For manual setup, use the project's [model repository](https://huggingface.co/hacksider/deep-live-cam/tree/main) and [model instructions](models/instructions.txt). Keep downloaded weights local; they are excluded from Git. Allow the first startup to finish downloading before evaluating performance.

## 6. Run on Windows

In PowerShell, after activation:

```powershell
python run.py --execution-provider cpu
```

For an NVIDIA GPU with a working CUDA runtime:

```powershell
python run.py --execution-provider cuda
```

## 7. Run on macOS

In bash/zsh, after activation:

```bash
python run.py --execution-provider cpu
```

Apple Silicon support is implemented in this repository, including CoreML provider configuration and model optimizations. If the installed ONNX Runtime exposes CoreML, use:

```bash
python run.py --execution-provider coreml
```

Intel Macs may also expose CoreML; use it only when available. CUDA is not a supported macOS option. Actual acceleration depends on successful model-session initialization; CPU remains the fallback.

## 8. Run on Linux

In bash/zsh, after activation:

```bash
python run.py --execution-provider cpu
```

For an NVIDIA GPU with a working CUDA runtime:

```bash
python run.py --execution-provider cuda
```

Use a desktop session with camera access; a headless server cannot display the GUI.

## 9. GPU and provider verification

On every platform, inspect the providers compiled into the installed runtime:

```sh
python -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
python run.py --help
```

The CLI lists the execution-provider choices available in that installation. A listed provider does not prove GPU initialization succeeded: check the terminal's model-session providers and the live UI's actual provider.

**NVIDIA CUDA (Windows/Linux):** requires an NVIDIA GPU, its working driver, and CUDA/cuDNN libraries compatible with the installed ONNX Runtime. Windows requirements request the CUDA/cuDNN pip extras; Linux requirements install ONNX Runtime GPU, so compatible runtime libraries must also be available. Python packages do not replace the NVIDIA driver.

With NVIDIA driver tools installed:

```sh
nvidia-smi
python tests/verify_cuda.py
```

The verification script exercises a small ONNX model; also confirm `CUDAExecutionProvider` on the application's actual model sessions. On macOS, use the provider-list command above to check for `CoreMLExecutionProvider`. Return to CPU mode if an accelerator cannot initialize.

## 10. Live webcam and file processing

1. Select a clear source face image in the left image control.
2. Select the physical webcam from the camera list.
3. Set Transparency to 100% and Mouth Mask to 0 for an initial fully swapped preview. Leave enhancers off initially.
4. Click **Live** or **Start**. The webcam supplies the target frames; a static target image is unnecessary.
5. Keep your face visible and well lit. Check the preview, measured FPS, and actual provider.
6. Click **Stop** or close the live preview to release the camera.

Live mode caches the selected source face and detects target faces on each processed webcam frame. Terminal diagnostics report source detection, target detection, and whether a swap was applied. Frames without a detected face may show the original camera image with `No target face detected` logged.

For file processing, select a source image and a target image/video, then use **Preview** or **Process file**. **Start** opens live mode in this version.

## 11. Troubleshooting

| Issue | Action |
| --- | --- |
| Camera does not open | Close other camera apps, select the correct device, and check Windows/macOS camera permissions or Linux device access. |
| No face detected | Improve lighting, move closer, or choose a clearer source image. |
| Original face remains | Check Transparency and Mouth Mask; inspect the terminal for model, embedding, inference, or unchanged-frame errors. |
| Low FPS | Verify the actual provider, disable enhancers, and try 640 x 480 capture with detection size 320. |
| CUDA initialization fails | Check the NVIDIA driver and runtime libraries. Missing `nvcuda.dll` on Windows indicates a driver problem. Use CPU while resolving it. |
| CoreML unavailable | Check the installed ONNX Runtime's providers and platform dependencies; use CPU. |
| Dependency build/install fails | Confirm Python 3.11 and the platform's native build tools; retain the repository's dependency pins. |
| Model download fails | Check network access and certificate trust, or install the model files manually. |
| GUI fails on Linux | Use a graphical session and install the system libraries required by Qt/OpenCV. |

## 12. Verification and tests

Run these commands after activating the virtual environment on any platform:

```sh
python run.py --help
python -m compileall -q run.py tkinter_fix.py modules tests benchmark_pipeline.py
python -m unittest discover -s tests -p test_live_swap.py
python -m unittest discover -s tests -p test_swap_failures.py
python -m unittest discover -s tests -p test_runtime.py
```

To reproduce the Ruff CI check:

```sh
python -m pip install ruff==0.15.7
python -m ruff check .
```

Optional integration checks (require model files; may take longer):

```sh
python tests/verify_pipeline.py
python tests/verify_live.py
python tests/verify_camera.py
```

Run `verify_pipeline.py` first to generate the test images in `diagnostics/`. `verify_live.py` uses controlled camera input. `verify_camera.py` opens physical camera 0, uses the generated source image, measures FPS, and stops after verification. These scripts write local diagnostic artifacts.

Windows startup/import and controlled-input pipeline checks have been run locally. macOS/Linux commands and Apple Silicon support were verified against repository code; they have not been runtime-tested on those platforms in this session. GPU performance and physical webcam results must be verified on the target machine.
