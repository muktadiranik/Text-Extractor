# Camera Number Reader

Read printed English text, a single row of printed or handwritten digits, or a phone number in the format `01234-567890`. Center the text or number in the yellow box and hold it steady. The preview displays the result after two consecutive matching samples. Leading zeros and the phone number's hyphen are preserved. With the preview focused, press **Q** to quit.

If a visible number is missed, press **S** with the preview focused. This saves only the area inside the yellow box to `ocr-sample.png` in the project folder, replacing the previous sample. You can pass that file to `--image` to test the same crop.

## Handwritten digits

Launch the app normally to read handwriting. Write in dark ink on light paper and leave a visible gap between digits. Digit sequences have no fixed length limit. Keep the whole row visible and each digit clear enough for the camera to read. The app uses the bundled MNIST model through OpenCV when Tesseract misses the digits. Both readers run locally. [ONNX Model Zoo MNIST model](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)

For handwritten text, write separate block letters. The app can enhance faint pencil strokes before running Tesseract.

## Windows PowerShell

Install Python 3.12 and enable the Python launcher in the installer. Install Tesseract 5 with English language data using the Windows installer listed in the installation guide. [Python downloads](https://www.python.org/downloads/), [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html#windows).

From the workspace root, create the Windows environment and launch the app.

```powershell
Set-Location camera-number-reader
py.exe -3.12 -m venv .venv-win
.\.venv-win\Scripts\python.exe -m pip install -r requirements.txt
.\number_reader.cmd
```

The app searches `PATH` and the standard `C:\Program Files\Tesseract-OCR` install folder. If you installed Tesseract elsewhere, set its executable path in the same PowerShell session before launching. Use your actual install path in place of the example.

```powershell
$env:TESSERACT_CMD = 'C:\Tools\Tesseract-OCR\tesseract.exe'
```

```powershell
.\.venv-win\Scripts\python.exe number_reader.py --camera-index 1
.\.venv-win\Scripts\python.exe number_reader.py --image 'C:\path\to\number.png'
```

## Linux (Ubuntu or Debian)

Use Python 3.10 or newer and run the webcam app from a graphical desktop session. Install Python, Tesseract, English language data, and OpenCV's GUI runtime libraries.

```bash
sudo apt update
sudo apt install python3 python3-venv tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0t64 libxcb-cursor0
```

On Ubuntu 22.04 or Debian 12, use `libglib2.0-0` in place of `libglib2.0-0t64`. From the workspace root, run the following commands.

```bash
cd camera-number-reader
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python number_reader.py
```

```bash
./.venv/bin/python number_reader.py --camera-index 1
./.venv/bin/python number_reader.py --image /path/to/number.png
```

## Launch native Windows Python from WSL

Complete the Windows setup above in the project folder on a Windows drive. From the workspace root in WSL, run the following commands. Windows Python opens the webcam and preview through Windows.

```bash
cd camera-number-reader
./.venv-win/Scripts/python.exe "$(wslpath -w "$PWD/number_reader.py")"
```

If Tesseract is in a custom Windows location, export `TESSERACT_CMD` using its Windows path and add the variable to `WSLENV` so Windows Python receives it.

```bash
export TESSERACT_CMD='C:\Tools\Tesseract-OCR\tesseract.exe'
export WSLENV="${WSLENV:+$WSLENV:}TESSERACT_CMD/w"
```

## Still images and automated checks

`--image` reads the entire image and prints the detected text or number in the terminal. Use a crop around the text you want to read. Long text wraps below the yellow box in the preview, and `--image` prints the full reading. Image mode runs without a webcam or preview window. The tests require Tesseract and cover printed text, phone numbers, handwriting, confirmation, and error messages.

Run from `camera-number-reader` on Windows PowerShell.

```powershell
.\.venv-win\Scripts\python.exe -m unittest discover -s tests -v
```

Run from `camera-number-reader` on Linux.

```bash
./.venv/bin/python -m unittest discover -s tests -v
```
