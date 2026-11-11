@echo off
setlocal
set "reader_python=%~dp0.venv-win\Scripts\python.exe"
if not exist "%reader_python%" (
    >&2 echo Windows Python environment is missing. Follow the Windows setup in README.md.
    exit /b 1
)
"%reader_python%" "%~dp0number_reader.py" %*
exit /b %errorlevel%
