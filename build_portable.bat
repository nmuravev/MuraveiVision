@echo off
REM MuraveiVision Portable build (PyInstaller, onedir)
setlocal

echo.
echo [1/6] Cleaning old build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [2/6] Building via PyInstaller (onedir, windowed)...
python -m PyInstaller MuraveiVision.spec --noconfirm
if errorlevel 1 (
    echo ERROR: PyInstaller failed
    exit /b 1
)

echo.
echo [3/6] Copying models to dist\MuraveiVision\models\...
if not exist dist\MuraveiVision\models mkdir dist\MuraveiVision\models
if exist models\*.onnx (
    copy /y models\*.onnx dist\MuraveiVision\models\
) else (
    echo WARNING: No ONNX files in models\
)

echo.
echo [4/6] Creating output\ and copying .env.example...
if not exist dist\MuraveiVision\output mkdir dist\MuraveiVision\output
if exist .env.example copy /y .env.example dist\MuraveiVision\.env.example

echo.
echo [5/6] Copying README_portable.txt...
if exist README_portable.txt copy /y README_portable.txt dist\MuraveiVision\README_portable.txt

echo.
echo [6/6] Unpacking Tcl/Tk libraries for Python 3.14 compatibility...
powershell -Command "if (Test-Path 'C:\Python314\tcl\libtcl9.0.4.zip') { Expand-Archive -Path 'C:\Python314\tcl\libtcl9.0.4.zip' -DestinationPath 'dist\MuraveiVision\_internal\_tcl_data' -Force; Expand-Archive -Path 'C:\Python314\tcl\libtk9.0.4.zip' -DestinationPath 'dist\MuraveiVision\_internal\_tk_data' -Force; Write-Host 'Tcl/Tk libraries unpacked successfully' } else { Write-Host 'WARNING: Tcl/Tk zip archives not found - Python may not be 3.14' }"

echo.
echo BUILD COMPLETE: dist\MuraveiVision\
echo.
echo Verify: dist\MuraveiVision\MuraveiVision.exe --selftest
echo.
endlocal
