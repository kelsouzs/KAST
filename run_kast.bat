@echo off
setlocal enabledelayedexpansion

cd /d "C:\Users\kessi\Videos\IT2024_testes\KAST_v1.0.0_git"

REM Find conda.exe  
set CONDA_EXE=
if exist "C:\ProgramData\Anaconda3\Scripts\conda.exe" (
    set CONDA_EXE=C:\ProgramData\Anaconda3\Scripts\conda.exe
)

if not defined CONDA_EXE (
    start cmd /k "echo ERROR: Conda not found! & pause"
    exit /b 1
)

REM Get conda.bat location
set CONDA_BAT=C:\ProgramData\Anaconda3\condabin\conda.bat

REM Open KAST in a new window with proper conda activation
if exist "!CONDA_BAT!" (
    start "K-talysticFlow 1.0.0" cmd /k ""!CONDA_BAT!" activate ktalysticflow && python main.py"
) else (
    start "K-talysticFlow 1.0.0" cmd /k ""!CONDA_EXE!" activate ktalysticflow && python main.py"
)
