@echo off
setlocal
title Luminance Extractor for Photometric Stereo Build

echo.
echo ==================================================
echo Luminance Extractor for Photometric Stereo - Auto Build
echo ==================================================
echo.

REM Move to the folder where this .bat file is located.
cd /d "%~dp0"

set "SCRIPT_DIR=%~dp0"
set "VENV_DIR=%SCRIPT_DIR%venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo [1/4] Verifica virtual environment locale...
if not exist "%VENV_PY%" (
	echo - venv non trovato. Creo %VENV_DIR%
	where py >nul 2>&1
	if not errorlevel 1 (
		py -3 -m venv "%VENV_DIR%"
	) else (
		python -m venv "%VENV_DIR%"
	)
	if errorlevel 1 goto :error
) else (
	echo - venv trovato in %VENV_DIR%
)

echo.
echo [2/4] Verifica pip e aggiorna strumenti base...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :error

echo.
echo [3/4] Verifica librerie richieste...
call :EnsurePackage numpy numpy
if errorlevel 1 goto :error
call :EnsurePackage rawpy rawpy
if errorlevel 1 goto :error
call :EnsurePackage cv2 opencv-python
if errorlevel 1 goto :error
call :EnsurePackage PyQt5 PyQt5
if errorlevel 1 goto :error
call :EnsurePackage PyInstaller pyinstaller
if errorlevel 1 goto :error

echo.
echo [4/4] Build con PyInstaller...
if not exist "%TEMP%\luminance_app_dist" mkdir "%TEMP%\luminance_app_dist"
if exist "%SCRIPT_DIR%dist\Luminance_Extractor_for_Photometric_Stereo.exe" del /f /q "%SCRIPT_DIR%dist\Luminance_Extractor_for_Photometric_Stereo.exe"
"%VENV_PY%" -m PyInstaller --noconfirm --workpath "%TEMP%\luminance_app_build" --distpath "%TEMP%\luminance_app_dist" luminance_app.spec
if errorlevel 1 goto :error

if not exist "%SCRIPT_DIR%dist" mkdir "%SCRIPT_DIR%dist"
copy /y "%TEMP%\luminance_app_dist\Luminance_Extractor_for_Photometric_Stereo.exe" "%SCRIPT_DIR%dist\Luminance_Extractor_for_Photometric_Stereo.exe" >nul
if errorlevel 1 goto :error

echo.
echo ==================================================
echo Build completata con successo.
echo Eseguibile disponibile in dist\
echo ==================================================
echo.
pause
exit /b 0

:EnsurePackage
set "IMPORT_NAME=%~1"
set "PIP_NAME=%~2"
"%VENV_PY%" -c "import %IMPORT_NAME%" >nul 2>&1
if errorlevel 1 (
	echo - Installo %PIP_NAME%...
	"%VENV_PY%" -m pip install "%PIP_NAME%"
	if errorlevel 1 exit /b 1
) else (
	echo - %PIP_NAME% gia disponibile.
)
exit /b 0

:error
echo.
echo ==================================================
echo ERRORE: build non completata.
echo Controlla i messaggi sopra.
echo ==================================================
echo.
pause
exit /b 1