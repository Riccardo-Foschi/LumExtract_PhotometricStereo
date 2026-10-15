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
if exist "%VENV_PY%" (
	"%VENV_PY%" -c "import sys" >nul 2>&1
	if errorlevel 1 (
		echo - venv non funzionante ^(interprete base mancante^). Lo ricreo.
		rmdir /s /q "%VENV_DIR%"
	) else (
		echo - venv trovato in %VENV_DIR%
		goto :venv_ready
	)
)

echo - Creo %VENV_DIR%
set "PY_VER="
for %%V in (3.12 3.11 3.10 3.13 3.14 3.15) do (
	if not defined PY_VER (
		py -%%V -c "import sys" >nul 2>&1 && set "PY_VER=%%V"
	)
)
if not defined PY_VER (
	echo ERRORE: nessun Python 3.10-3.15 trovato ^(installa da python.org^).
	goto :error
)
echo - Uso Python %PY_VER%
py -%PY_VER% -m venv "%VENV_DIR%"
if errorlevel 1 goto :error

:venv_ready

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
"%VENV_PY%" -m PyInstaller --noconfirm --clean --workpath "%TEMP%\luminance_app_build" --distpath "%TEMP%\luminance_app_dist" luminance_app.spec
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