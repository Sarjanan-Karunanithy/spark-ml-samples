@echo off
title Music Genre Classifier
color 0A
cls

echo.
echo  =====================================================
echo   MUSIC GENRE CLASSIFIER - Apache Spark MLlib
echo  =====================================================
echo.

:: ─────────────────────────────────────────────────────────
:: CONFIG — update these if your paths are different
:: ─────────────────────────────────────────────────────────
set JAR_PATH=api\build\libs\api-1.0-SNAPSHOT.jar
set PROPS_DIR=%USERPROFILE%
set HTML_FILE=%~dp0music_classifier.html
set PORT=9090

:: ─────────────────────────────────────────────────────────
:: CHECK — JAR exists?
:: ─────────────────────────────────────────────────────────
if not exist "%JAR_PATH%" (
    echo  [ERROR] JAR file not found at: %JAR_PATH%
    echo.
    echo  Please build first by running in terminal:
    echo    gradlew clean build -x test
    echo.
    pause
    exit /b 1
)
echo  [OK] JAR found

:: ─────────────────────────────────────────────────────────
:: CHECK — HTML exists?
:: ─────────────────────────────────────────────────────────
if not exist "%HTML_FILE%" (
    echo  [ERROR] music_classifier.html not found!
    echo          Make sure it is in the same folder as run.bat
    echo.
    pause
    exit /b 1
)
echo  [OK] HTML found

:: ─────────────────────────────────────────────────────────
:: CHECK — Already running?
:: ─────────────────────────────────────────────────────────
echo  [..] Checking port %PORT%...
netstat -ano | findstr ":%PORT%" >nul 2>&1
if %errorlevel% == 0 (
    echo  [OK] App already running on port %PORT%
    goto OPEN_BROWSER
)

:: ─────────────────────────────────────────────────────────
:: START — Launch Spring Boot in background
:: ─────────────────────────────────────────────────────────
echo  [..] Starting API server...
echo.
start "Spark ML API" /MIN java -jar "%JAR_PATH%" --spring.config.location="%PROPS_DIR%\"

:: ─────────────────────────────────────────────────────────
:: WAIT — Until port 9090 responds (max 90 seconds)
:: ─────────────────────────────────────────────────────────
echo  [..] Waiting for API to be ready...
set /a WAITED=0

:WAIT_LOOP
timeout /t 3 /nobreak >nul
set /a WAITED+=3

netstat -ano | findstr ":%PORT%" >nul 2>&1
if %errorlevel% == 0 (
    echo  [OK] API ready after %WAITED% seconds!
    goto OPEN_BROWSER
)

if %WAITED% GEQ 90 (
    echo.
    echo  [ERROR] API did not start in 90 seconds.
    echo          Check the "Spark ML API" window for errors.
    echo.
    pause
    exit /b 1
)

echo         Still starting... %WAITED%s
goto WAIT_LOOP

:: ─────────────────────────────────────────────────────────
:: OPEN — Browser
:: ─────────────────────────────────────────────────────────
:OPEN_BROWSER
echo.
echo  [..] Opening browser...
start ""  http://localhost:9090/index.html

echo.
echo  =====================================================
echo   Ready!
echo.
echo   Paste lyrics into the browser and click
echo   "Classify Genre" to see the pie chart result.
echo.
echo   To STOP: close the "Spark ML API" window
echo  =====================================================
echo.
pause
