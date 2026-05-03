@echo off
echo ============================================================
echo  crop_detection conda ortami kuruluyor...
echo ============================================================

call conda env list | findstr /C:"crop_detection" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Mevcut ortam siliniyor...
    call conda env remove -n crop_detection -y
)

call conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo HATA: Ortam olusturulamadi!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BASARILI! Ortami aktif etmek icin:
echo    conda activate crop_detection
echo  Sonra pipeline'i calistir:
echo    python check_dataset.py
echo    python main.py
echo ============================================================
pause
