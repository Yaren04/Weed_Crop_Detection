# PowerShell'de calistir:
#   Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
#   .\setup_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "`n" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  crop_detection conda ortami kuruluyor..." -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "`n"

# Conda'nin kurulu oldugunu kontrol et
try {
    $null = conda --version 2>$null
} catch {
    Write-Host "HATA: Conda bulunamadi! Anaconda/Miniconda yuklu mu?" -ForegroundColor Red
    exit 1
}

# Ortamı kaldır (var ise)
$envExists = conda env list | Select-String "crop_detection"
if ($envExists) {
    Write-Host "Mevcut ortam siliniyor..." -ForegroundColor Yellow
    conda env remove -n crop_detection -y
}

# Yeni ortam oluştur
Write-Host "Yeni ortam olusturuluyor..." -ForegroundColor Cyan
conda env create -f environment.yml

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nHATA: Ortam olusturulamadi!" -ForegroundColor Red
    exit 1
}

Write-Host "`n" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "  BASARILI! Simdi:" -ForegroundColor Green
Write-Host ("=" * 60) -ForegroundColor Green
Write-Host "`n"
Write-Host "1. Ortami aktif et:" -ForegroundColor Cyan
Write-Host "   conda activate crop_detection`n" -ForegroundColor White

Write-Host "2. Dataset'i koy:" -ForegroundColor Cyan
Write-Host "   - Goruntuleri: data/images/" -ForegroundColor White
Write-Host "   - XML'leri: data/annotations/`n" -ForegroundColor White

Write-Host "3. Dataset'i dogrula:" -ForegroundColor Cyan
Write-Host "   python check_dataset.py`n" -ForegroundColor White

Write-Host "4. Pipeline'i calistir:" -ForegroundColor Cyan
Write-Host "   python main.py`n" -ForegroundColor White
