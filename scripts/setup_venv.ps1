# Fase 0 — menyiapkan lingkungan Python 3.11/3.12 untuk model .keras.
# Jalankan dari root proyek:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1
#
# TensorFlow BELUM mendukung Python 3.14, jadi venv dibuat dengan Python 3.11.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Uji kandidat interpreter: harus benar-benar bisa dieksekusi DAN mencetak tanda.
# Catatan: stderr native command dibungkus jadi ErrorRecord oleh Windows PowerShell,
# jadi ErrorActionPreference diturunkan sementara selama probing.
function Test-Python([string]$cmd, [string[]]$argList) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { return $false }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = (& $cmd @argList -c "print('VENV_PROBE_OK')" 2>&1 | Out-String)
        return (($LASTEXITCODE -eq 0) -and ($out -match "VENV_PROBE_OK"))
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prev
    }
}

$candidates = @(
    @{ cmd = "py"; args = @("-3.12") },
    @{ cmd = "py"; args = @("-3.11") },
    @{ cmd = "python3.12"; args = @() },
    @{ cmd = "python3.11"; args = @() }
)
$sel = $null
foreach ($c in $candidates) {
    if (Test-Python $c.cmd $c.args) { $sel = $c; break }
}
if (-not $sel) {
    Write-Host "[!] Python 3.11/3.12 tidak ditemukan. Install dulu (mis. winget install Python.Python.3.12)" -ForegroundColor Red
    exit 1
}

Write-Host "== Membuat .venv dengan $($sel.cmd) $($sel.args -join ' ') ==" -ForegroundColor Cyan
& $sel.cmd @($sel.args) -m venv .venv
if ($LASTEXITCODE -ne 0 -or -not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[!] Gagal membuat .venv (exit=$LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "== Meng-upgrade pip ==" -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Gagal upgrade pip (exit=$LASTEXITCODE)" -ForegroundColor Red; exit 1 }

Write-Host "== Menginstall requirements.txt (unduhan besar) ==" -ForegroundColor Cyan
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Gagal install requirements (exit=$LASTEXITCODE)" -ForegroundColor Red; exit 1 }

Write-Host "== Cek versi ==" -ForegroundColor Cyan
& $venvPy -c "import keras, tensorflow as tf; print('keras', keras.__version__, '| tensorflow', tf.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Host "[!] Gagal import keras/tensorflow" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Selesai. Langkah berikutnya:" -ForegroundColor Green
Write-Host "  1. .venv\Scripts\activate"
Write-Host "  2. copy .env.example .env   (isi CLASS_INDICES_PATH)"
Write-Host "  3. python scripts\verify_model.py"
Write-Host "  4. pytest"
Write-Host "  5. uvicorn app.main:app --reload"
