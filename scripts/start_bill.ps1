# Start BILL stack: both vLLM servers (WSL, via systemd user units) + FastAPI app (Windows).
# Idempotent -- skips any service that is already healthy.
#
# Why systemd and not `wsl.exe ... bash start_vllm.sh &`:
# that form returns immediately and WSL2 tears down the session's process tree,
# so the servers die as soon as the launcher exits (verified: ports go to 000,
# VRAM drops to idle, /tmp output vanishes because /tmp is RAM-backed).
# systemd user units are owned by the distro's user manager instead, so they
# survive every wsl.exe client exiting -- see config/braind-vllm-*.
$ErrorActionPreference = "SilentlyContinue"

$Distro  = "Ubuntu"
$WslRepo = "/mnt/d/AI/braind"
$RepoDir = "D:\AI\braind"
$AppPort = 8010
$DeadlineMinutes = 15

function Test-Url($url) {
    try { (Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5).StatusCode -eq 200 } catch { $false }
}

# --- 1. vLLM servers inside WSL, supervised by systemd ---
$chatUp  = Test-Url "http://127.0.0.1:8001/v1/models"
$embedUp = Test-Url "http://127.0.0.1:8002/v1/models"
if ($chatUp -and $embedUp) {
    Write-Host "[start_bill] vLLM already up"
} else {
    # Units must exist before they can be started. Installing is idempotent and
    # does NOT start anything, so it is safe to run here.
    $known = wsl.exe -d $Distro -e systemctl --user list-unit-files "braind-vllm*" --no-legend 2>$null
    if (-not ($known -match "braind-vllm\.target")) {
        Write-Host "[start_bill] systemd units not installed -- installing ..."
        & "$RepoDir\scripts\install_units.ps1"
    }

    Write-Host "[start_bill] starting vLLM via systemd (braind-vllm.target) ..."
    wsl.exe -d $Distro -e systemctl --user start braind-vllm.target
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "[start_bill] systemctl start failed (exit $LASTEXITCODE). Recent journal:"
        wsl.exe -d $Distro -e journalctl --user -u braind-vllm-embed.service -u braind-vllm-llm.service -n 30 --no-pager
        exit 1
    }
}

# --- 2. Wait for both vLLM endpoints (model load takes minutes on first run) ---
$deadline = (Get-Date).AddMinutes($DeadlineMinutes)
while ((Get-Date) -lt $deadline) {
    if ((Test-Url "http://127.0.0.1:8001/v1/models") -and (Test-Url "http://127.0.0.1:8002/v1/models")) { break }
    Start-Sleep -Seconds 10
}

if (-not ((Test-Url "http://127.0.0.1:8001/v1/models") -and (Test-Url "http://127.0.0.1:8002/v1/models"))) {
    # The old script fell through here silently and then started the app against
    # dead model servers. Fail loudly instead, with the logs needed to diagnose.
    Write-Warning "[start_bill] vLLM did not become ready within $DeadlineMinutes minutes."
    Write-Warning "[start_bill] unit states:"
    wsl.exe -d $Distro -e systemctl --user is-active braind-vllm-embed.service braind-vllm-llm.service
    Write-Warning "[start_bill] last journal lines:"
    wsl.exe -d $Distro -e journalctl --user -u braind-vllm-embed.service -u braind-vllm-llm.service -n 40 --no-pager
    exit 1
}
Write-Host "[start_bill] vLLM ready (chat :8001, embed :8002)"

# --- 3. FastAPI app (Windows venv, port from .env) ---
if (-not (Test-Url "http://127.0.0.1:$AppPort/docs")) {
    Write-Host "[start_bill] launching BILL app on :$AppPort ..."
    Start-Process -WindowStyle Hidden -FilePath "$RepoDir\.venv\Scripts\python.exe" `
        -ArgumentList "-m","uvicorn","api.main:app","--host","127.0.0.1","--port","$AppPort" `
        -WorkingDirectory "$RepoDir" `
        -RedirectStandardOutput "$RepoDir\logs\app-stdout.log" `
        -RedirectStandardError "$RepoDir\logs\app-stderr.log"
} else {
    Write-Host "[start_bill] app already up"
}

# --- 4. Wait for the app, then report ---
$appDeadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $appDeadline) {
    if (Test-Url "http://127.0.0.1:$AppPort/docs") { break }
    Start-Sleep -Seconds 5
}
if (-not (Test-Url "http://127.0.0.1:$AppPort/docs")) {
    Write-Warning "[start_bill] app did not answer on :$AppPort -- see logs\app-stderr.log"
    Get-Content "$RepoDir\logs\app-stderr.log" -Tail 30 -ErrorAction SilentlyContinue
    exit 1
}

Start-Sleep -Seconds 5
& "$RepoDir\scripts\status_bill.ps1"
