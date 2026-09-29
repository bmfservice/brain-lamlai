# Stop the whole BILL stack: vLLM servers in WSL + the FastAPI app on Windows.
#
# The WSL side is delegated to scripts/stop_vllm.sh, which handles both the
# systemd units and the legacy ad-hoc holder, validates every PID through /proc
# before signalling it, and never uses `wsl --terminate` (that would kill
# unrelated sessions).
#
# The app is found by the port it listens on rather than by matching a command
# line: `Get-Process` in Windows PowerShell 5.1 has no CommandLine property, so
# the `Where-Object { $_.CommandLine -like '*uvicorn*' }` filter used by the old
# Makefile target silently matched nothing.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop_bill.ps1
$ErrorActionPreference = "Continue"

$Distro   = "Ubuntu"
$WslRepo  = "/mnt/d/AI/braind"
$AppPort  = 8010

Write-Host "[stop_bill] stopping BILL stack ..."

# --- 1. vLLM inside WSL (systemd units and/or legacy holder) ---
Write-Host "[stop_bill] stopping vLLM in WSL ..."
wsl.exe -d $Distro -e bash "$WslRepo/scripts/stop_vllm.sh"
$wslExit = $LASTEXITCODE
if ($wslExit -ne 0) {
    Write-Warning "[stop_bill] stop_vllm.sh reported exit $wslExit (see output above)"
}

# --- 2. FastAPI app on Windows ---
$conns = Get-NetTCPConnection -LocalPort $AppPort -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    $pids = $conns | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $pids) {
        if ($procId -eq 0) { continue }
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $proc) { continue }
        # Only kill python: a mis-identified listener must never be terminated.
        if ($proc.ProcessName -match '^python') {
            Write-Host "[stop_bill] stopping app pid $procId ($($proc.ProcessName)) on :$AppPort"
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        } else {
            Write-Warning "[stop_bill] :$AppPort is held by $($proc.ProcessName) (pid $procId) -- not killing"
        }
    }
} else {
    Write-Host "[stop_bill] no listener on :$AppPort (app already stopped)"
}

Start-Sleep -Seconds 3

# --- 3. report ---
Write-Host ""
Write-Host "[stop_bill] final state:"
$checks = @(
    @{ Name = "vLLM chat  :8001"; Url = "http://127.0.0.1:8001/v1/models" },
    @{ Name = "vLLM embed :8002"; Url = "http://127.0.0.1:8002/v1/models" },
    @{ Name = "BILL app   :8010"; Url = "http://127.0.0.1:8010/docs" }
)
$allDown = $true
foreach ($c in $checks) {
    try {
        $r = Invoke-WebRequest -Uri $c.Url -UseBasicParsing -TimeoutSec 5
        Write-Host "  STILL UP   $($c.Name) (HTTP $($r.StatusCode))"
        $allDown = $false
    } catch {
        Write-Host "  down       $($c.Name)"
    }
}

$units = wsl.exe -d $Distro -e systemctl --user is-active braind-vllm-embed.service braind-vllm-llm.service 2>$null
Write-Host "  systemd units: $($units -join ', ')"

if ($allDown -and $wslExit -eq 0) {
    Write-Host ""
    Write-Host "[stop_bill] all services stopped"
    exit 0
} else {
    Write-Host ""
    Write-Warning "[stop_bill] something is still listening -- re-run, or check the output above"
    exit 1
}
