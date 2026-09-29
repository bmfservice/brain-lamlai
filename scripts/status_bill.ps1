# BILL services status helper
# Shows HTTP health, the systemd unit states that own vLLM, and VRAM.
#
# The unit state matters because HTTP 200 alone cannot tell a durable stack from
# a temporary one: if the servers answer but the units are inactive, something
# else is holding them (e.g. an ad-hoc `wsl.exe` session) and they will die when
# that session closes.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\status_bill.ps1
$ErrorActionPreference = "SilentlyContinue"

$Distro = "Ubuntu"

function Check-Http($name, $url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 10
        if ($r.StatusCode -eq 200) { Write-Host "OK   $name ($url)"; return $true }
        Write-Host "FAIL $name (HTTP $($r.StatusCode))"; return $false
    } catch { Write-Host "DOWN $name ($url)"; return $false }
}

Write-Host "Services:"
$chatUp  = Check-Http "vLLM chat   :8001" "http://127.0.0.1:8001/v1/models"
$embedUp = Check-Http "vLLM embed  :8002" "http://127.0.0.1:8002/v1/models"
$appUp   = Check-Http "BILL app    :8010" "http://127.0.0.1:8010/docs"

# --- systemd unit states (who actually owns vLLM) ---
# Call systemctl directly via `wsl.exe -e` (no `bash -c` wrapper): nesting a bash
# for-loop with printf "%-30s" and $(...) inside a PowerShell single-quoted
# string gets mangled by native argument passing and yields a syntax error.
Write-Host ""
Write-Host "systemd user units (WSL):"
$unitNames = @("braind-vllm-embed.service", "braind-vllm-llm.service", "braind-vllm.target")
$states = @()
foreach ($u in $unitNames) {
    $state = (wsl.exe -d $Distro -e systemctl --user is-active $u 2>$null) -join ""
    $state = "$state".Trim()
    if ([string]::IsNullOrWhiteSpace($state)) { $state = "unknown" }
    Write-Host ("  {0,-30} {1}" -f $u, $state)
    $states += $state
}

if (($chatUp -or $embedUp) -and ($states -notcontains "active")) {
    Write-Host ""
    Write-Warning "vLLM is answering but the systemd units are NOT active."
    Write-Warning "The servers are held by an ad-hoc WSL session and will die when it closes."
    Write-Warning "Fix: powershell -File scripts\stop_bill.ps1 ; powershell -File scripts\start_bill.ps1"
}

# --- VRAM (CLAUDE.md hard constraint: <=19GB used and >=5GB free per card) ---
# nvidia-smi lives in /usr/lib/wsl/lib and is NOT on the PATH that `wsl.exe -e`
# uses (it fails with execvpe), so go through bash, which has the WSL PATH.
$nvidia = wsl.exe -d $Distro -e bash -c "nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader" 2>$null
if ($nvidia) {
    Write-Host ""
    Write-Host "VRAM:"
    $nvidia | ForEach-Object {
        $parts = $_ -split ','
        if ($parts.Count -ge 5) {
            $usedMiB = [int](($parts[2].Trim() -replace '[^0-9]',''))
            $freeMiB = [int](($parts[3].Trim() -replace '[^0-9]',''))
            $flag = if ($usedMiB -le 19456 -and $freeMiB -ge 5120) { "OK " } else { "CHECK" }
            Write-Host ("  {0} GPU{1} {2} used={3}MiB free={4}MiB (min free >= 5120MiB)" -f $flag, $parts[0].Trim(), $parts[1].Trim(), $usedMiB, $freeMiB)
        } else {
            Write-Host "  $_"
        }
    }
}
