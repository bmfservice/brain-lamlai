# Install the BILL vLLM systemd user units into WSL.
#
# Why: `wsl.exe -e bash -c "... &"` returns immediately and WSL2 tears the
# session's process tree down, killing the servers. systemd user units are owned
# by the distro's PID-1 user manager instead, so they survive wsl.exe exiting.
#
# This copies config/braind-vllm-*.{service,target} to ~/.config/systemd/user/,
# reloads the daemon and enables the units WITHOUT starting them, so an already
# running stack is never disturbed. Run scripts/start_bill.ps1 (or
# `make start`) afterwards to actually start them.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_units.ps1
$ErrorActionPreference = "Stop"

$Repo  = "D:\AI\braind"
$WslRepo = "/mnt/d/AI/braind"
$Distro = "Ubuntu"

Write-Host "[install_units] installing systemd user units for BILL vLLM ..."

# Copy the unit files. Single-quoted so PowerShell does not expand $HOME or ~ --
# the tilde must reach bash, which expands it inside the distro.
$copy = 'mkdir -p ~/.config/systemd/user && ' +
        "cp $WslRepo/config/braind-vllm-embed.service " +
        "$WslRepo/config/braind-vllm-llm.service " +
        "$WslRepo/config/braind-vllm.target " +
        '~/.config/systemd/user/ && ' +
        'chmod 644 ~/.config/systemd/user/braind-vllm-*.service ' +
        '~/.config/systemd/user/braind-vllm-*.target && ' +
        "chmod 755 $WslRepo/start_vllm.sh && " +
        'ls -1 ~/.config/systemd/user/braind-vllm-*'
$installed = wsl.exe -d $Distro -e bash -c $copy
if ($LASTEXITCODE -ne 0) { Write-Error "failed to copy unit files"; exit 1 }
$installed | ForEach-Object { Write-Host "  installed: $_" }

# Reload and enable. Enable WITHOUT --now: this must never restart a live stack.
$enable = 'systemctl --user daemon-reload && ' +
          'systemctl --user enable braind-vllm-embed.service braind-vllm-llm.service braind-vllm.target && ' +
          'echo "--- unit-file state ---" && ' +
          'systemctl --user list-unit-files "braind-vllm*"'
wsl.exe -d $Distro -e bash -c $enable
if ($LASTEXITCODE -ne 0) { Write-Error "daemon-reload/enable failed"; exit 1 }

Write-Host ""
Write-Host "[install_units] done. Units are enabled but NOT started."
Write-Host "[install_units] start with:  powershell -File scripts\start_bill.ps1   (or: make start)"
Write-Host ""
Write-Host "[install_units] OPTIONAL -- survive 'wsl --shutdown' / machine reboot."
Write-Host "[install_units] Without it the units still survive closing terminals and"
Write-Host "[install_units] wsl.exe clients, which is the bug this fixes. To also start"
Write-Host "[install_units] them at WSL boot, run this once yourself (needs your password):"
Write-Host "    wsl -d $Distro -u root -- loginctl enable-linger bmf"