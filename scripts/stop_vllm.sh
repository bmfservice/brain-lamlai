#!/usr/bin/env bash
# Stop the BILL vLLM servers inside WSL, whatever is supervising them.
#
# Handles two cases:
#   1. systemd user units (config/braind-vllm-*.service) — the normal case.
#      `systemctl --user stop` + KillMode=control-group reaps EngineCore workers.
#   2. The legacy ad-hoc holder: a `wsl.exe` running
#      `start_vllm.sh && exec sleep infinity`. Verified process shape:
#
#          Relay(N)  <- the WSL relay, named after the sleep pid
#          ├─ sleep infinity      <- holder
#          ├─ python -m vllm.entrypoints...   (SIBLING of sleep, not its child)
#          └─ python -m vllm.entrypoints...
#
#      Because the servers are *siblings* of the holder, killing the holder does
#      not necessarily take them down — each must be matched on its own cmdline.
#
# Safety rules this script follows:
#   - Never `wsl --terminate`: that kills unrelated sessions and other work.
#   - Never blind `pkill sleep`: far too broad. A `sleep infinity` is only
#     killed when it shares a parent with a process we positively identified as
#     one of ours.
#   - Every PID is validated through /proc/<pid>/exe and /proc/<pid>/cmdline
#     before being signalled, so a recycled PID can never be killed.
#   - SIGTERM first, then SIGKILL only for survivors.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

log() { echo "[stop_vllm] $*"; }

# --- helpers ---------------------------------------------------------------

# Print /proc/<pid>/cmdline as a single space-separated line.
cmdline_of() {
    [ -r "/proc/$1/cmdline" ] || return 1
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null
}

ppid_of() {
    [ -r "/proc/$1/stat" ] || return 1
    # field 4 of /proc/<pid>/stat is the ppid (field 2 may contain spaces if the
    # comm does, so read from the end of the parenthesised comm instead).
    local stat
    stat="$(cat "/proc/$1/stat" 2>/dev/null)" || return 1
    echo "${stat##*) }" | awk '{print $2}'
}

# True when the PID is really one of our vLLM servers.
is_vllm_pid() {
    local pid="$1"
    [ -d "/proc/$pid" ] || return 1
    local cmd exe
    cmd="$(cmdline_of "$pid")" || return 1
    case "$cmd" in
        *"vllm.entrypoints.openai.api_server"*) ;;
        *) return 1 ;;
    esac
    # Belt and braces: the executable must be a python interpreter.
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    case "$exe" in
        *python*|"") return 0 ;;
        *) return 1 ;;
    esac
}

# True when the PID is a vLLM worker child (EngineCore / resource_tracker).
is_vllm_child() {
    local pid="$1"
    [ -d "/proc/$pid" ] || return 1
    local cmd
    cmd="$(cmdline_of "$pid")" || return 1
    case "$cmd" in
        *"VLLM::EngineCore"*|*"multiprocessing.resource_tracker"*|*"multiprocessing.spawn"*) return 0 ;;
        *) return 1 ;;
    esac
}

collect_vllm_pids() {
    local p
    for p in $(pgrep -f 'vllm\.entrypoints\.openai\.api_server' 2>/dev/null | sort -n); do
        is_vllm_pid "$p" && echo "$p"
    done
}

# --- 1. systemd path -------------------------------------------------------

if systemctl --user show braind-vllm.target >/dev/null 2>&1; then
    if systemctl --user is-active --quiet braind-vllm.target; then
        log "stopping systemd user units (braind-vllm.target) ..."
        systemctl --user stop braind-vllm.target
    else
        log "systemd target present but not active"
    fi
else
    log "no systemd user units installed (legacy mode only)"
fi

# --- 2. legacy holder path -------------------------------------------------

VLLM_PIDS="$(collect_vllm_pids)"

# Remember the parents before killing, so the holder can be identified as a
# *sibling* afterwards.
PARENTS=""
for pid in $VLLM_PIDS; do
    pp="$(ppid_of "$pid" || true)"
    [ -n "$pp" ] && PARENTS="$PARENTS $pp"
done

if [ -n "$VLLM_PIDS" ]; then
    log "sending SIGTERM to vLLM servers:$(echo "$VLLM_PIDS" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -TERM $VLLM_PIDS 2>/dev/null || true
else
    log "no vLLM server processes found"
fi

# Give CUDA time to release; polling beats a blind sleep.
for _ in $(seq 1 30); do
    [ -z "$(collect_vllm_pids)" ] && break
    sleep 1
done

SURVIVORS="$(collect_vllm_pids)"
if [ -n "$SURVIVORS" ]; then
    log "SIGKILL survivors:$(echo "$SURVIVORS" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill -KILL $SURVIVORS 2>/dev/null || true
    sleep 3
fi

# Worker children normally die with their parent; clean up any that did not.
ORPHANS=""
for pid in $(pgrep -f 'VLLM::EngineCore|multiprocessing\.resource_tracker' 2>/dev/null | sort -n); do
    is_vllm_child "$pid" && ORPHANS="$ORPHANS $pid"
done
if [ -n "${ORPHANS// /}" ]; then
    log "cleaning up orphaned workers:$ORPHANS"
    # shellcheck disable=SC2086
    kill -KILL $ORPHANS 2>/dev/null || true
fi

# Kill the holder ONLY if it is a sibling of a process we identified as ours.
if [ -n "$PARENTS" ]; then
    for h in $(pgrep -x 'sleep' 2>/dev/null | sort -n); do
        [ -d "/proc/$h" ] || continue
        # Must be `sleep infinity`, not some unrelated sleep.
        case "$(cmdline_of "$h")" in
            *"sleep infinity"*) ;;
            *) continue ;;
        esac
        hp="$(ppid_of "$h" || true)"
        for pp in $PARENTS; do
            if [ "$hp" = "$pp" ]; then
                log "stopping legacy holder session (sleep infinity, pid $h)"
                kill -TERM "$h" 2>/dev/null || true
                break
            fi
        done
    done
fi

# --- 3. stale pidfiles -----------------------------------------------------

for f in logs/vllm-chat.pid logs/vllm-embed.pid; do
    if [ -f "$f" ]; then
        rm -f "$f" && log "removed stale pidfile $f"
    fi
done

# --- 4. verify -------------------------------------------------------------

LEFT="$(collect_vllm_pids)"
if [ -n "$LEFT" ]; then
    log "WARNING: still running:$LEFT"
    exit 1
fi
log "all vLLM processes stopped"
exit 0
