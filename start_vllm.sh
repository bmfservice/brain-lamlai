#!/usr/bin/env bash
# Start the BILL vLLM servers inside WSL: chat on GPU 0 (port 8001), embedding on
# GPU 1 (port 8002). Every model id, GPU index and VRAM fraction comes from
# .env — nothing about the models or hardware is hard-coded here.
#
# Two modes:
#
#   start_vllm.sh chat     # foreground, exec one server — for systemd (ExecStart)
#   start_vllm.sh embed    # foreground, exec one server — for systemd (ExecStart)
#   start_vllm.sh          # LEGACY: background both, wait for health, exit
#
# Role mode is what the systemd user units run (config/braind-vllm-*.service):
# it ``exec``s a single server so systemd owns the process directly, can restart
# it on failure and reap the whole cgroup on stop. The legacy both-servers mode
# is kept as an emergency fallback for running by hand, but note that anything
# it backgrounds dies when the launching WSL session is torn down — that is
# exactly why the units exist.
#
#   DRY_RUN=1 start_vllm.sh chat   # print the command instead of running it
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "ERROR: .env not found in $(pwd)" >&2
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

VENV="${VLLM_VENV:-$HOME/.venv-vllm}"
PYTHON="$VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Run: python3 -m venv $VENV && $VENV/bin/pip install -r requirements-vllm.txt" >&2
    exit 1
fi

mkdir -p logs

# WSL2: vLLM disables pinned memory by default (vllm/platforms/cuda.py gates on
# this env var). The chat engine requires UVA for its all_token_ids buffer
# (v1/worker/gpu/states.py uva_instead_of_gpu=True), and UVA requires pinned
# memory — without this the server dies with "RuntimeError: UVA is not
# available". Safe to enable: kernel 6.18.33.2 >> the 4.19.121 minimum.
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
fi

# flashinfer JIT-compiles kernels at engine startup and needs nvcc + ninja.
# nvcc ships in the pip nvidia-cu13 package; ninja ships in the venv bin —
# put both on PATH instead of requiring a system CUDA toolkit install.
# (VENV may be absolute, e.g. ~/.venv-vllm, so do NOT join it with $PWD.)
export PATH="$VENV/bin:$PATH"
PIP_NVCC="$VENV/lib/python3.14/site-packages/nvidia/cu13"
if [ -x "$PIP_NVCC/bin/nvcc" ]; then
    export CUDA_HOME="$PIP_NVCC"
    export PATH="$PIP_NVCC/bin:$PATH"
fi
# nvcc 13.4 + the only on-disk CUDART headers (cu13/include = 13.0) trip
# flashinfer's CCCL toolkit-compatibility #error. The check exists to catch
# *header* upgrades past the compiler; here the headers are OLDER, which is the
# supported direction, so inject -D via flashinfer's documented env hook
# (an env var alone does not reach nvcc — it must be a -D compile flag).
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"

# --- Server command lines (single source of truth for both modes) -----------
# Built as arrays so role mode can ``exec`` them verbatim and legacy mode can
# background the exact same argv — no drift between the two paths.
EMBED_CMD=(
    env "CUDA_VISIBLE_DEVICES=${VLLM_EMBED_GPU:-1}"
    "$PYTHON" -m vllm.entrypoints.openai.api_server
        --model "${EMBED_MODEL}"
        --host 0.0.0.0
        --port "${VLLM_EMBED_PORT:-8002}"
        --dtype auto
        --runner pooling
        --max-model-len "${VLLM_EMBED_MAX_MODEL_LEN:-8192}"
        --gpu-memory-utilization "${VLLM_EMBED_GPU_MEM_UTIL:-0.12}"
)

# Tool-calling flags are mandatory: core/agent/loop.py depends on native
# tool_calls; without them the agent can never invoke its tools.
CHAT_CMD=(
    env "CUDA_VISIBLE_DEVICES=${VLLM_LLM_GPU:-0}"
    "$PYTHON" -m vllm.entrypoints.openai.api_server
        --model "${LLM_MODEL}"
        --host 0.0.0.0
        --port "${VLLM_LLM_PORT:-8001}"
        --dtype auto
        --max-model-len "${VLLM_LLM_MAX_MODEL_LEN:-16384}"
        --gpu-memory-utilization "${VLLM_LLM_GPU_MEM_UTIL:-0.62}"
        --enable-auto-tool-choice
        --tool-call-parser "${VLLM_TOOL_CALL_PARSER:-gemma4}"
)

print_cmd() {
    # %q so the printed line is shell-safe to copy-paste and diff against
    # `pgrep -af` output of a running server.
    printf '%q ' "$@"
    printf '\n'
}

start_server() {
    local name="$1"; shift
    echo "[start_vllm] starting $name ..."
    "$@" >"logs/$name.log" 2>&1 &
    echo $! >"logs/$name.pid"
}

wait_for() {
    local name="$1" port="$2" tries="${3:-120}"
    echo "[start_vllm] waiting for $name on :$port ..."
    for _ in $(seq 1 "$tries"); do
        if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
            echo "[start_vllm] $name ready"
            return 0
        fi
        sleep 5
    done
    echo "[start_vllm] $name FAILED to become ready; see logs/$name.log" >&2
    return 1
}

ROLE="${1:-}"

case "$ROLE" in
    embed)
        if [ "${DRY_RUN:-0}" = "1" ]; then
            print_cmd "${EMBED_CMD[@]}"
            exit 0
        fi
        # exec: systemd supervises the server process itself, not a wrapper
        # shell, so Restart=/KillMode= behave on the real process tree.
        exec "${EMBED_CMD[@]}"
        ;;
    chat)
        if [ "${DRY_RUN:-0}" = "1" ]; then
            print_cmd "${CHAT_CMD[@]}"
            exit 0
        fi
        exec "${CHAT_CMD[@]}"
        ;;
    "")
        # --- LEGACY both-servers mode (manual/emergency use) ---
        if [ "${DRY_RUN:-0}" = "1" ]; then
            print_cmd "${EMBED_CMD[@]}"
            print_cmd "${CHAT_CMD[@]}"
            exit 0
        fi
        # Embedding first: it claims the smaller share of VRAM, and starting the
        # chat model first can leave too little headroom for it on the card.
        start_server vllm-embed "${EMBED_CMD[@]}"
        start_server vllm-chat "${CHAT_CMD[@]}"

        wait_for vllm-embed "${VLLM_EMBED_PORT:-8002}"
        wait_for vllm-chat "${VLLM_LLM_PORT:-8001}"

        echo "[start_vllm] both servers up. Stop with: kill \$(cat logs/vllm-embed.pid logs/vllm-chat.pid)"
        echo "[start_vllm] NOTE: these die if this WSL session is torn down — use the systemd units (make install-units) for a durable stack."
        ;;
    *)
        echo "usage: $(basename "$0") [chat|embed]" >&2
        echo "  chat   run the chat server in the foreground (systemd role)" >&2
        echo "  embed  run the embedding server in the foreground (systemd role)" >&2
        echo "  (none) legacy: background both servers and wait for health" >&2
        echo "  DRY_RUN=1 prints the command instead of running it" >&2
        exit 2
        ;;
esac
