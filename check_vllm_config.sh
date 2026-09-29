#!/bin/bash
# Check vLLM server configuration for function calling support

echo "🔍 Checking vLLM Server Configuration"
echo "======================================="
echo

# Find vLLM processes
echo "1. Finding vLLM processes..."
VLLM_PROCS=$(pgrep -f "vllm.*api_server" 2>/dev/null)

if [ -z "$VLLM_PROCS" ]; then
    echo "❌ No vLLM processes found!"
    echo "   Start vLLM servers first."
    exit 1
fi

echo "✅ Found vLLM processes: $VLLM_PROCS"
echo

# Check each process
for PID in $VLLM_PROCS; do
    echo "----------------------------------------"
    echo "Process ID: $PID"
    
    # Get command line
    if [ -f "/proc/$PID/cmdline" ]; then
        CMDLINE=$(tr '\0' ' ' < /proc/$PID/cmdline)
        echo "Command: $CMDLINE"
        echo
        
        # Check for important flags
        echo "Configuration flags:"
        
        if echo "$CMDLINE" | grep -q -- "--port 8001"; then
            echo "  Type: LLM Server (port 8001)"
        elif echo "$CMDLINE" | grep -q -- "--port 8002"; then
            echo "  Type: Embedding Server (port 8002)"
        fi
        
        if echo "$CMDLINE" | grep -q -- "--model"; then
            MODEL=$(echo "$CMDLINE" | grep -oP -- '--model\s+\K[^\s]+')
            echo "  Model: $MODEL"
        fi
        
        if echo "$CMDLINE" | grep -q -- "--enable-auto-tool-choice"; then
            echo "  ✅ Auto tool choice: ENABLED"
        else
            echo "  ⚠️  Auto tool choice: NOT ENABLED (may need this)"
        fi
        
        if echo "$CMDLINE" | grep -q -- "--tool-call-parser"; then
            PARSER=$(echo "$CMDLINE" | grep -oP -- '--tool-call-parser\s+\K[^\s]+')
            echo "  ✅ Tool call parser: $PARSER"
        else
            echo "  ℹ️  Tool call parser: NOT SET (may use auto-detect)"
        fi
        
        if echo "$CMDLINE" | grep -q -- "--chat-template"; then
            TEMPLATE=$(echo "$CMDLINE" | grep -oP -- '--chat-template\s+\K[^\s]+')
            echo "  Chat template: $TEMPLATE"
        fi
        
        if echo "$CMDLINE" | grep -q -- "--gpu-memory-utilization"; then
            GPU_MEM=$(echo "$CMDLINE" | grep -oP -- '--gpu-memory-utilization\s+\K[^\s]+')
            echo "  GPU memory: $GPU_MEM"
        fi
        
    else
        echo "  Cannot read /proc/$PID/cmdline"
    fi
    
    echo
done

echo "======================================="
echo
echo "🔧 RECOMMENDATIONS:"
echo

# Check if LLM server has function calling support
LLM_PID=$(pgrep -f "vllm.*8001" | head -1)
if [ ! -z "$LLM_PID" ]; then
    CMDLINE=$(tr '\0' ' ' < /proc/$LLM_PID/cmdline 2>/dev/null)
    
    if echo "$CMDLINE" | grep -q -- "--enable-auto-tool-choice"; then
        echo "✅ LLM server has function calling enabled"
    else
        echo "⚠️  LLM server missing function calling flags"
        echo
        echo "Add these flags when starting LLM server (port 8001):"
        echo
        echo "  --enable-auto-tool-choice"
        echo "  --tool-call-parser hermes"
        echo
        echo "Example command:"
        echo "  python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model Qwen/Qwen2.5-32B-Instruct \\"
        echo "    --port 8001 \\"
        echo "    --host 0.0.0.0 \\"
        echo "    --gpu-memory-utilization 0.9 \\"
        echo "    --enable-auto-tool-choice \\"
        echo "    --tool-call-parser hermes"
        echo
    fi
fi

echo
echo "📝 Next steps:"
echo "1. If flags missing, restart vLLM with recommended configuration"
echo "2. Run: python test_function_calling.py"
echo "3. Check AGENT_FIX_GUIDE.md for detailed troubleshooting"
echo

# Check vLLM server health
echo "🏥 Health check:"
echo
curl -s http://127.0.0.1:8001/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ LLM server (8001) is responding"
else
    echo "❌ LLM server (8001) not responding"
fi

echo
curl -s http://127.0.0.1:8002/v1/models 2>/dev/null | python3 -m json.tool 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Embedding server (8002) is responding"
else
    echo "❌ Embedding server (8002) not responding"
fi

echo
echo "Done!"
