#!/bin/bash
# start_web.sh - Start Running Form Analyzer services
# Usage: bash start_web.sh [api|streamlit|all]

set -e
cd "$(dirname "$0")"

MODE="${1:-api}"

fix_proxy() {
    local port=$1
    local ip=$(hostname -I | awk '{print $1}')
    powershell.exe -Command "netsh interface portproxy delete v4tov4 listenport=$port" 2>/dev/null || true
    powershell.exe -Command "netsh interface portproxy add v4tov4 listenport=$port listenaddress=0.0.0.0 connectport=$port connectaddress=$ip" 2>/dev/null || true
    powershell.exe -Command "New-NetFirewallRule -DisplayName 'Streamlit $port' -Direction Inbound -LocalPort $port -Protocol TCP -Action Allow -ErrorAction SilentlyContinue" 2>/dev/null || true
}

if [ "$MODE" = "api" ] || [ "$MODE" = "all" ]; then
    echo "🚀 Starting API server (port 8000)..."
    pkill -f "uvicorn api.main" 2>/dev/null || true
    sleep 1
    setsid uvicorn api.main:app --host 0.0.0.0 --port 8000 </dev/null >/tmp/api.log 2>&1 &
    API_PID=$!
    sleep 3
    echo "   API PID: $API_PID"
    curl -s -o /dev/null http://127.0.0.1:8000/api/health && echo "   ✅ API running"
    fix_proxy 8000
    echo "   ✅ http://localhost:8000"
fi

if [ "$MODE" = "streamlit" ] || [ "$MODE" = "all" ]; then
    echo "🚀 Starting Streamlit UI (port 8501)..."
    pkill -f "streamlit run app.py" 2>/dev/null || true
    sleep 1
    setsid streamlit run app.py --server.headless true --server.port 8501 </dev/null >/tmp/sl.log 2>&1 &
    ST_PID=$!
    sleep 4
    echo "   Streamlit PID: $ST_PID"
    curl -s -o /dev/null http://127.0.0.1:8501/ && echo "   ✅ Streamlit running"
    fix_proxy 8501
    echo "   ✅ http://localhost:8501"
fi

echo ""
echo "🎯 Done!"
