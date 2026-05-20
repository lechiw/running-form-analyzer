#!/bin/bash
# start_web.sh - Start or restart the Running Form Analyzer web UI

set -e

cd "$(dirname "$0")"

echo "🏃 Starting Running Form Analyzer Web UI..."

# Kill any existing streamlit
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 1

# Start streamlit
nohup streamlit run app.py --server.headless true --server.port 8501 > /tmp/sl.log 2>&1 &
PID=$!
echo "   Streamlit PID: $PID"
sleep 4

# Verify it's running
if ! curl -s -o /dev/null http://127.0.0.1:8501/; then
    echo "❌ Failed to start streamlit"
    exit 1
fi

# Fix Windows port forwarding
WSL_IP=$(hostname -I | awk '{print $1}')
echo "   WSL IP: $WSL_IP"

powershell.exe -Command "netsh interface portproxy delete v4tov4 listenport=8501" 2>/dev/null || true
powershell.exe -Command "netsh interface portproxy add v4tov4 listenport=8501 listenaddress=0.0.0.0 connectport=8501 connectaddress=$WSL_IP" 2>/dev/null

# Ensure firewall rule
powershell.exe -Command "New-NetFirewallRule -DisplayName 'Streamlit 8501' -Direction Inbound -LocalPort 8501 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue" 2>/dev/null || true

echo ""
echo "✅ Running at: http://localhost:8501"
echo "   PID: $PID"
