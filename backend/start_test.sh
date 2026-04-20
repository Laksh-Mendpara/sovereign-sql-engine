pkill -f "uvicorn"
cd /home/laksh/repos/sovereign-sql-engine/backend
uv run start > /tmp/sse_server.log 2>&1 &
SERVER_PID=$!
sleep 15
curl -s http://localhost:8000/health
kill $SERVER_PID
