#!/bin/bash

# Start API server in background
echo "Starting API server..."
python -m tools.api_server --llama-checkpoint-path /app/fish-speech/checkpoints/fish-agent-v0.1-3b --mode agent --compile 2>&1 | tee /var/log/api_server.log &

API_PID=$!

# Wait for API server to be ready
echo "Waiting for API server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
while ! curl -s http://localhost:8080/v1/health > /dev/null; do
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "API server failed to start. Check logs at /var/log/api_server.log"
        exit 1
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Timeout waiting for API server"
        exit 1
    fi
    
    echo "Waiting for API server... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done

echo "API server is ready"

# Start RunPod handler
echo "Starting RunPod handler..."
exec python -u runpod/handler.py
