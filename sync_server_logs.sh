#!/bin/bash

# Auto-sync logs from H100 server
# Usage: bash sync_server_logs.sh SERVER_IP
# Example: bash sync_server_logs.sh ubuntu@192.168.1.100

set -e

SERVER=$1
REMOTE_DIR="~/nanochat/local_rls_experiments_full"
LOCAL_DIR="./local_rls_experiments_full"

if [ -z "$SERVER" ]; then
    echo "Usage: bash sync_server_logs.sh SERVER_IP"
    echo "Example: bash sync_server_logs.sh ubuntu@192.168.1.100"
    exit 1
fi

echo "Starting log sync from $SERVER:$REMOTE_DIR"
echo "Local directory: $LOCAL_DIR"
echo "Press Ctrl+C to stop"
echo ""

mkdir -p "$LOCAL_DIR"

# Sync every 30 seconds
while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Syncing logs..."

    rsync -avz --progress \
        --include="*.log" \
        --include="*/" \
        --exclude="*" \
        "$SERVER:$REMOTE_DIR/" "$LOCAL_DIR/" 2>&1 | grep -E "sending|receiving|total size" || true

    # Show last few lines of each log
    if [ -f "$LOCAL_DIR/rls.log" ]; then
        echo ""
        echo "--- RLS (last 3 lines) ---"
        tail -3 "$LOCAL_DIR/rls.log" | grep "step" || echo "(no recent step output)"
    fi

    if [ -f "$LOCAL_DIR/baseline.log" ]; then
        echo ""
        echo "--- Baseline (last 3 lines) ---"
        tail -3 "$LOCAL_DIR/baseline.log" | grep "step" || echo "(no recent step output)"
    fi

    echo ""
    echo "Next sync in 30 seconds..."
    sleep 30
done
