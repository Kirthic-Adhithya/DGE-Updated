#!/usr/bin/env bash
# Wrapper: runs setup_wsl.sh, logs to ~/setup.log, records the real exit code.
cd "$(dirname "$0")/.."
bash wsl/setup_wsl.sh > ~/setup.log 2>&1
echo "EXIT_CODE=$?" >> ~/setup.log
