#! /bin/bash

# Workaround for newer linux kernel 
# https://github.com/devcontainers/features/issues/1235#event-21749942947
set -ex
if ! docker info > /dev/null 2>&1; then
    sudo update-alternatives --set iptables /usr/sbin/iptables-nft
fi

# 1. Take ownership of the .venv directory from root
# $(id -u):$(id -g) dynamically grabs your current container user/group (e.g., vscode)
sudo chown -R $(id -u):$(id -g) .venv 2>/dev/null || true

# 1. Initialize the virtual environment in your workspace root (.venv)
uv venv .venv

# 2. Sync all core, optional (ui), and dev dependency groups
uv sync --all-groups

# # Automate virtual environment activation for all interactive bash shells
# echo "source /workspaces/Table-Reclamation-Demo/.venv/bin/activate" >> ~/.bashrc