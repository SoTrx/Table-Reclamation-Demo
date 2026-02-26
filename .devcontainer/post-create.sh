#! /bin/bash

# Workaround for newer linux kernel 
# https://github.com/devcontainers/features/issues/1235#event-21749942947
set -ex
if ! docker info > /dev/null 2>&1; then
    sudo update-alternatives --set iptables /usr/sbin/iptables-nft
fi