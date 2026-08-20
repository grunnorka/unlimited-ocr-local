#!/usr/bin/env bash
# Install Podman compose + NVIDIA Container Toolkit on Fedora for Unlimited-OCR.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Re-run with sudo: sudo bash $0"
  exit 1
fi

echo "==> Installing podman-compose"
dnf install -y podman-compose

echo "==> Adding NVIDIA Container Toolkit repo"
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null

echo "==> Installing nvidia-container-toolkit"
dnf install -y nvidia-container-toolkit

echo "==> Generating CDI spec for Podman"
nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

echo "==> Enabling lingering for rootless Podman (current invoking user if set)"
if [[ -n "${SUDO_USER:-}" ]]; then
  loginctl enable-linger "${SUDO_USER}" || true
fi

echo
echo "Done. Verify GPU in a container with:"
echo "  podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi"
echo
echo "Then start the stack:"
echo "  cd $(dirname "$(dirname "$(readlink -f "$0")")")"
echo "  podman compose up -d --build"
