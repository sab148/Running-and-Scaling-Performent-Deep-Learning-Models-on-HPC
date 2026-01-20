#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://gitlab.jsc.fz-juelich.de/sdlaml/sc_venv_template_HPC_supporter_course.git"
DIR="sc_venv_template_HPC_supporter_course"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: This script must be sourced, not executed."
  echo "Use: source ${BASH_SOURCE[0]}"
  exit 1
fi

# Clone if missing
if [[ ! -d "$DIR" ]]; then
  git clone "$REPO_URL" "$DIR"
fi

# Install/update venv
pushd "$DIR" >/dev/null
bash setup.sh

# Source activation into CURRENT shell
source ./activate.sh
popd >/dev/null

echo "Activated venv from: $(realpath "$DIR")"
echo "python: $(command -v python)"
