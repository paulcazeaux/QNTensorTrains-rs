#!/usr/bin/env bash
# Copyright (C) 2023-2026 Paul Cazeaux
#
# This file is part of qntt. qntt is free software: you can
# redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your
# option) any later version. See LICENSE for details.

# Build the C ABI reference into docs/build/capi/.
#
# Doxygen is an optional dependency of the documentation build — the header carries its full
# contract in comments and is perfectly readable on its own. Install it with:
#   macOS   brew install doxygen
#   Debian  apt-get install doxygen
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v doxygen >/dev/null 2>&1; then
  echo "doxygen not found — skipping the C ABI reference." >&2
  echo "Install it (brew install doxygen) and re-run to generate docs/build/capi/." >&2
  exit 0
fi

mkdir -p docs/build/capi
doxygen docs/Doxyfile
echo "C ABI reference written to docs/build/capi/index.html"
