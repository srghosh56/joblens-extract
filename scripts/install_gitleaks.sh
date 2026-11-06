#!/usr/bin/env bash
# Downloads a pinned gitleaks binary into ./.tools (project-local, never system-wide)
# and verifies its SHA-256 against the release's checksum file.
set -euo pipefail
VERSION="8.30.1"
case "$(uname -s)-$(uname -m)" in
  Darwin-x86_64) PLAT="darwin_x64" ;;
  Darwin-arm64)  PLAT="darwin_arm64" ;;
  Linux-x86_64)  PLAT="linux_x64" ;;
  Linux-aarch64) PLAT="linux_arm64" ;;
  *) echo "unsupported platform"; exit 1 ;;
esac
DIR="$(cd "$(dirname "$0")/.." && pwd)/.tools"
[ -x "$DIR/gitleaks" ] && "$DIR/gitleaks" version | grep -q "$VERSION" && exit 0
mkdir -p "$DIR/dl"
BASE="https://github.com/gitleaks/gitleaks/releases/download/v${VERSION}"
TAR="gitleaks_${VERSION}_${PLAT}.tar.gz"
curl -sSfL -o "$DIR/dl/$TAR" "$BASE/$TAR"
curl -sSfL -o "$DIR/dl/checksums.txt" "$BASE/gitleaks_${VERSION}_checksums.txt"
(cd "$DIR/dl" && grep " $TAR\$" checksums.txt | shasum -a 256 -c -)
tar -xzf "$DIR/dl/$TAR" -C "$DIR" gitleaks
rm -rf "$DIR/dl"
"$DIR/gitleaks" version
