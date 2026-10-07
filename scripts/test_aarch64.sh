#!/usr/bin/env bash
# The runtime's tests on emulated aarch64 (PLAN.md M4): the Pi's onnxruntime build, with its ARM
# kernels (int8 included), before there is a Pi. Timings under emulation mean nothing; parity does.
#   scripts/test_aarch64.sh [pytest args]
# Needs qemu user-mode emulation (sudo apt install qemu-user-static); everything else is fetched
# without root into data/aarch64/: CPython (uv), aarch64 wheels, Debian's arm64 C libraries.
set -euo pipefail
cd "$(dirname "$0")/.."
A=$PWD/data/aarch64
PY_VERSION=3.11 # Raspberry Pi OS (Debian 12)

if ! compgen -G "$A/python/cpython-$PY_VERSION*-linux-aarch64-gnu/bin/python$PY_VERSION" > /dev/null; then
  env -u VIRTUAL_ENV UV_PYTHON_INSTALL_DIR="$A/python" uv python install "cpython-$PY_VERSION-linux-aarch64-gnu"
fi
PY=$(ls -d "$A"/python/cpython-$PY_VERSION*-linux-aarch64-gnu/bin/python$PY_VERSION | head -1)

if [[ ! -d $A/site/onnxruntime ]]; then
  env -u VIRTUAL_ENV uv pip install -q --python-platform aarch64-unknown-linux-gnu \
    --python-version $PY_VERSION --target "$A/site" -r runtime/requirements.txt pytest
fi

if [[ ! -e $A/sysroot/.done ]]; then
  mkdir -p "$A/apt/lists/partial" "$A/apt/cache/archives/partial" "$A/debs" "$A/sysroot"
  echo "deb [arch=arm64] http://deb.debian.org/debian trixie main" > "$A/apt/sources.list"
  apt=(apt-get -q -o "Dir::Etc::SourceList=$A/apt/sources.list" -o Dir::Etc::SourceParts=/dev/null
       -o "Dir::State::Lists=$A/apt/lists" -o "Dir::Cache=$A/apt/cache"
       -o APT::Architecture=arm64 -o APT::Architectures=arm64 -o Debug::NoLocking=1)
  "${apt[@]}" update
  (cd "$A/debs" && "${apt[@]}" download libc6:arm64 libstdc++6:arm64 libgcc-s1:arm64 zlib1g:arm64)
  for deb in "$A"/debs/*.deb; do dpkg-deb -x "$deb" "$A/sysroot"; done
  # Debian's merged /usr: the loader lives in usr/lib; absolute links would point at the host.
  ln -sfn usr/lib "$A/sysroot/lib"
  find "$A/sysroot" -type l -lname '/*' | while read -r link; do
    ln -sfn "$A/sysroot$(readlink "$link")" "$link"
  done
  touch "$A/sysroot/.done"
fi

QEMU=$(command -v qemu-aarch64-static || command -v qemu-aarch64 || true)
if [[ -z $QEMU ]]; then
  echo "aarch64 environment ready in $A; needs qemu: sudo apt install qemu-user-static" >&2
  exit 1
fi
export QEMU_LD_PREFIX=$A/sysroot PYTHONPATH=$PWD:$A/site
"$QEMU" "$PY" -c "import platform, numpy, onnxruntime as o
print(platform.machine(), 'python', platform.python_version(), 'onnxruntime', o.__version__, 'numpy', numpy.__version__)"
"$QEMU" "$PY" -m pytest -q -p no:cacheprovider runtime/tests "$@"
