#!/usr/bin/env bash
set -euo pipefail

log()  { printf '\033[1;32m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*"; }
err()  { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

first_cmd() {
  local x
  for x in "$@"; do
    if command -v "$x" >/dev/null 2>&1; then
      command -v "$x"
      return 0
    fi
  done
  return 1
}

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    die "Neither curl nor wget is available; cannot download dataset."
  fi
}

find_konect_out_file() {
  local search_dir="$1"
  local f base
  local -a matches=()

  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    if [[ "$base" =~ ^out\.[^.]+$ ]]; then
      matches+=("$f")
    fi
  done < <(find "$search_dir" -type f -print0)

  case "${#matches[@]}" in
    0)
      return 1
      ;;
    1)
      printf '%s\n' "${matches[0]}"
      ;;
    *)
      printf '%s\n' "${matches[@]}" >&2
      return 2
      ;;
  esac
}

detect_nvcc() {
  local cand=""

  if [ -n "${CMAKE_CUDA_COMPILER:-}" ]; then
    [ -x "${CMAKE_CUDA_COMPILER}" ] || die "CMAKE_CUDA_COMPILER is not executable: ${CMAKE_CUDA_COMPILER}"
    echo "${CMAKE_CUDA_COMPILER}"
    return 0
  fi

  if [ -n "${CUDACXX:-}" ]; then
    [ -x "${CUDACXX}" ] || die "CUDACXX is not executable: ${CUDACXX}"
    echo "${CUDACXX}"
    return 0
  fi

  cand="$(command -v nvcc 2>/dev/null || true)"
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    echo "$cand"
    return 0
  fi

  if [ -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "/usr/local/cuda/bin/nvcc"
    return 0
  fi

  cand="$(ls -1d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -n1 || true)"
  if [ -n "$cand" ] && [ -x "$cand" ]; then
    echo "$cand"
    return 0
  fi

  return 1
}

detect_vcpkg_root() {
  if [ -n "${VCPKG_ROOT:-}" ]; then
    echo "$VCPKG_ROOT"
    return 0
  fi

  local candidates=(
    "$PROJECT_DIR/.vcpkg"
    "$PROJECT_DIR/../vcpkg"
    "$HOME/vcpkg"
  )

  local d
  for d in "${candidates[@]}"; do
    if [ -f "$d/scripts/buildsystems/vcpkg.cmake" ]; then
      echo "$d"
      return 0
    fi
  done

  echo "$PROJECT_DIR/.vcpkg"
}

check_manifest_dep() {
  local dep="$1"
  if ! grep -Eq "\"$dep\"" "$VCPKG_JSON"; then
    warn "Dependency \"$dep\" was not clearly found in the existing vcpkg.json."
  fi
}

if [ "$(uname -s)" != "Linux" ]; then
  die "This script is written for Linux. Current system: $(uname -s)"
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"
JOBS="${JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

CMAKE_BIN="${CMAKE_BIN:-$(first_cmd cmake || true)}"
NINJA_BIN="${NINJA_BIN:-$(first_cmd ninja ninja-build || true)}"
CC_BIN="${CC_BIN:-${CC:-$(first_cmd cc gcc clang || true)}}"
CXX_BIN="${CXX_BIN:-${CXX:-$(first_cmd c++ g++ clang++ || true)}}"

[ -n "$CMAKE_BIN" ] && [ -x "$CMAKE_BIN" ] || die "cmake was not found. You can specify it via CMAKE_BIN=/path/to/cmake ./build.sh"
[ -n "$NINJA_BIN" ] && [ -x "$NINJA_BIN" ] || die "ninja was not found."
[ -n "$CC_BIN" ] && [ -x "$CC_BIN" ] || die "No C compiler found (cc/gcc/clang)."
[ -n "$CXX_BIN" ] && [ -x "$CXX_BIN" ] || die "No C++ compiler found (c++/g++/clang++)."
command -v tar >/dev/null 2>&1 || die "tar is required but not found."
command -v readlink >/dev/null 2>&1 || die "readlink is required but not found."

log "Project directory: $PROJECT_DIR"
log "Build directory:   $BUILD_DIR"




DATA_URL="http://konect.cc/files/download.tsv.bag-nips.tar.bz2"
DATA_DIR="$PROJECT_DIR/data/konect"
DATA_ARCHIVE="$DATA_DIR/download.tsv.bag-nips.tar.bz2"
DATA_MARKER="$DATA_DIR/.bag-nips.extracted"

mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_ARCHIVE" ]; then
  log "Downloading dataset from: $DATA_URL"
  download_file "$DATA_URL" "$DATA_ARCHIVE"
else
  log "Dataset archive already exists: $DATA_ARCHIVE"
fi

if [ ! -f "$DATA_MARKER" ]; then
  log "Extracting dataset to: $DATA_DIR"
  tar -xjf "$DATA_ARCHIVE" -C "$DATA_DIR"
  touch "$DATA_MARKER"
else
  log "Dataset already extracted. Skipping extraction."
fi




if ! NVCC_BIN="$(detect_nvcc)"; then
  die "CUDA toolkit was not detected (nvcc not found). Please install CUDA manually and rerun this script."
fi

CUDA_ROOT="$(cd -- "$(dirname -- "$NVCC_BIN")/.." && pwd -P)"
log "CUDA detected successfully."
log "nvcc:      $NVCC_BIN"
log "CUDA root: $CUDA_ROOT"




VCPKG_ROOT="$(detect_vcpkg_root)"
export VCPKG_ROOT

if [ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]; then
  log "Using existing vcpkg at: $VCPKG_ROOT"
else
  command -v git >/dev/null 2>&1 || die "git was not found; cannot clone vcpkg automatically."
  log "vcpkg was not found. Cloning into: $VCPKG_ROOT"
  git clone --depth=1 https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
fi

if [ ! -x "$VCPKG_ROOT/vcpkg" ]; then
  [ -x "$VCPKG_ROOT/bootstrap-vcpkg.sh" ] || die "Invalid vcpkg directory: $VCPKG_ROOT"
  log "Bootstrapping vcpkg..."
  (
    cd "$VCPKG_ROOT"
    ./bootstrap-vcpkg.sh -disableMetrics
  )
fi




VCPKG_JSON="$PROJECT_DIR/vcpkg.json"

if [ ! -f "$VCPKG_JSON" ]; then
  BASELINE="$(git -C "$VCPKG_ROOT" rev-parse HEAD 2>/dev/null || true)"
  log "vcpkg.json was not found. Creating: $VCPKG_JSON"

  if [ -n "$BASELINE" ]; then
    cat > "$VCPKG_JSON" <<EOF
{
  "name": "gmkbpe",
  "version-string": "0.1.0",
  "builtin-baseline": "$BASELINE",
  "dependencies": [
    "fmt",
    "nlohmann-json",
    "tbb"
  ]
}
EOF
  else
    cat > "$VCPKG_JSON" <<'EOF'
{
  "name": "gmkbpe",
  "version-string": "0.1.0",
  "dependencies": [
    "fmt",
    "nlohmann-json",
    "tbb"
  ]
}
EOF
  fi
else
  log "Existing vcpkg.json detected. Keeping current content."
  check_manifest_dep "fmt"
  check_manifest_dep "nlohmann-json"
  check_manifest_dep "tbb"
fi




if [ -z "${VCPKG_TARGET_TRIPLET:-}" ]; then
  case "$(uname -m)" in
    x86_64|amd64) VCPKG_TARGET_TRIPLET="x64-linux" ;;
    aarch64|arm64) VCPKG_TARGET_TRIPLET="arm64-linux" ;;
    *)
      die "Unable to infer VCPKG_TARGET_TRIPLET from architecture: $(uname -m). Please set it manually."
      ;;
  esac
fi
log "Using VCPKG_TARGET_TRIPLET=$VCPKG_TARGET_TRIPLET"

mkdir -p "$BUILD_DIR"




log "Configuring CMake project..."
"$CMAKE_BIN" \
  -S "$PROJECT_DIR" \
  -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_MAKE_PROGRAM="$NINJA_BIN" \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
  -DVCPKG_TARGET_TRIPLET="$VCPKG_TARGET_TRIPLET" \
  -DVCPKG_MANIFEST_MODE=ON \
  -DVCPKG_MANIFEST_INSTALL=ON




log "Building target: graph_convert"
"$CMAKE_BIN" --build "$BUILD_DIR" --target graph_convert --parallel "$JOBS"




log "Searching extracted KONECT data file: out.xxx"
OUT_FILE="$(find_konect_out_file "$DATA_DIR")" || {
  rc=$?
  case "$rc" in
    1) die "No file matching out.xxx was found under: $DATA_DIR" ;;
    2) die "Multiple files matching out.xxx were found under: $DATA_DIR" ;;
    *) die "Failed to search out.xxx under: $DATA_DIR" ;;
  esac
}

OUT_FILE="$(readlink -f -- "$OUT_FILE")"
GRAPH_CONVERT_BIN="$BUILD_DIR/graph_convert"
[ -x "$GRAPH_CONVERT_BIN" ] || die "graph_convert executable not found: $GRAPH_CONVERT_BIN"

GRAPH_OUTPUT="${OUT_FILE}"

log "Found KONECT input:  $OUT_FILE"
log "Graph output path:   $GRAPH_OUTPUT.graph.bin"
log "Running graph_convert..."
"$GRAPH_CONVERT_BIN" -i "$OUT_FILE" -o "$GRAPH_OUTPUT"




log "Building target: gmkbpe"
"$CMAKE_BIN" --build "$BUILD_DIR" --target gmkbpe --parallel "$JOBS"

GMKBPE_BIN="$BUILD_DIR/gmkbpe"
[ -x "$GMKBPE_BIN" ] || die "gmkbpe executable not found: $GMKBPE_BIN"
GRAPH_OUTPUT_BIN="$GRAPH_OUTPUT.graph.bin"
[ -f "$GRAPH_OUTPUT_BIN" ] || die "Graph input file not found: $GRAPH_OUTPUT_BIN"


log "Running gmkbpe..."
log "Input graph: $GRAPH_OUTPUT_BIN"
"$GMKBPE_BIN" -f "$GRAPH_OUTPUT_BIN" -k 1 -r 1000000 -q 1 -m 0 -d 0

log "Build completed successfully."
log "Build directory:   $BUILD_DIR"
log "Dataset directory: $DATA_DIR"
log "Graph file:        $GRAPH_OUTPUT"
