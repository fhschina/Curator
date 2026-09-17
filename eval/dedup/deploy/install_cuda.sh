#!/usr/bin/env bash
# Install the CUDA 12.9.1 components needed by the H100 model worker.
set -euo pipefail
umask 077

if [[ $# != 2 ]]; then
    printf 'Usage: PYTHON_BIN=/path/to/python bash install_cuda.sh TOOLKIT_DIR DOWNLOAD_DIR\n' >&2
    exit 2
fi

"${PYTHON_BIN:-python}" - "$1" "$2" <<'PY'
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
    raise SystemExit("This installer requires Linux x86_64")
toolkit, cache = (Path(value).expanduser().resolve() for value in sys.argv[1:])
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
cache.mkdir(parents=True, exist_ok=True)
toolkit.mkdir(parents=True, exist_ok=True)

def download(name, relative_path, digest):
    archive = cache / name
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".partial")
        with urllib.request.urlopen(base + relative_path, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
        partial.rename(archive)
    with archive.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != digest:
        raise RuntimeError(f"Checksum mismatch; remove {archive} before retrying")
    return archive

manifest_path = download(
    "redistrib_12.9.1.json", "redistrib_12.9.1.json",
    "8335301010b0023ee1ff61eb11e2600ca62002d76780de4089011ad77e0c7630",
)
manifest = json.loads(manifest_path.read_text())
installed = {}
for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcurand"):
    component = manifest[name]["linux-x86_64"]
    archive = download(Path(component["relative_path"]).name, component["relative_path"], component["sha256"])
    subprocess.run(["tar", "-xJf", str(archive), "--strip-components=1", "-C", str(toolkit)], check=True)
    installed[name] = component
if not (toolkit / "lib64").exists():
    (toolkit / "lib64").symlink_to("lib", target_is_directory=True)
with tempfile.TemporaryDirectory(prefix="cuda-probe-", dir=cache) as temp:
    source = Path(temp) / "probe.cu"
    source.write_text(
        "#include <cuda_runtime.h>\n#include <cuda_fp8.h>\n#include <cub/cub.cuh>\n"
        "#include <curand.h>\n#include <curand_kernel.h>\n"
        "__global__ void probe(float* x) { x[threadIdx.x] = 1.0f; }\n"
        "int main() { return (int)cudaFree(nullptr); }\n"
    )
    subprocess.run([str(toolkit / "bin/nvcc"), "-arch=sm_90", str(source), "-o", str(Path(temp) / "probe")], check=True)
(toolkit / "installed-components.json").write_text(json.dumps(installed, indent=2) + "\n")
print(f"CUDA toolkit installed in {toolkit}; H100 compile/link probe passed (no GPU execution)")
PY
