# Prerequisites

This document describes how to set up an environment on Windows Subsystem for Linux (WSL) to run the Offload samples on NVIDIA GPUs. The instructions presented here were tested on a laptop running Windows 11 Home with:

* AMD Ryzen AI 9 HX (12 cores)
* 32 GB RAM
* NVIDIA GeForce RTX 4070

WSL 2 and a recent NVIDIA Windows GPU driver must already be enabled and installed.

> [!IMPORTANT]
> AMD GPUs and other operating systems are not yet _officially_ supported.

## 1. Install Ubuntu 24.04 on WSL 2

Open the Windows PowerShell and run:

```powershell
wsl --install Ubuntu-24.04
```

Launch the new distribution, choose a UNIX username and password, and execute the next steps.

## 2. Install the CUDA Toolkit for WSL 2

Follow the [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2) and use the WSL-Ubuntu meta-package so you avoid overwriting the Windows host driver.

For example, the commands to install the CUDA Toolkit 12.9 are:

```bash
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-9-local_12.9.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt -y install cuda-toolkit-12-9
```

Verify that the GPU is visible from WSL and CUDA is correctly installed:

```bash
nvidia-smi
```

## 3. Set up environment variables

Append the following to `~/.bashrc`:

```bash
echo 'export CUDA_HOME="/usr/local/cuda"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc
echo 'export LLVM_HOME="$HOME/opt/llvm"' >> ~/.bashrc
echo 'export PATH="$CUDA_HOME/bin:$LLVM_HOME/bin:$PATH"' >> ~/.bashrc
```

Activate the changes and confirm:

```bash
source ~/.bashrc
echo "$LLVM_HOME"
```

## 4. Install the basic toolchain

Install the minimal toolchain required to configure, build and test the needed LLVM subprojects.

```bash
sudo apt update
sudo apt -y install build-essential git cmake ninja-build gcc-multilib python3 python3-pip
```

## 5. Check out the LLVM project

A shallow clone is fast and sufficient for users who only build:

```bash
git clone --depth=1 https://github.com/llvm/llvm-project.git
```

## 6. Configure the LLVM build

Below is a simple recipe to get a release build of the LLVM subprojects _Offload_ and _C library_ (targeting CPUs and NVIDIA GPUs).

```bash
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_RUNTIMES="openmp;offload;libc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_PARALLEL_LINK_JOBS=1 \
  -DCMAKE_INSTALL_PREFIX="$LLVM_HOME" \
  -DRUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc \
  -DLLVM_RUNTIME_TARGETS="default;nvptx64-nvidia-cuda"
```

**Why these options?**

| CMake flag | Rationale |
| ---------- | --------- |
| `LLVM_ENABLE_PROJECTS="clang;lld"` | Provide a recent Clang to compile the GPU C library and LLD to link GPU executables. |
| `LLVM_ENABLE_RUNTIMES="openmp;offload;libc"` | Include OpenMP (required by Offload), Offload itself, and C library for the host. |
| `LLVM_ENABLE_ASSERTIONS=ON` | Keep assertion checks even in a release build (see _Note_ below). |
| `LLVM_PARALLEL_LINK_JOBS=1` | Limit concurrent link jobs to avoid OOM issues. (see _Important_ below). |
| `RUNTIMES_nvptx64-nvidia-cuda_LLVM_ENABLE_RUNTIMES=libc` | Build the C library for the NVIDIA target as well. |
| `LLVM_RUNTIME_TARGETS="default;nvptx64-nvidia-cuda"` | Set the enabled targets to build; in this case, the host and the NVIDIA target. |

> [!NOTE]
> `LLVM_ENABLE_ASSERTIONS=ON` keeps assertion checks even in a release build (default is `OFF`). Remove it if the raw performance matters more than the extra safety and some debuggability.

> [!IMPORTANT]
> To avoid out-of-memory (OOM) issues, configure the option `LLVM_PARALLEL_LINK_JOBS` to permit only one link job per 15GB of RAM available on the host machine.

## 7. Build, install, and run tests

After configuring the build with the above CMake command, build and install the required LLVM subprojects:

```bash
ninja -C build install -j8
```

> [!NOTE]
> Running Ninja with high parallelism can cause spurious failures, out-of-resource errors, or indefinite hangs. Limit the number of jobs with `-j<N>` if you hit such issues.

Now build and run the tests for the new Offload API:

```bash
ninja -C build/runtimes/runtimes-bins offload.unittests
./build/runtimes/runtimes-bins/offload/unittests/OffloadAPI/offload.unittests
```

Finally, run the tests for the GPU C library:
* All tests:
    ```bash
    ninja -C build check-libc-nvptx64-nvidia-cuda -j1
    ```
* Hermetic tests only:
    ```bash
    ninja -C build/runtimes/runtimes-nvptx64-nvidia-cuda-bins libc-hermetic-tests -j1
    ```
* Integration tests only:
    ```bash
    ninja -C build/runtimes/runtimes-nvptx64-nvidia-cuda-bins libc-integration-tests -j1
    ```
