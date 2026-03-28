# 🚀 High-Performance Texture Feature Extraction (Textons + LTxXORp)

## 📌 Overview

This project demonstrates how to transform a **functionally correct implementation** into a **high-performance, production-ready system** using:

- **C++**
- **OpenCV**
- **MPI (Distributed Computing)**
- **CUDA (GPU Acceleration)**

We implement and optimize a **parallel texture feature extraction pipeline** based on:

- **Texton Calculation**
- **LTxXORp (Local Texture XOR Pattern)**

---

## 🧠 Algorithm Summary

### 1. Texton Calculation
- Operates on **2×2 pixel blocks**
- Produces a **texton-coded image**
- Reduces spatial redundancy

### 2. LTxXORp Calculation
- Similar to **Local Binary Pattern (LBP)**
- Compares each pixel with its neighbors
- Encodes local texture structure

---

## ⚡ Optimization Journey

| Stage | Description |
|------|------------|
| ❌ Naive Implementation | Functional but slow |
| ✅ Optimized Implementation | High-performance, scalable |

---

## 🔑 Key Optimization Principles

### 1️⃣ Master Your Tools (OpenCV)

#### ❌ Naive
- Manual memory allocation (`new`, `delete`)
- Pixel access using `.at<>()`

#### ✅ Optimized
- Use `cv::Mat` (contiguous memory, cache-friendly)
- Direct pointer access:
  ```cpp
  uchar* row_ptr = image.ptr<uchar>(i);

# Optimizing Parallel Texture Feature Extraction (Textons and LTxXORp)

It’s a scenario every developer knows well: you’ve written the code, the logic is sound, and it produces the correct output. But it’s slow. Getting from a functional prototype to a high-performance, production-ready implementation is an art form, grounded in a deep understanding of memory, hardware, and the tools we use.

Using the example of a parallel texture feature extraction algorithm (Textons and LTxXORp), let's dissect two implementations—a common first attempt and a highly optimized version—to uncover the core principles that separate slow code from fast code. Whether you're working with C++, OpenCV, MPI, or CUDA, these lessons are universal.

## The Algorithm at a Glance

The goal is to process an image to extract texture features. This involves two main steps:

* **Texton Calculation:** A transformation that analyzes 2x2 pixel blocks to generate a "texton code."
* **LTxXORp Calculation:** A local pattern encoding, similar to a Local Binary Pattern (LBP), that compares a texton pixel with its neighbors.

Let's explore the key optimization principles revealed by comparing the naive and optimized approaches.

## Principle 1: Master Your Tools—Don't Fight the Framework

The most significant performance gains often come from using a library like OpenCV as it was intended, rather than treating it as a simple pixel container.

### Memory Management: `cv::Mat` vs. Manual Allocation
* **Naive Approach:** Manually allocating memory with `new int*[]` and deallocating with `delete`. This is not only error-prone (hello, memory leaks!) but also inefficient. The operating system overhead for memory management is significant, and generic arrays aren't optimized for the 2D spatial locality of image data.
* **Optimized Approach:** Using OpenCV’s `cv::Mat`. This powerful class handles its own memory management, often using a continuous memory block for 2D data. It's optimized for image processing tasks, minimizing overhead and improving cache performance.

### Pixel Access: The `.at<>()` Method vs. Direct Pointer Access
* **Naive Approach:** Accessing pixels in a loop using `image.at<Vec3b>(i, j)`. While safe (it performs bounds checking), the function call overhead for every single pixel in a tight loop is substantial.
* **Optimized Approach:** Getting a pointer to the start of each row once per row: `uchar* row_ptr = image.ptr<uchar>(i)`. Then, iterating through the row using simple pointer arithmetic (`row_ptr[j]`). This eliminates millions of function calls, leverages CPU cache more effectively, and is drastically faster.

![alt text](image1)

## Principle 2: Speak the GPU's Language with CUDA

When moving from the CPU to the GPU, performance depends entirely on your ability to think in terms of thousands of parallel threads.

### Memory Allocation and Coalescing
* **Naive Approach:** Using `int*` for pixel data that is actually `unsigned char`. This wastes 75% of the memory bandwidth—a critical bottleneck in GPU computing. When threads in a warp access memory, they should access contiguous, aligned blocks. Using the wrong data type breaks this "memory coalescing" and serializes memory access, destroying performance.
* **Optimized Approach:** Using the correct `unsigned char` type and allocating only the necessary memory for each stage (e.g., a half-sized buffer for the texton image). This maximizes bandwidth and allows the GPU hardware to fetch data for multiple threads in a single transaction.

### Kernel Correctness: Avoiding Race Conditions
* **Naive Approach:** Reading from and writing to the same global memory buffer within a single kernel. In the LTxXORp calculation, where a thread needs to read its neighbors' values, another thread might be simultaneously overwriting one of those values. This is a classic race condition that produces unpredictable, incorrect results.
* **Optimized Approach:** Using separate input and output buffers. The kernel reads exclusively from `d_texton_image` and writes exclusively to `d_lbp_image`. This is the fundamental pattern for correct parallel processing and guarantees deterministic results.

![alt text](image2)

## Principle 3: Communicate Intelligently in Parallel (MPI)

In distributed computing, communication is often the biggest bottleneck. Minimizing and structuring data transfer is paramount.

* **Naive Approach:** Complex, manual calculations for data distribution (`Scatterv`) using multiple offset and count arrays. This is hard to debug and often leads to sending more data than necessary, such as overlapping regions for every process.
* **Optimized Approach:** Implementing an explicit halo exchange. Each process works on its core data block. For neighborhood operations like LTxXORp, it only needs the one-pixel-thick border rows/columns from its neighbors. Using `MPI_Sendrecv`, processes efficiently swap just these "halo" regions. This minimizes data transfer and keeps communication clean and targeted.

![alt text](image3)

## Principle 4: If You Don't Measure It Correctly, You Can't Improve It

Accurate timing is essential for identifying bottlenecks.

* **Naive Approach:** Using host-side timers like `clock_gettime()` to measure GPU kernel execution. This is inaccurate because it includes OS jitter and the latency of launching the kernel, and it doesn't wait for the GPU to actually finish.
* **Optimized Approach:** * **For CUDA:** Use `cudaEvent_t`. These are lightweight markers placed in the CUDA stream that record timestamps directly on the GPU, providing precise measurement of kernel execution time.
    * **For MPI:** Use `MPI_Wtime()`, a portable, high-resolution timer synchronized across processes. Combine it with a reduction operation like `MPI_Reduce` to find the maximum execution time across all ranks, which represents the true bottleneck for the entire parallel job.

![alt text](image4)

## Performance Comparison

* **Before Optimization:** Sequential, MPI & CUDA benchmarks.
* **After Optimization:** Optimized code utilizing the techniques described above.

![alt text](image5)

## Conclusion

The journey from a working implementation to a high-performance one is a transition from what the code does to how it does it. By leveraging the full power of specialized libraries, understanding the memory models of your target hardware, communicating intelligently, and measuring accurately, you can achieve dramatic speedups. The optimized code isn't just faster—it's safer, more maintainable, and a better foundation for future work.

---

## System Specifications

### CPU: Intel Core i5-8300H
* **Cores:** 4
* **Threads:** 8
* **Base Clock:** 2.30 GHz
* **Max Turbo Frequency:** 4.00 GHz
* **L3 Cache:** 8 MiB
* **Virtualization:** VT-x
* **Vulnerabilities Mitigated:** Spectre, Meltdown, L1tf, MDS, Retbleed

### GPU: NVIDIA GeForce GTX 1050 Ti
* **Driver Version:** 570.169
* **CUDA Version:** 12.8
* **Memory:** 4 GB GDDR5
* **Max Power Usage:** 5001W

# Build OpenCV with CUDA and C++ Support on Ubuntu

This guide explains how to build and install **OpenCV** with **CUDA** and **cuDNN** support for **C++ development** on Ubuntu. The result is a GPU-accelerated OpenCV installation ready for high-performance computer vision tasks.

---

## ✅ System Configuration (Tested With)

- **OS**: Ubuntu
- **GPU**: NVIDIA GeForce GTX 1050 Ti
- **NVIDIA Driver**: 575.64.03
- **CUDA Toolkit**: 12.2
- **cuDNN**: 8.x (for CUDA 12.x)
- **OpenCV Version**: 4.9.0
- **GCC Version**: 12

---

## ⚙️ Step 1: Install Dependencies and Build Tools
First, update your package list and install the essential tools required for the build process, including cmake, git, and libraries for handling various media formats.

```bash
sudo apt update && sudo apt install -y \
  build-essential \
  cmake \
  git \
  wget \
  unzip \
  libgtk2.0-dev \
  libavcodec-dev \
  libavformat-dev \
  libswscale-dev \
  libtbb-dev \
  libjpeg-dev \
  libpng-dev \
  libtiff-dev \
  libdc1394-dev \
  pkg-config
```

## Step 2: Install a CUDA-Compatible C++ Compiler

The NVIDIA CUDA Toolkit has strict requirements for the host C++ compiler version. For CUDA 12.x, the nvcc compiler requires a GCC version of 12 or older. Modern Ubuntu versions often default to a newer, incompatible compiler.

Install gcc-12 and g++-12 to ensure compatibility.

```bash
sudo apt install -y gcc-12 g++-12
```
## Step 3: Download OpenCV and OpenCV-Contrib Sources
You must download both the main OpenCV repository and the opencv_contrib repository, which contains additional modules needed for full CUDA functionality. It is critical that both repositories are of the exact same version.

Create a build directory:
```bash
mkdir -p ~/opencv_build && cd ~/opencv_build
```
Clone the repositories (version 4.9.0):
```bash
git clone https://github.com/opencv/opencv.git --branch 4.9.0
git clone https://github.com/opencv/opencv_contrib.git --branch 4.9.0
```
## Step 4: Configure the Build with CMake
This is the most critical step. We will configure the build using cmake, pointing it to our specific compiler and enabling all necessary CUDA options.

Create a build directory inside the opencv folder:
```bash
cd opencv
mkdir build && cd build
```
Run the cmake command:
This command is long, but each flag is important. We are:

Specifying the compatible C/C++ compiler (gcc-12/g++-12).

Enabling CUDA, cuBLAS, and cuDNN support.

Setting the CUDA architecture for your GPU (e.g., 6.1 for a GTX 1050 Ti). Find your card's "Compute Capability" if you have a different GPU.

Pointing to the opencv_contrib modules.

Forcing the generation of the pkg-config file (opencv4.pc), which is crucial for compiling your own projects later.
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D CMAKE_C_COMPILER=/usr/bin/gcc-12 \
      -D CMAKE_CXX_COMPILER=/usr/bin/g++-12 \
      -D WITH_CUDA=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D CUDA_ARCH_BIN=6.1 \ #change according to your GPU
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=ON ..
```
Verify the CMake Output:
After the command finishes, scroll through the output summary. Ensure you see YES for NVIDIA CUDA and that the C++ Compiler is listed as /usr/bin/g++-12.

## Step 5: Compile and Install OpenCV
This process is resource-intensive and can take a long time.

Run the make command:
The -j$(nproc) flag uses all available CPU cores to speed up compilation.
```bash
make -j$(nproc)
```
⚠️ Potential Error: Out of Memory
If the compilation fails with an error like make: *** [Makefile:166: all] Error 2, it's likely your system ran out of RAM. Re-run the command with fewer parallel jobs. Using half your cores or just four is a safe alternative.

# If the first command fails, try this one:
```bash
make -j4
```
Install the compiled libraries:
```bash
sudo make install
```
Update the library cache:
```bash
sudo ldconfig
```
## Step 6: Final Verification
Let's confirm that the installation was successful and that your system can find the new libraries.

Find and Configure pkg-config:
The system needs to know where to find the opencv4.pc file.

First, locate the file:
```bash
sudo find /usr/local -name "opencv4.pc"
```
This will output a path, for example: /usr/local/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc.

Copy the directory part of that path.

Add this directory to your PKG_CONFIG_PATH for your current session. Use the actual path you found.

# Example command - replace the path with your own
export PKG_CONFIG_PATH=/usr/local/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

To make this change permanent, add that same export line to the end of your ~/.bashrc file and run source ~/.bashrc.

Check the OpenCV Version:
```bash
pkg-config --modversion opencv4
```
This should correctly output: 4.9.0.

Compile and Run a C++ Test Program:

Create a test file:
```bash
nano cv_test.cpp
```
# Paste the following code inside:
```bash
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_devices > 0) {
        std::cout << "CUDA is enabled." << std::endl;
        std::cout << "Number of CUDA devices: " << cuda_devices << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    } else {
        std::cout << "CUDA is NOT enabled in this OpenCV build." << std::endl;
    }
    return 0;
}
```
# Save and exit (Ctrl+X, Y, Enter).

# Compile the program using pkg-config to supply the flags:
```bash
g++ cv_test.cpp -o cv_test $(pkg-config --cflags --libs opencv4)
```
# Run the executable:
```bash
./cv_test
```
You should see output confirming your OpenCV version and that CUDA support is enabled, along with details about your GPU.

Congratulations! You are now ready to build GPU-accelerated computer vision projects with OpenCV and C++.
