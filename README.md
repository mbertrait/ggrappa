# gGRAPPA: A generalized GRAPPA for MRI reconstruction
<img align="left" width="60%" src="https://github.com/user-attachments/assets/4153e441-03d2-4f6c-9f9f-f338dc07b62b"> 
<h1> A generalized GRAPPA reconstruction package accelerated by GPU </br></br></h1>

gGRAPPA is a Python package allowing GRAPPA reconstruction of k-space data, with batching capabilities allowing huge files to be reconstructed on GPU.


## Features

- **GRAPPA Reconstruction**: Efficiently reconstructs MRI images using the GRAPPA algorithm, which improves imaging speed while maintaining quality.
- **GPU Acceleration**: Utilizes CUDA for GPU acceleration to speed up computation, with various CUDA modes to balance performance and memory usage.
- **Batch Processing**: Processes reconstruction as batched windows to allow GPU acceleration for large data.
- **Flexible Kernel Size**: Supports customizable GRAPPA kernel sizes to suit various needs.
- **Precomputed Kernels**: Allows the use of precomputed GRAPPA kernels for faster reconstruction, or computes them on-the-fly if not provided.
- **Generalized GRAPPA**: CAIPIRINHA support (still WIP)
- **Masking**: Allow use of a binary mask to focus reconstruction on specific regions of interest.

## Installation

To install this package, follow these steps:
1. **Clone the Repository**

   First, clone the repository:
    ```bash
     git clone git@github.com:mbertrait/ggrappa.git
    ```
2. **Navigate to the Project Directory**
    ```bash
     cd ggrappa
    ```
3. **Install the Package**

   Install the package and its dependencies using pip:
    ```bash
     pip install .
    ```
