#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <vector>


__global__ void check_distance_kernel(
    const float* d_coord_vecs1,
    const float* d_coord_vecs2,
    const float* d_distances_vecs,
    const int* d_distances_offsets,
    const int* d_distances_sizes,
    bool* d_results,
    int n_vecs,
    int n_dim,
    float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_vecs) {
        d_results[idx] = false;
        float dist = 0.0f;
        for (int i = 0; i < n_dim; ++i) {
            float diff = d_coord_vecs1[idx * n_dim + i] - d_coord_vecs2[idx * n_dim + i];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        // for (int i = 0; i < d_distances_sizes[idx]; ++i) {
        //     if (fabsf(dist - d_distances_vecs[d_distances_offsets[idx] + i]) <= threshold) {
        //         d_results[idx] = true;
        //         break;
        //     }
        // }
        // Binary search for a value within the threshold
        int left = d_distances_offsets[idx];
        int right = left + d_distances_sizes[idx] - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            float mid_val = d_distances_vecs[mid];

            if (fabsf(dist - mid_val) <= threshold) {
                d_results[idx] = true;
                break;
            } else if (dist < mid_val) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
    }
}

bool* cuda_check_distances(
    float* coord_vecs1,
    float* coord_vecs2,
    float* distances_vecs,
    int* distances_offsets,
    int* distances_sizes,
    int n_vecs,
    int n_dim,
    float threshold) {
    float *d_coord_vecs1, *d_coord_vecs2, *d_distances_vecs;
    int *d_distances_offsets, *d_distances_sizes;
    bool *d_results;
    int distances_length = 0;
    if (n_vecs >= 1) {
        distances_length = distances_offsets[n_vecs - 1] + distances_sizes[n_vecs - 1];
    }
    cudaError_t cudaStatus;

    // Create a CUDA stream
    cudaStream_t stream;
    cudaStatus = cudaStreamCreate(&stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaStreamCreate error: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Allocate device memory
    cudaStatus = cudaMalloc(&d_coord_vecs1, n_vecs * n_dim * sizeof(float));
    cudaStatus = cudaMalloc(&d_coord_vecs2, n_vecs * n_dim * sizeof(float));
    cudaStatus = cudaMalloc(&d_distances_vecs, distances_length * sizeof(float));
    cudaStatus = cudaMalloc(&d_distances_offsets, n_vecs * sizeof(int));
    cudaStatus = cudaMalloc(&d_distances_sizes, n_vecs * sizeof(int));
    cudaStatus = cudaMalloc(&d_results, n_vecs * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc error: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Copy vectors to device
    cudaStatus = cudaMemcpyAsync(d_coord_vecs1, coord_vecs1, n_vecs * n_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStatus = cudaMemcpyAsync(d_coord_vecs2, coord_vecs2, n_vecs * n_dim * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStatus = cudaMemcpyAsync(d_distances_vecs, distances_vecs, distances_length * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStatus = cudaMemcpyAsync(d_distances_offsets, distances_offsets, n_vecs * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaStatus = cudaMemcpyAsync(d_distances_sizes, distances_sizes, n_vecs * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync error: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Define the number of threads and blocks
    int block_size = 256; // Number of threads per block
    // if (n_vecs > 1000000) {
    //     block_size = 512;
    // } else if (n_vecs > 100000) {
    //     block_size = 256;
    // } else {
    //     block_size = 128;
    // }
    int n_blocks = (n_vecs + block_size - 1) / block_size; // Calculate number of blocks needed

    // Launch kernel
    check_distance_kernel<<<n_blocks, block_size, 0, stream>>>(d_coord_vecs1, d_coord_vecs2, d_distances_vecs, d_distances_offsets, d_distances_sizes, d_results, n_vecs, n_dim, threshold);
    // cudaStatus = cudaDeviceSynchronize();
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(cudaStatus));
    // }

    // Allocate host memory for results
    bool* results = new bool[n_vecs];

    // Copy results back to host
    cudaStatus = cudaMemcpyAsync(results, d_results, n_vecs * sizeof(bool), cudaMemcpyDeviceToHost, stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyAsync error: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Wait for all operations in the stream to complete
    cudaStreamSynchronize(stream);

    // Clean up
    cudaFree(d_coord_vecs1);
    cudaFree(d_coord_vecs2);
    cudaFree(d_distances_vecs);
    cudaFree(d_distances_offsets);
    cudaFree(d_distances_sizes);
    cudaFree(d_results);

    // Destroy the stream
    cudaStreamDestroy(stream);

    return results;
}

bool is_gpu_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (device_count > 1) {
        std::srand(std::time(nullptr));
        int random_device = std::rand() % device_count;
        err = cudaSetDevice(random_device);
        if (err != cudaSuccess) {
            const char* visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
            std::vector<int> devices;
            if (visible_devices) {
                std::istringstream ss(visible_devices);
                std::string device_str;
                while (std::getline(ss, device_str, ',')) {
                    try {
                        devices.push_back(std::stoi(device_str));
                    } catch (...) {}
                }
            }
            err = cudaSetDevice(devices[random_device]);
        }
    }
    if (err == cudaSuccess && device_count > 0) {       
        return true;
    } else {
        return false;
    }
}
