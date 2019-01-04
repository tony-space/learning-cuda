#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

constexpr size_t kNumbers = 704185;
constexpr int kDefaultValue = 0;

static inline __device__ __host__ int divCeil(int a, int b)
{
	return (a - 1) / b + 1;
}

static inline __device__ int warpReduce(int val, int size)
{
	size = min(size, warpSize);
	for (int offset = size / 2; offset > 0; offset >>= 1)
	{
		auto neighbour = __shfl_down_sync(0xFFFFFFFF, val, offset);
		val += neighbour;
	}
	return val;
}

__global__ void reduceKernel(int* values, size_t size, int* output)
{
	int val = kDefaultValue;

	auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
	auto warpId = threadIdx.x / warpSize;
	auto laneId = threadIdx.x % warpSize;
	auto gridSize = blockDim.x * gridDim.x;
	
	extern __shared__ int cache[];
	int cacheSize = divCeil(blockDim.x, warpSize); //equals to amount of warps in blocks

	if (threadId < size)
		val = values[threadId];
	if (threadId + gridSize < size)
		val = val + values[threadId + gridSize];

	val = warpReduce(val, size);
	if (laneId == 0)
		cache[warpId] = val;

	int threads = divCeil(cacheSize, 2);
	if (threadIdx.x >= threads)
		return;
	__syncthreads();

	for (;; threads = divCeil(cacheSize, 2))
	{
		if (threadIdx.x >= threads)
			return;

		auto x = cache[threadIdx.x];
		auto y = threadIdx.x + threads >= cacheSize ? kDefaultValue : cache[threadIdx.x + threads];
		cache[threadIdx.x] = x + y;
		cacheSize = threads;

		if (threads == 1)
			break;

		__syncthreads();
	}

	output[blockIdx.x] = cache[0];
}

int main(int argc, char** argv)
{
	srand(42);
	thrust::host_vector<int> hostNumbers;
	hostNumbers.reserve(kNumbers);
	for (size_t i = 0; i < kNumbers; ++i)
		hostNumbers.push_back(rand() % 101 - 50);

	auto begin = std::chrono::high_resolution_clock::now();
	int controlResult = std::accumulate(hostNumbers.begin(), hostNumbers.end(), 0);
	auto end = std::chrono::high_resolution_clock::now();
	printf("Control result: %d\r\n", controlResult);
	printf("Elapsed time on CPU: %.3f ms\r\n", float(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) * 1e-6f);

	int pairs = divCeil(kNumbers, 2);
	int warps = divCeil(pairs, 32);
	dim3 blockSize(min(1024, warps * 32));
	dim3 gridSize(divCeil(pairs, blockSize.x));

	thrust::device_vector<int> deviceNumbers = hostNumbers;
	thrust::device_vector<int> intermediate(gridSize.x);

	cudaError_t status;
	
	cudaEvent_t start;
	status = cudaEventCreate(&start);
	assert(status == cudaSuccess);
	
	cudaEvent_t stop;
	status = cudaEventCreate(&stop);
	assert(status == cudaSuccess);

	status = cudaEventRecord(start);
	assert(status == cudaSuccess);
	reduceKernel <<<gridSize, blockSize, blockSize.x / 32>>> (thrust::raw_pointer_cast(deviceNumbers.data()), kNumbers, thrust::raw_pointer_cast(intermediate.data()));
	
	cudaDeviceSynchronize();
	int result2 = std::accumulate(intermediate.begin(), intermediate.end(), 0);
	size_t results = gridSize.x;
	blockSize.x = divCeil(gridSize.x, 2);
	gridSize.x = 1;
	reduceKernel <<<gridSize, blockSize, divCeil(blockSize.x, 32) >>> (thrust::raw_pointer_cast(intermediate.data()), results, thrust::raw_pointer_cast(intermediate.data()));

	status = cudaEventRecord(stop);
	assert(status == cudaSuccess);

	int result = intermediate[0];

	status = cudaEventSynchronize(stop);
	assert(status == cudaSuccess);

	float ms;
	status = cudaEventElapsedTime(&ms, start, stop);
	assert(status == cudaSuccess);

	printf("Elapsed time on GPU: %.3f ms", ms);

	status = cudaEventDestroy(stop);
	assert(status == cudaSuccess);
	status = cudaEventDestroy(start);
	assert(status == cudaSuccess);

	assert(controlResult == result);

	return 0;
}