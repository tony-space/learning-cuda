#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <chrono>

constexpr size_t kNumbers = 704185371;
constexpr size_t kMaxBlockSize = 512;

constexpr int kDefaultValue = 0;

template<typename T>
static inline __device__ __host__ T divCeil(T a, T b)
{
	return (a - 1) / b + 1;
}

static inline __device__ int warpReduce(int val)
{
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
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
	int cacheSize = divCeil<int>(blockDim.x, warpSize); //equals to amount of warps in blocks

	if (threadId < size)
		val = values[threadId];
	if (threadId + gridSize < size)
		val = val + values[threadId + gridSize];

	val = warpReduce(val);
	if (laneId == 0)
		cache[warpId] = val;

	if (warpId > 0)
		return;

	__syncthreads();

	val = laneId < cacheSize ? cache[laneId] : kDefaultValue;
	val = warpReduce(val);

	if (laneId == 0)
		output[blockIdx.x] = val;
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

	thrust::device_vector<int> deviceNumbers = hostNumbers;
	thrust::device_vector<int> intermediate(divCeil(divCeil(kNumbers, size_t(2)), kMaxBlockSize));
	auto x = intermediate.size();

	cudaError_t status;

	cudaEvent_t start;
	status = cudaEventCreate(&start);
	assert(status == cudaSuccess);

	cudaEvent_t stop;
	status = cudaEventCreate(&stop);
	assert(status == cudaSuccess);

	status = cudaEventRecord(start);
	assert(status == cudaSuccess);


	auto buffer1 = thrust::raw_pointer_cast(deviceNumbers.data());
	auto buffer2 = thrust::raw_pointer_cast(intermediate.data());

	for (size_t numbers = kNumbers; numbers > 1;)
	{
		size_t pairs = divCeil(numbers, size_t(2));
		size_t warps = divCeil(pairs, size_t(32));
		dim3 blockSize(min(kMaxBlockSize, warps * size_t(32)));
		dim3 gridSize(divCeil(pairs, size_t(blockSize.x)));

		reduceKernel <<<gridSize, blockSize, blockSize.x / 32>>> (buffer1, numbers, buffer2);
		std::swap(buffer1, buffer2);
		numbers = gridSize.x;
	}

	status = cudaEventRecord(stop);
	assert(status == cudaSuccess);

	int result;
	status = cudaMemcpy(&result, buffer1, sizeof(result), cudaMemcpyDeviceToHost);
	assert(status == cudaSuccess);

	status = cudaEventSynchronize(stop);
	assert(status == cudaSuccess);

	float ms;
	status = cudaEventElapsedTime(&ms, start, stop);
	assert(status == cudaSuccess);

	printf("Elapsed time on GPU: %.3f ms\r\n", ms);
	printf("Result: %d \r\n", result);

	status = cudaEventDestroy(stop);
	assert(status == cudaSuccess);
	status = cudaEventDestroy(start);
	assert(status == cudaSuccess);

	assert(controlResult == result);

	system("pause");
	return 0;
}