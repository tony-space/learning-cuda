#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr int kBlocks = 5;
constexpr int kWarpsPerBlock = 5;
constexpr size_t kNumbers = kBlocks * kWarpsPerBlock * 32;
constexpr int kDefaultValue = INT_MAX;

__inline__ __device__ int warpReduce(int val, int size)
{
	size = min(size, warpSize);
	for (int offset = size / 2; offset > 0; offset >>= 1)
	{
		auto neighbour = __shfl_down_sync(0xFFFFFFFF, val, offset);
		val = min(val, neighbour);
	}
	return val;
}

__global__ void reduceKernel(int* values, size_t size)
{
	int val = kDefaultValue;

	auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
	auto warpId = threadIdx.x / warpSize;
	auto laneId = threadIdx.x % warpSize;
	auto gridSize = blockDim.x * gridDim.x;
	
	extern __shared__ int cache[];
	auto cacheSize = blockDim.x / warpSize; //equals to amount of warps in blocks

	if (threadId < size)
		val = values[threadId];
	if (threadId + gridSize < size)
		val = min(val, values[threadId + gridSize]);

	val = warpReduce(val, size);
	if (laneId == 0)
		cache[warpId] = val;

	if (threadIdx.x >= cacheSize / 2 + cacheSize % 2)
		return;
	__syncthreads();

	for (int threads = cacheSize / 2 + cacheSize % 2; ; threads = threads / 2 + threads % 2)
	{
		if (threadIdx.x >= threads)
			return;

		auto x = cache[threadIdx.x];
		auto y = threadIdx.x + threads >= cacheSize ? kDefaultValue : cache[threadIdx.x + threads];
		cache[threadIdx.x] = min(x, y);
		cacheSize = threads;

		if (threads == 1)
			break;

		__syncthreads();
	}

	values[blockIdx.x] = cache[0];
}

int main(int argc, char** argv)
{
	srand(42);
	thrust::host_vector<int> hostNumbers;
	hostNumbers.reserve(kNumbers);
	for (size_t i = 0; i < kNumbers; ++i)
		hostNumbers.push_back(rand());

	int controlResult = *std::min_element(hostNumbers.begin(), hostNumbers.end());
	printf("Control result: %d\r\n", controlResult);

	thrust::device_vector<int> deviceNumbers = hostNumbers;

	reduceKernel <<<kBlocks, kWarpsPerBlock * 32, kWarpsPerBlock >>> (thrust::raw_pointer_cast(deviceNumbers.data()), kNumbers);
	reduceKernel <<<1, 1 * 32, 1 >>> (thrust::raw_pointer_cast(deviceNumbers.data()), kBlocks);

	int result = deviceNumbers[0];

	assert(controlResult == result);

	return 0;
}