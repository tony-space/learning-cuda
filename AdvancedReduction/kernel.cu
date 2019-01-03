#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr size_t kNumbers = 2000;
//constexpr int kDefaultValue = INT_MAX;
constexpr int kDefaultValue = 0;

inline __device__ int warpReduce(int val, int size)
{
	size = min(size, warpSize);
	for (int offset = size / 2; offset > 0; offset >>= 1)
	{
		auto neighbour = __shfl_down_sync(0xFFFFFFFF, val, offset);
		//val = min(val, neighbour);
		val += neighbour;
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
	int cacheSize = blockDim.x / warpSize; //equals to amount of warps in blocks

	if (threadId < size)
		val = values[threadId];
	if (threadId + gridSize < size)
		//val = min(val, values[threadId + gridSize]);
		val = val + values[threadId + gridSize];

	val = warpReduce(val, size);
	if (laneId == 0)
		cache[warpId] = val;

	int threads = (cacheSize - 1 ) / 2 + 1;
	if (threadIdx.x >= threads)
		return;
	__syncthreads();

	for (;; threads = (cacheSize - 1) / 2 + 1)
	{
		if (threadIdx.x >= threads)
			return;

		auto x = cache[threadIdx.x];
		auto y = threadIdx.x + threads >= cacheSize ? kDefaultValue : cache[threadIdx.x + threads];
		//cache[threadIdx.x] = min(x, y);
		cache[threadIdx.x] = x + y;
		cacheSize = threads;

		if (threads == 1)
			break;

		__syncthreads();
	}

	values[blockIdx.x] = cache[0];
}

inline int divCeil(int a, int b)
{
	return (a - 1) / b + 1;
}

int main(int argc, char** argv)
{
	srand(42);
	thrust::host_vector<int> hostNumbers;
	hostNumbers.reserve(kNumbers);
	for (size_t i = 0; i < kNumbers; ++i)
		hostNumbers.push_back(rand() % 50);

	//int controlResult = *std::min_element(hostNumbers.begin(), hostNumbers.end());
	int controlResult = std::accumulate(hostNumbers.begin(), hostNumbers.end(), 0);
	printf("Control result: %d\r\n", controlResult);

	int pairs = divCeil(kNumbers, 2);
	int warps = divCeil(pairs, 32);
	dim3 blockDim(min(1024, warps * 32));
	dim3 gridSize(divCeil(pairs, blockDim.x));

	thrust::device_vector<int> deviceNumbers = hostNumbers;

	reduceKernel <<<gridSize, blockDim, blockDim.x / 32>>> (thrust::raw_pointer_cast(deviceNumbers.data()), kNumbers);
	reduceKernel <<<1, gridSize.x, 1 >>> (thrust::raw_pointer_cast(deviceNumbers.data()), gridSize.x);

	int result = deviceNumbers[0];

	assert(controlResult == result);

	return 0;
}