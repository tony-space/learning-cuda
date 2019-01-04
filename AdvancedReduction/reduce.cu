#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr size_t kMaxBlockSize = 256;

constexpr int kDefaultValue = 0;

template<typename T>
static inline __device__ __host__ T divCeil(T a, T b)
{
	return (a - 1) / b + 1;
}

template<typename T>
static inline __device__ T warpReduce(T val)
{
	for (auto offset = warpSize >> 1; offset > 0; offset >>= 1)
	{
		T neighbour = __shfl_down_sync(0xFFFFFFFF, val, offset);
		val += neighbour;
	}
	return val;
}

__global__ void reduceKernel(const float* __restrict__ values, size_t size, float* __restrict__ output)
{
	auto val = kDefaultValue;

	auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
	auto warpId = threadIdx.x / unsigned(warpSize);
	auto laneId = threadIdx.x % warpSize;
	auto gridSize = blockDim.x * gridDim.x;

	extern __shared__ float cache[];
	auto cacheSize = divCeil(blockDim.x, unsigned(warpSize)); //equals to amount of warps in blocks

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

void cudaReduce(const std::vector<float>& data)
{
	auto totalNumbers = data.size();
	thrust::device_vector<float> deviceNumbers(data.data(), data.data() + data.size());
	thrust::device_vector<float> intermediate(divCeil(divCeil(totalNumbers, size_t(2)), kMaxBlockSize));
	//auto x = intermediate.size();

	cudaError_t status;

	cudaEvent_t start;
	status = cudaEventCreate(&start);
	assert(status == cudaSuccess);

	cudaEvent_t stop;
	status = cudaEventCreate(&stop);
	assert(status == cudaSuccess);

	auto buffer1 = thrust::raw_pointer_cast(deviceNumbers.data());
	auto buffer2 = thrust::raw_pointer_cast(intermediate.data());

	printf("Parallel GPU started...\r\n");
	status = cudaEventRecord(start);
	assert(status == cudaSuccess);

	for (size_t numbers = totalNumbers; numbers > 1;)
	{
		size_t pairs = divCeil(numbers, size_t(2));
		size_t warps = divCeil(pairs, size_t(32));
		dim3 blockSize(unsigned(min(kMaxBlockSize, warps * size_t(32))));
		dim3 gridSize(unsigned(divCeil(pairs, size_t(blockSize.x))));

		reduceKernel <<<gridSize, blockSize, blockSize.x / 32 * sizeof(float)>>> (buffer1, numbers, buffer2);
		//cudaDeviceSynchronize();
		std::swap(buffer1, buffer2);
		numbers = gridSize.x;
	}

	status = cudaEventRecord(stop);
	assert(status == cudaSuccess);

	float result;
	status = cudaMemcpy(&result, buffer1, sizeof(result), cudaMemcpyDeviceToHost);
	assert(status == cudaSuccess);

	status = cudaEventSynchronize(stop);
	assert(status == cudaSuccess);

	float ms;
	status = cudaEventElapsedTime(&ms, start, stop);
	assert(status == cudaSuccess);

	printf("Elapsed time on GPU: %.3f ms\r\n", ms);
	printf("Result: %.3f\r\n", result);
	//deviceNumbers = thrust::host_vector<float>(data);
	//printf("Result2: %.3f\r\n", thrust::reduce(deviceNumbers.begin(), deviceNumbers.end()));

	status = cudaEventDestroy(stop);
	assert(status == cudaSuccess);
	status = cudaEventDestroy(start);
	assert(status == cudaSuccess);
}