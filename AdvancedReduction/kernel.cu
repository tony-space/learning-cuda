#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <climits>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

constexpr size_t kNumbers = 128;

__global__ void reduceKernel(int* values, size_t size)
{
	int minVal = INT_MAX;
	for (size_t i = 0; i < size; ++i)
		minVal = min(values[i], minVal);
	values[0] = minVal;
}

int main(int argc, char** argv)
{
	thrust::host_vector<int> hostNumbers;
	hostNumbers.reserve(kNumbers);
	for (size_t i = 0; i < kNumbers; ++i)
		hostNumbers.push_back(rand());

	int controlResult = *std::min_element(hostNumbers.begin(), hostNumbers.end());

	thrust::device_vector<int> deviceNumbers = hostNumbers;
	
	reduceKernel <<<1, 1 >>> (thrust::raw_pointer_cast(deviceNumbers.data()), kNumbers);

	int result = deviceNumbers[0];
	
	assert(controlResult == result);

	return 0;
}