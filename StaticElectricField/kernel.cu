#include <GL\glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <cassert>
#include "kernel.hpp"

__device__ static const float2 kElectronPosition = { -0.3f, 0.0f };
__device__ static const float2 kProtonPosition = { +0.3f, 0.0f };

thrust::device_vector<float> electricField;

__global__ void electricFieldKernel(float* grid, unsigned width, unsigned height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		//get position coords from -1 to +1
		float2 pixelPosition =
		{
			(x / (float)(width - 1)) * 2.0f - 1.0f,
			(y / (float)(height - 1)) * 2.0f - 1.0f
		};

		float dx;
		float dy;
		float distanceSq;

		dx = kElectronPosition.x - pixelPosition.x;
		dy = kElectronPosition.y - pixelPosition.y;
		distanceSq = dx * dx + dy * dy;

		float electronField = -0.01 / distanceSq;

		if (distanceSq < 1e-4)
		{
			grid[x + y * width] = 0.0f;
			return;
		}


		dx = kProtonPosition.x - pixelPosition.x;
		dy = kProtonPosition.y - pixelPosition.y;
		distanceSq = dx * dx + dy * dy;

		if (distanceSq < 1e-4)
		{
			grid[x + y * width] = 0.0f;
			return;
		}

		float protonField = 1 / distanceSq;


		grid[x + y * width] = electronField + protonField;
	}
}

__global__ void renderFieldKernel(float* grid, cudaSurfaceObject_t surfObj, unsigned width, unsigned height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		float field = grid[x + y * width] / 3.0f;

		float4 result = { 0, 0, 0, 1.0f };

		if (field >= 0.0f)
			result.x = field;
		else
			result.z = -field;

		surf2Dwrite(result, surfObj, x * sizeof(result), y);
	}
}

void ProcessElectronField(unsigned textureId, unsigned width, unsigned height)
{
	if (electricField.size() != width * height)
		electricField.resize(width * height);

	cudaError_t error;

	cudaGraphicsResource* cuResource;
	error = cudaGraphicsGLRegisterImage(&cuResource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	assert(!error);

	error = cudaGraphicsMapResources(1, &cuResource);
	assert(!error);

	cudaArray* cuArray;
	error = cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
	assert(!error);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaSurfaceObject_t cuSurfaceObject;
	error = cudaCreateSurfaceObject(&cuSurfaceObject, &resDesc);
	assert(!error);

	dim3 blockDim(32, 32); //32*32 = 1024 threads per block
	dim3 gridDim((width - 1) / 32 + 1, (height - 1) / 32 + 1);

	electricFieldKernel <<<gridDim, blockDim>>> (electricField.data().get(), width, height);
	error = cudaGetLastError();
	assert(!error);

	/*auto pair = thrust::minmax_element(thrust::device, electricField.begin(), electricField.end());

	float* d_min = thrust::raw_pointer_cast(&(*pair.first));
	float* d_max = thrust::raw_pointer_cast(&(*pair.second));*/

	renderFieldKernel <<<gridDim, blockDim>>> (electricField.data().get(), cuSurfaceObject, width, height);
	error = cudaGetLastError();
	assert(!error);

	error = cudaDeviceSynchronize();
	assert(!error);

	error = cudaDestroySurfaceObject(cuSurfaceObject);
	assert(!error);

	error = cudaGraphicsUnmapResources(1, &cuResource);
	assert(!error);

	error = cudaGraphicsUnregisterResource(cuResource);
	assert(!error);
}