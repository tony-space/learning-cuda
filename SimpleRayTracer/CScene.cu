#include <GL\glew.h>

//#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>

#include "CScene.hpp"

__device__ const float3 spherePos = { 0.0f, 0.0f, -2.0f };
__device__ const float sphereRad = 1.0f;
__device__ const float3 pointLightPos = { 1.0f, 1.0f, 0.0f };

__global__ void renderFieldKernel(cudaSurfaceObject_t surfObj, unsigned width, unsigned height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float3 ray = { float(x) / float(width - 1) * 2.0f - 1.0f, float(y) / float(height - 1) * 2.0f - 1.0f, -1.0f };
	float length = norm3df(ray.x, ray.y, ray.z);
	ray.x /= length;
	ray.y /= length;
	ray.z /= length;

	float projection = spherePos.x * ray.x + spherePos.y * ray.y + spherePos.z * ray.z;
	float3 sphereCenterProj = { ray.x * projection, ray.y * projection, ray.z * projection };
	float3 diff = { sphereCenterProj.x - spherePos.x, sphereCenterProj.y - spherePos.y, sphereCenterProj.z - spherePos.z };
	float diffLength = norm3df(diff.x, diff.y, diff.z);
	if (diffLength > sphereRad)
	{
		float4 result = { 0.0f, 0.0f, 0.0f, 1.0f };
		surf2Dwrite(result, surfObj, x * sizeof(result), y);
		return;
	}

	float halfChord = sqrtf(sphereRad * sphereRad - diffLength * diffLength);
	
	float intersectionLength = projection - halfChord;
	float3 intersection = { ray.x * intersectionLength, ray.y * intersectionLength, ray.z * intersectionLength };
	
	float3 normal = { intersection.x - spherePos.x, intersection.y - spherePos.y, intersection.z - spherePos.z };
	float normalLength = norm3df(normal.x, normal.y, normal.z);
	normal = { normal.x / normalLength, normal.y / normalLength, normal.z / normalLength };

	float3 lightDiff = { pointLightPos.x - intersection.x, pointLightPos.y - intersection.y, pointLightPos.z - intersection.z };
	float lightDiffLength = norm3df(lightDiff.x, lightDiff.y, lightDiff.z);

	float intensity = normal.x * lightDiff.x / lightDiffLength + normal.y * lightDiff.y / lightDiffLength + normal.z * lightDiff.z / lightDiffLength;
	if (intensity <= 0.0f)
		intensity = 0.0f;
	
	float4 result = { intensity, intensity, intensity, 1.0f };
	surf2Dwrite(result, surfObj, x * sizeof(result), y);
}

void CScene::UpdateTexture(float dt)
{
	cudaError_t error;

	dim3 blockDim(32, 32); //32*32 = 1024 threads per block
	dim3 gridDim((m_width - 1) / 32 + 1, (m_height - 1) / 32 + 1);

	cudaGraphicsResource* cuResource;
	error = cudaGraphicsGLRegisterImage(&cuResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
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


	renderFieldKernel << <gridDim, blockDim >> > (cuSurfaceObject, m_width, m_height);

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