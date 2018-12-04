#include <GL\glew.h>

//#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <helper_math.h>

#include "CScene.hpp"

__device__ const float3 spherePos = { 0.0f, 0.0f, -1.5f };
__device__ const float sphereRad = 1.0f;

__global__ void rayTracingKernel(cudaSurfaceObject_t surfObj, unsigned width, unsigned height, float time)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float3 ray = normalize(make_float3(float(x) / float(width - 1) * 2.0f - 1.0f, float(y) / float(height - 1) * 2.0f - 1.0f, -1.0f));

	float projection = dot(spherePos, ray);
	float3 sphereCenterProj = ray * projection;
	float3 diff = sphereCenterProj - spherePos;
	float diffLength = norm3df(diff.x, diff.y, diff.z);
	if (diffLength > sphereRad)
	{
		float4 result = { 0.0f, 0.0f, 0.0f, 1.0f };
		surf2Dwrite(result, surfObj, x * sizeof(result), y);
		return;
	}

	float halfChord = sqrtf(sphereRad * sphereRad - diffLength * diffLength);
	
	float intersectionLength = projection - halfChord;
	float3 intersection = ray * intersectionLength;
	float3 normal = normalize(intersection - spherePos);

	float3 pointLightPos = make_float3(sinf(time) * 3.0f, 2.0f, cosf(time) * 3.0f) + spherePos;

	float intensity = dot(normal, normalize(pointLightPos - intersection));

	if (intensity <= 0.0f)
		intensity = 0.0f;
	
	float4 result = { intensity, intensity, intensity, 1.0f };
	surf2Dwrite(result, surfObj, x * sizeof(result), y);
}

void CScene::UpdateTexture(float dt)
{
	m_time += dt;
	cudaError_t error;

	dim3 blockDim(8, 8);
	dim3 gridDim((m_width - 1) / blockDim.x + 1, (m_height - 1) / blockDim.y + 1);

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

	rayTracingKernel <<<gridDim, blockDim>>>(cuSurfaceObject, m_width, m_height, m_time);

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