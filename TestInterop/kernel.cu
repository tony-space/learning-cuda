#include <GL\glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>
#include <math_constants.h>

#include <cassert>
#include <chrono>

#include "kernel.hpp"

__global__ void textureFetchKernel(float* output, cudaTextureObject_t texObj, int width, int height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	float4 pixel = tex2D<float4>(texObj, u, v);
	pixel.x *= 2;
	pixel.y *= 2;
	pixel.z *= 2;
	((float4*)output)[y * width + x] = pixel;
}

void TextureFetchTest()
{
	float pixels[] =
	{
		1.0f, 1.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f
	};

	cudaError_t error;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	error = cudaMallocArray(&cuArray, &channelDesc, 2, 2);
	assert(!error);

	error = cudaMemcpyToArray(cuArray, 0, 0, pixels, sizeof(pixels), cudaMemcpyHostToDevice);
	assert(!error);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	error = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
	assert(!error);

	float* d_output;
	error = cudaMalloc(&d_output, 2 * 2 * 4 * sizeof(float));
	assert(!error);

	dim3 block(2, 2);
	textureFetchKernel <<<1, block>>> (d_output, texObj, 2, 2);
	error = cudaGetLastError();
	assert(!error);

	error = cudaDeviceSynchronize();
	assert(!error);

	for (float& p : pixels) p = 0.0f;

	error = cudaMemcpy(pixels, d_output, sizeof(pixels), cudaMemcpyDeviceToHost);
	assert(!error);

	error = cudaFree(d_output);
	assert(!error);

	error = cudaDestroyTextureObject(texObj);
	assert(!error);

	error = cudaFreeArray(cuArray);
	assert(!error);
}


__global__ void openGLTextureFetchKernel(unsigned char* output, cudaTextureObject_t texObj, unsigned width, unsigned height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	char4 pixel = tex2D<char4>(texObj, u, v);
	((char4*)output)[y * width + x] = pixel;
}

void OpenGLTextureFetchTest(unsigned textureId)
{
	cudaError_t error;

	cudaGraphicsResource* cuResource;
	error = cudaGraphicsGLRegisterImage(&cuResource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone);
	assert(!error);

	error = cudaGraphicsMapResources(1, &cuResource);
	assert(!error);

	cudaArray* cuArray;
	error = cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);
	assert(!error);

	cudaChannelFormatDesc channelDesc;
	error = cudaGetChannelDesc(&channelDesc, cuArray);
	assert(!error);

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	error = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
	assert(!error);

	unsigned char* d_output;
	error = cudaMalloc(&d_output, 2 * 2 * 4 * sizeof(float));
	assert(!error);

	unsigned char pixels[2 * 2 * 4] = {};

	dim3 blockDim(2, 2);
	openGLTextureFetchKernel <<<1, blockDim>>> (d_output, texObj, 2, 2);
	error = cudaGetLastError();
	assert(!error);

	error = cudaDeviceSynchronize();
	assert(!error);

	error = cudaMemcpy(pixels, d_output, sizeof(pixels), cudaMemcpyDeviceToHost);
	assert(!error);

	error = cudaFree(d_output);
	assert(!error);

	error = cudaDestroyTextureObject(texObj);
	assert(!error);

	error = cudaGraphicsUnmapResources(1, &cuResource);
	assert(!error);

	error = cudaGraphicsUnregisterResource(cuResource);
	assert(!error);
}

__global__ void pboGeneratorKernel(float4* storage, unsigned width, unsigned height, float time)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	float4 result = {
		cos(u * CUDART_PI_F * 2.0f * time + time) / 2.0f + 0.5f,
		0,
		sin(v * CUDART_PI_F * 2.0f * time + time) / 2.0f + 0.5f,
		1 };

	storage[y * width + x] = result;
}

void GeneratePBO(unsigned pboUnpackedBuffer, unsigned width, unsigned height)
{
	cudaError_t error;
	cudaGraphicsResource* cuResource;

	error = cudaGraphicsGLRegisterBuffer(&cuResource, pboUnpackedBuffer, cudaGraphicsRegisterFlagsWriteDiscard);
	assert(!error);

	error = cudaGraphicsMapResources(1, &cuResource);
	assert(!error);

	void* d_pboStorage;
	size_t storageSize;
	error = cudaGraphicsResourceGetMappedPointer(&d_pboStorage, &storageSize, cuResource);
	assert(!error);

	static auto startTime = std::chrono::system_clock::now();
	auto now = std::chrono::system_clock::now();
	auto delta = now - startTime;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
	float time = milliseconds.count() / 1000.0f;

	dim3 gridDim(width / 32, height / 32);
	dim3 blockDim(32, 32);
	pboGeneratorKernel <<<gridDim, blockDim>>> ((float4*)d_pboStorage, width, height, time);

	error = cudaGetLastError();
	assert(!error);

	error = cudaDeviceSynchronize();
	assert(!error);

	error = cudaGraphicsUnmapResources(1, &cuResource);
	assert(!error);

	error = cudaGraphicsUnregisterResource(cuResource);
	assert(!error);
}


__global__ void openGlTextureModifier(cudaSurfaceObject_t surfObj, unsigned width, unsigned height, float time)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	float u = x / (float)width;
	float v = y / (float)height;

	float value1 = sin(v * CUDART_PI_F * 2.0f * time + time) / 2.0f + 0.5f;
	float value2 = cos(u * CUDART_PI_F * 2.0f * time + time) / 2.0f + 0.5f;
	float value3 = value1 + value2;

	float4 result = {
		value1,
		value2,
		value3,
		1 };

	surf2Dwrite(result, surfObj, x * sizeof(result), y);
}

void ModifyTexture(unsigned textureId, unsigned width, unsigned height)
{
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
	
	static auto startTime = std::chrono::system_clock::now();
	auto now = std::chrono::system_clock::now();
	auto delta = now - startTime;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
	float time = milliseconds.count() / 1000.0f;

	dim3 gridDim(width / 32, height / 32);
	dim3 blockDim(32, 32);
	openGlTextureModifier <<<gridDim, blockDim>>> (cuSurfaceObject, width, height, time);

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