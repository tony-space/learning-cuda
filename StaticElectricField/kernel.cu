#include <GL\glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>
#include <cuda_gl_interop.h>

#include "kernel.hpp"

__global__ void testFetchKernel(float* output, cudaTextureObject_t texObj, int width, int height)
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
	error = cudaMemcpyToArray(cuArray, 0, 0, pixels, sizeof(pixels), cudaMemcpyHostToDevice);

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

	float* d_output;
	error = cudaMalloc(&d_output, 2 * 2 * 4 * sizeof(float));

	dim3 block(2, 2);
	testFetchKernel <<<1, block >>> (d_output, texObj, 2, 2);
	error = cudaGetLastError();
	error = cudaDeviceSynchronize();

	for (float& p : pixels) p = 0.0f;

	error = cudaMemcpy(pixels, d_output, sizeof(pixels), cudaMemcpyDeviceToHost);

	error = cudaFree(d_output);
	error = cudaDestroyTextureObject(texObj);
	error = cudaFreeArray(cuArray);
}


__global__ void testOpenGLFetchKernel(unsigned char* output, cudaTextureObject_t texObj, int width, int height)
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

	error = cudaGraphicsMapResources(1, &cuResource);

	cudaArray* cuArray;
	error = cudaGraphicsSubResourceGetMappedArray(&cuArray, cuResource, 0, 0);

	cudaChannelFormatDesc channelDesc;
	error = cudaGetChannelDesc(&channelDesc, cuArray);

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

	unsigned char* d_output;
	error = cudaMalloc(&d_output, 2 * 2 * 4 * sizeof(float));

	unsigned char pixels[2 * 2 * 4] = {};

	dim3 block(2, 2);
	testOpenGLFetchKernel <<<1, block >>>(d_output, texObj, 2, 2);
	error = cudaGetLastError();
	error = cudaDeviceSynchronize();

	error = cudaMemcpy(pixels, d_output, sizeof(pixels), cudaMemcpyDeviceToHost);

	error = cudaFree(d_output);
	error = cudaDestroyTextureObject(texObj);
	error = cudaGraphicsUnmapResources(1, &cuResource);
	error = cudaGraphicsUnregisterResource(cuResource);
}