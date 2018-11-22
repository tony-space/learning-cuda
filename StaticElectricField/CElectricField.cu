#include <GL\glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>

#include "CElectricField.hpp"

static __global__ void computeElectricVectorFieldKernel(float3* grid, unsigned width, unsigned height, float aspectRatio, CElectricField::SParticle* particles, size_t count)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	//get position coords from -1 to +1
	float2 pixelPosition =
	{
		((x / (float)(width - 1)) * 2.0f - 1.0f) * aspectRatio,
		(y / (float)(height - 1)) * 2.0f - 1.0f
	};

	float3 totalIntensity = {};

	for (size_t i = 0; i < count; ++i)
	{
		const CElectricField::SParticle p = particles[i];

		float dx = pixelPosition.x - p.position.x;
		float dy = pixelPosition.y - p.position.y;

		float distanceSqr = dx * dx + dy * dy;
		float invDistance = rsqrt(distanceSqr); // == 1 / distance

		if (distanceSqr < 0.0001f)
			continue;

		float scalarIntensity = p.charge / distanceSqr;

		float3 intensity =
		{
			scalarIntensity * (dx * invDistance),
			scalarIntensity * (dy * invDistance),
			//z component is for visual purpose only
			scalarIntensity
		};


		totalIntensity.x += intensity.x;
		totalIntensity.y += intensity.y;
		totalIntensity.z += intensity.z;
	}

	grid[x + y * width] = totalIntensity;
}

static __global__ void updateParticles(float3* grid, unsigned width, unsigned height, float aspectRatio, CElectricField::SParticle* particles, size_t count, float dt)
{
	auto index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= count) return;


	CElectricField::SParticle& p = particles[index];
	int2 pixel =
	{
		(int)roundf((p.position.x / aspectRatio + 1.0f) * (width - 1) / 2.0f),
		(int)roundf((p.position.y + 1.0f) * (height - 1) / 2.0f)
	};

	if (pixel.x < 0)
	{
		p.velocity.x = fabsf(p.velocity.x);
		p.position.x = -aspectRatio;
		pixel.x = 0;
	}
	if (pixel.y < 0)
	{
		p.velocity.y = fabsf(p.velocity.y);
		p.position.y = -1.0f;
		pixel.y = 0;
	}

	if (pixel.x >= width)
	{
		pixel.x = width - 1;
		p.position.x = aspectRatio;
		p.velocity.x = -fabsf(p.velocity.x);
	}

	if (pixel.y >= height)
	{
		pixel.y = height - 1;
		p.position.y = 1.0f;
		p.velocity.y = -fabsf(p.velocity.y);
	}

	const float drag = 0.05f;
	float3 intensity = grid[pixel.x + pixel.y * width];
	float2 force = { intensity.x * p.charge - p.velocity.x * drag, intensity.y * p.charge - p.velocity.y * drag };
	float2 accel = { force.x / p.mass, force.y / p.mass };

	if (fabsf(accel.x) > 100.0f)
		accel.x = fabsf(accel.x) / accel.x * 100.0f;

	if (fabsf(accel.y) > 100.0f)
		accel.y = fabsf(accel.y) / accel.y * 100.0f;

	p.velocity.x += accel.x * dt;
	p.velocity.y += accel.y * dt;

	p.position.x += p.velocity.x * dt;
	p.position.y += p.velocity.y * dt;
}

__global__ void renderFieldKernel(float3* grid, cudaSurfaceObject_t surfObj, unsigned width, unsigned height)
{
	auto x = blockIdx.x * blockDim.x + threadIdx.x;
	auto y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height)
	{
		float3 intensity = grid[x + y * width];
		float field = intensity.z / 2.5f;

		float4 result = { 0, 0, 0, 1.0f };

		if (field >= 0.0f)
		{
			result.x = field;
			result.y = field * 0.005f;
			result.z = field * 0.002f;
			if (result.y > 0.75f) result.y = 0.75f;
			if (result.z > 0.30f) result.z = 0.30f;
		}
		else
		{
			result.x = -field * 0.01f;
			result.y = -field * 0.01f;
			result.z = -field;

			if (result.x > 0.95f) result.x = 0.95f;
			if (result.y > 0.95f) result.y = 0.95f;
		}

		surf2Dwrite(result, surfObj, x * sizeof(result), y);
	}
}

void CElectricField::UpdateState(float dt)
{
	cudaError_t error;

	m_deviceVectorField.resize(m_width * m_height);

	dim3 blockDim(32, 32); //32*32 = 1024 threads per block
	dim3 gridDim((m_width - 1) / 32 + 1, (m_height - 1) / 32 + 1);

	computeElectricVectorFieldKernel <<<gridDim, blockDim >>> (m_deviceVectorField.data().get(), m_width, m_height, m_aspectRatio, m_deviceParticles.data().get(), m_deviceParticles.size());
	//error = cudaGetLastError();
	//assert(!error);

	updateParticles <<<1, unsigned(m_deviceParticles.size()) >>> (m_deviceVectorField.data().get(), m_width, m_height, m_aspectRatio, m_deviceParticles.data().get(), m_deviceParticles.size(), dt);
	//error = cudaGetLastError();
	//assert(!error);
}

void CElectricField::UpdateTexture()
{
	cudaError_t error;

	m_deviceVectorField.resize(m_width * m_height);

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


	renderFieldKernel << <gridDim, blockDim >> > (m_deviceVectorField.data().get(), cuSurfaceObject, m_width, m_height);

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