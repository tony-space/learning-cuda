#include <GL\glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <helper_math.h>

#include "Simulation.cuh"

namespace cudasim
{
	__global__ void processParticles(float3 posArray[], float3 velArray[], size_t particles, float dt)
	{
		auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
		if (threadId >= particles)
			return;

		float3 pos = posArray[threadId];
		float3 vel = velArray[threadId];

		pos += vel * dt;

		if (fabs(pos.x) >= 0.5f)
		{
			pos.x = copysignf(0.5f, pos.x);
			vel.x = -vel.x;
		}

		if (fabs(pos.y) >= 0.5f)
		{
			pos.y = copysignf(0.5f, pos.y);
			vel.y = -vel.y;
		}

		if (fabs(pos.z) >= 0.5f)
		{
			pos.z = copysignf(0.5f, pos.z);
			vel.z = -vel.z;
		}

		posArray[threadId] = pos;
		velArray[threadId] = vel;
	}

	void UpdateState(GLuint stateVBO, size_t particles, float dt)
	{
		cudaError_t error;

		cudaGraphicsResource_t resource;
		error = cudaGraphicsGLRegisterBuffer(&resource, stateVBO, cudaGraphicsRegisterFlagsNone);
		assert(error == cudaSuccess);

		error = cudaGraphicsMapResources(1, &resource);
		assert(error == cudaSuccess);

		void* d_state;
		size_t stateSize;
		error = cudaGraphicsResourceGetMappedPointer(&d_state, &stateSize, resource);
		assert(error == cudaSuccess);

		assert(particles * 3 * (sizeof(float) * 3) == stateSize);

		float3* d_positions = reinterpret_cast<float3*>(d_state);
		float3* d_velocities = reinterpret_cast<float3*>(d_state) + particles;

		dim3 blockDim(128);
		dim3 gridDim((unsigned(particles) - 1) / blockDim.x + 1);

		processParticles <<<gridDim, blockDim>>> (d_positions, d_velocities, particles, dt);

		error = cudaGraphicsUnmapResources(1, &resource);
		assert(error == cudaSuccess);

		error = cudaGraphicsUnregisterResource(resource);
		assert(error == cudaSuccess);
	}
}