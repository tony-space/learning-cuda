#include <GL/glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <helper_math.h>

#include "Simulation.hpp"

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

class CSimulation : public ISimulation
{
private:
	const GLuint m_stateVBO;
	const size_t m_particles;

	cudaGraphicsResource_t m_resource;
	float3* m_devicePositions;
	float3* m_deviceVelocities;

public:
	CSimulation(GLuint stateVBO, size_t particles) : m_stateVBO(stateVBO), m_particles(particles)
	{
		cudaError_t error;

		error = cudaGraphicsGLRegisterBuffer(&m_resource, m_stateVBO, cudaGraphicsRegisterFlagsNone);
		assert(error == cudaSuccess);

		error = cudaGraphicsMapResources(1, &m_resource);
		assert(error == cudaSuccess);

		void* d_stateVector;
		size_t stateSize;
		error = cudaGraphicsResourceGetMappedPointer(&d_stateVector, &stateSize, m_resource);
		assert(error == cudaSuccess);

		assert(m_particles * 3 * (sizeof(float) * 3) == stateSize);

		m_devicePositions = reinterpret_cast<float3*>(d_stateVector);
		m_deviceVelocities = reinterpret_cast<float3*>(d_stateVector) + m_particles;
	}

	virtual ~CSimulation() override
	{
		cudaError_t error;
		error = cudaGraphicsUnmapResources(1, &m_resource);
		assert(error == cudaSuccess);

		error = cudaGraphicsUnregisterResource(m_resource);
		assert(error == cudaSuccess);
	}
	virtual void UpdateState(float dt) override
	{
		dim3 blockDim(128);
		dim3 gridDim((unsigned(m_particles) - 1) / blockDim.x + 1);

		processParticles << <gridDim, blockDim >> > (m_devicePositions, m_deviceVelocities, m_particles, dt);
	}
};

std::unique_ptr<ISimulation> ISimulation::CreateInstance(GLuint stateVBO, size_t particles)
{
	return std::make_unique<CSimulation>(stateVBO, particles);
}