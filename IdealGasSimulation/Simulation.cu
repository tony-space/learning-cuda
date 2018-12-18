#include <GL/glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <helper_math.h>

#include "Simulation.hpp"

struct SParticle
{
	float3 pos;
	float3 vel;
};


__global__ void processParticles(SParticle particles[], size_t particlesCount, float particleRadius, float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	auto particle = particles[threadId];
	auto& pos = particle.pos;
	auto& vel = particle.vel;
	
	pos += vel * dt;

	auto penetration = (fabs(pos) + particleRadius) - 0.5f;
	if (penetration.x >= 0.0f)
	{
		pos.x = copysignf(0.5f - particleRadius - penetration.x, pos.x);
		vel.x = -vel.x;
	}

	if (penetration.y >= 0.0f)
	{
		pos.y = copysignf(0.5f - particleRadius - penetration.y, pos.y);
		vel.y = -vel.y;
	}

	if (penetration.z >= 0.0f)
	{
		pos.z = copysignf(0.5f - particleRadius - penetration.z, pos.z);
		vel.z = -vel.z;
	}

	particles[threadId] = particle;
}

class CSimulation : public ISimulation
{
private:
	const GLuint m_stateVBO;
	const size_t m_particlesCount;
	const float m_particleRadius;

	cudaGraphicsResource_t m_resource;
	SParticle* m_deviceParticles;

public:
	CSimulation(GLuint stateVBO, size_t particlesCount, float particleRadius) : m_stateVBO(stateVBO), m_particlesCount(particlesCount), m_particleRadius(particleRadius)
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

		m_deviceParticles = reinterpret_cast<SParticle*>(d_stateVector);
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
		dim3 gridDim((unsigned(m_particlesCount) - 1) / blockDim.x + 1);

		processParticles <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, m_particleRadius, dt);
	}
};

std::unique_ptr<ISimulation> ISimulation::CreateInstance(GLuint stateVBO, size_t particlesCount, float particleRadius)
{
	return std::make_unique<CSimulation>(stateVBO, particlesCount, particleRadius);
}