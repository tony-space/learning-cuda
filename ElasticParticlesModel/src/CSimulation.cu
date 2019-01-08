#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "CSimulation.hpp"

__device__ SParticle resolveParticle2ParticleCollision(SParticle& a, SParticle b)
{
	auto centerOfMassVel = (a.vel + b.vel) / 2.0f;
	auto v1 = a.vel - centerOfMassVel;
	auto v2 = b.vel - centerOfMassVel;

	auto planeNormal = normalize(b.pos - a.pos);

	v1 = reflect(v1, planeNormal) * 0.98f;
	v2 = reflect(v2, planeNormal) * 0.98f;

	a.vel = v1 + centerOfMassVel;
	b.vel = v2 + centerOfMassVel;

	return b;
}

__device__ void resolveParticle2PlaneCollision(SPlane plane, SParticle& particle)
{
	particle.vel = reflect(particle.vel, plane.normal) * 0.98f;
}

__global__ void moveParticlesKernel(SParticle* __restrict__ particles, const size_t particlesCount, float dt, const SObjectsCollision* __restrict__ earilestCollision)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	dt = fminf(earilestCollision->predictedTime, dt);

	SParticle self = particles[threadId];
	self.pos += self.vel * dt;
	self.vel.y -= 1.0f * dt;

	particles[threadId] = self;
}

__global__ void resolveCollisionsKernel(
	SParticle* __restrict__ particles,
	const size_t particlesCount,
	const float dt,
	const SObjectsCollision* __restrict__ pEarilestCollision,
	const SPlane* __restrict__ pPlanes)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	auto collision = *pEarilestCollision;

	if (dt < collision.predictedTime)
		return;

	if (collision.object1 != threadId)
		return;

	SParticle self = particles[threadId];

	switch (collision.collisionType)
	{
	case SObjectsCollision::CollisionType::ParticleToPlane:
		resolveParticle2PlaneCollision(pPlanes[collision.object2], self);
		break;

	case SObjectsCollision::CollisionType::ParticleToParticle:
		particles[collision.object2] = resolveParticle2ParticleCollision(self, particles[collision.object2]);
		break;
	}

	particles[threadId] = self;
}

CSimulation::CSimulation(void* d_particles, size_t particlesCount, float particleRadius) :
	m_deviceParticles(reinterpret_cast<SParticle*>(d_particles)),
	m_particlesCount(particlesCount),
	m_particleRadius(particleRadius)
{
	thrust::host_vector<SPlane> hostPlanes;
	hostPlanes.push_back(SPlane(make_float3(1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(-1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, -1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, 1.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, -1.0), -0.5));

	m_collisionDetector = std::make_unique<CCollisionDetector>(m_deviceParticles, m_particlesCount, m_particleRadius, hostPlanes);
}

float CSimulation::UpdateState(float dt)
{
	dim3 blockDim(64);
	dim3 gridDim((unsigned(m_particlesCount) - 1) / blockDim.x + 1);

	auto d_earliestCollistion = m_collisionDetector->FindEarliestCollision();

	moveParticlesKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, dt, d_earliestCollistion);
	resolveCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, dt, d_earliestCollistion, m_collisionDetector->GetPlanes());

	/*SObjectsCollision col;
	auto status = cudaMemcpy(&col, d_earliestCollistion, sizeof(col), cudaMemcpyDeviceToHost);
	assert(status == cudaSuccess);
	dt = fminf(dt, col.predictedTime);*/

	return dt;
}

std::unique_ptr<ISimulation> ISimulation::CreateInstance(void* d_particles, size_t particlesCount, float particleRadius)
{
	return std::make_unique<CSimulation>(d_particles, particlesCount, particleRadius);
}