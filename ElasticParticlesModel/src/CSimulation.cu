#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "CSimulation.hpp"

__device__ void resolveParticle2ParticleCollision(const float3& pos1, float3& vel1, const float3& pos2, float3& vel2)
{
	auto centerOfMassVel = (vel1 + vel2) / 2.0f;
	auto v1 = vel1 - centerOfMassVel;
	auto v2 = vel2 - centerOfMassVel;

	auto planeNormal = normalize(pos1 - pos2);

	v1 = reflect(v1, planeNormal);
	v2 = reflect(v2, planeNormal);

	vel1 = v1 + centerOfMassVel;
	vel2 = v2 + centerOfMassVel;
}

__device__ __constant__ const SObjectsCollision earilestCollision;

__global__ void moveParticlesKernel(SParticleSOA particles, float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particles.count)
		return;

	dt = fminf(earilestCollision.predictedTime, dt);

	auto pos = particles.pos[threadId];
	auto vel = particles.vel[threadId];

	pos += vel * dt;
	particles.pos[threadId] = pos;
}

__global__ void resolveCollisionsKernel(
	SParticleSOA particles,
	const float dt,
	const SPlane* __restrict__ pPlanes)
{
	if (dt < earilestCollision.predictedTime)
		return;

	auto pos1 = particles.pos[earilestCollision.object1];
	auto vel1 = particles.vel[earilestCollision.object1];

	switch (earilestCollision.collisionType)
	{
	case SObjectsCollision::CollisionType::ParticleToPlane:
		vel1 = reflect(vel1, pPlanes[earilestCollision.object2].normal);
		break;

	case SObjectsCollision::CollisionType::ParticleToParticle:
		auto pos2 = particles.pos[earilestCollision.object2];
		auto vel2 = particles.vel[earilestCollision.object2];

		resolveParticle2ParticleCollision(pos1, vel1, pos2, vel2);
		particles.vel[earilestCollision.object2] = vel2;

		break;
	}

	particles.vel[earilestCollision.object1] = vel1;
}

CSimulation::CSimulation(SParticleSOA d_particles) : m_deviceParticles(d_particles)
{
	thrust::host_vector<SPlane> hostPlanes;
	hostPlanes.push_back(SPlane(make_float3(1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(-1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, -1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, 1.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, -1.0), -0.5));

	m_collisionDetector = std::make_unique<CCollisionDetector>(m_deviceParticles, hostPlanes);
}

float CSimulation::UpdateState(float dt)
{
	dim3 blockDim(64);
	dim3 gridDim((unsigned(m_deviceParticles.count) - 1) / blockDim.x + 1);

	auto d_earliestCollistion = m_collisionDetector->FindEarliestCollision();
	auto status = cudaMemcpyToSymbolAsync(earilestCollision, d_earliestCollistion, sizeof(SObjectsCollision), 0, cudaMemcpyDeviceToDevice);
	assert(status == cudaSuccess);
	moveParticlesKernel << <gridDim, blockDim >> > (m_deviceParticles, dt);
	resolveCollisionsKernel <<<1, 1 >>> (m_deviceParticles, dt, m_collisionDetector->GetPlanes());

	return dt;
}

std::unique_ptr<ISimulation> ISimulation::CreateInstance(SParticleSOA d_particles)
{
	return std::make_unique<CSimulation>(d_particles);
}