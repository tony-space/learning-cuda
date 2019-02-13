#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "CSimulation.hpp"
#include <cub/device/device_segmented_reduce.cuh>

__global__ void rebuildSprings(const SParticle* __restrict__ particles, const size_t particlesCount, const float particleRadius, bool* __restrict__ springsMat)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particlesCount || j >= particlesCount)
		return;

	bool result = false;

	auto matIndex = i * particlesCount + j;

	if (i != j)
	{
		auto connected = springsMat[matIndex];
		SParticle p1 = particles[i];
		SParticle p2 = particles[j];

		auto r = p1.pos - p2.pos;
		auto magnitude = length(r);
		auto diameter = particleRadius * 2.0f;
		if (!connected)
		{
			result = magnitude < diameter;
		}
		else
		{
			result = magnitude <= diameter * 1.25f;
		}
	}

	springsMat[matIndex] = result;
}

__global__ void computeForcesMatrix(const SParticle* __restrict__ particles, const size_t particlesCount, const float particleRadius, const bool* __restrict__ springsMat,
	float3* __restrict__ forcesMat)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particlesCount || j >= particlesCount)
		return;

	auto matIndex = i * particlesCount + j;

	float3 result = make_float3(0.0f);

	if (i != j)
	{
		bool connected = springsMat[matIndex];
		SParticle p1 = particles[i];
		SParticle p2 = particles[j];

		auto r = p1.pos - p2.pos;
		auto magnitude = length(r);
		auto diameter = particleRadius * 2.0f;

		if (!connected && magnitude > diameter)
		{
			constexpr float k = 8e-7f;
			result = -k * r / (magnitude * magnitude * magnitude);
		}
		else
		{
			constexpr float stiffness = 100.0f;
			constexpr float damp = 1.0f;

			auto delta = magnitude - diameter;
			auto v = dot(p1.vel - p2.vel, r / magnitude);

			result = (-r / magnitude) * (delta * stiffness + v * damp);
		}
	}

	forcesMat[matIndex] = result;
}

__global__ void moveParticlesKernel2(
	SParticle* __restrict__ particles, const float3* __restrict__ forces, const size_t particlesCount, const float particleRadius,
	const SPlane* __restrict__ planes, const size_t planesCount,
	const float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];
	float3 force = forces[threadId];
	//force.y -= 0.1f; // gravity

	force -= 0.01f * self.vel;

	self.pos += self.vel * dt;
	self.vel += force * dt;

	for (size_t i = 0; i < planesCount; ++i)
	{
		SPlane p = planes[i];
		if (p.Distance(self, particleRadius) >= 0.0f)
			continue;

		if (dot(p.normal, self.vel) >= 0.0f)
			continue;

		self.vel = reflect(self.vel, p.normal) * 0.75f;
		break;
	}

	particles[threadId] = self;
}

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
	m_devicePlanes = hostPlanes;

	m_collisionDetector = std::make_unique<CCollisionDetector>(m_deviceParticles, m_particlesCount, m_particleRadius, hostPlanes);

	m_deviceForces.resize(m_particlesCount, make_float3(0.0f));
	m_deviceForcesMatrix.resize(m_particlesCount * m_particlesCount, make_float3(0.0f));
	m_deviceSpringsMatrix.resize(m_particlesCount * m_particlesCount, false);

	thrust::host_vector<size_t> hostSegments(m_particlesCount + 1);
	for (size_t i = 0; i < m_particlesCount + 1; ++i)
		hostSegments[i] = i * m_particlesCount;

	m_deviceReductionSegments = hostSegments;
}

//float CSimulation::UpdateState(float dt)
//{
//	dim3 blockDim(64);
//	dim3 gridDim((unsigned(m_particlesCount) - 1) / blockDim.x + 1);
//
//	auto d_earliestCollistion = m_collisionDetector->FindEarliestCollision();
//
//	moveParticlesKernel << <gridDim, blockDim >> > (m_deviceParticles, m_particlesCount, dt, d_earliestCollistion);
//	resolveCollisionsKernel << <gridDim, blockDim >> > (m_deviceParticles, m_particlesCount, dt, d_earliestCollistion, m_collisionDetector->GetPlanes());
//
//	/*SObjectsCollision col;
//	auto status = cudaMemcpy(&col, d_earliestCollistion, sizeof(col), cudaMemcpyDeviceToHost);
//	assert(status == cudaSuccess);
//	dt = fminf(dt, col.predictedTime);*/
//
//	return dt;
//}

float CSimulation::UpdateState(float dt)
{
	cudaError_t cudaStatus;
	auto blocks = unsigned((m_particlesCount - 1) / 32 + 1);

	dim3 blockDim(32, 32);
	dim3 gridDim(blocks, blocks);

	rebuildSprings << <gridDim, blockDim >> > (m_deviceParticles, m_particlesCount, m_particleRadius, m_deviceSpringsMatrix.data().get());

	computeForcesMatrix << <gridDim, blockDim >> > (m_deviceParticles, m_particlesCount, m_particleRadius, m_deviceSpringsMatrix.data().get(), m_deviceForcesMatrix.data().get());

	size_t storageBytes = 0;
	cudaStatus = cub::DeviceSegmentedReduce::Sum(
		nullptr, storageBytes,
		m_deviceForcesMatrix.data().get(), m_deviceForces.data().get(),
		int(m_particlesCount), m_deviceReductionSegments.data().get(), m_deviceReductionSegments.data().get() + 1);
	assert(cudaStatus == cudaSuccess);

	if (m_segmentedReductionStorage.size() != storageBytes)
		m_segmentedReductionStorage.resize(storageBytes);

	cudaStatus = cub::DeviceSegmentedReduce::Sum(
		m_segmentedReductionStorage.data().get(), storageBytes,
		m_deviceForcesMatrix.data().get(), m_deviceForces.data().get(),
		int(m_particlesCount), m_deviceReductionSegments.data().get(), m_deviceReductionSegments.data().get() + 1);
	assert(cudaStatus == cudaSuccess);

	blockDim = dim3(64);
	gridDim = dim3((unsigned(m_particlesCount) - 1) / blockDim.x + 1);

	moveParticlesKernel2 << <gridDim, blockDim >> > (m_deviceParticles, m_deviceForces.data().get(), m_particlesCount, m_particleRadius, m_devicePlanes.data().get(), m_devicePlanes.size(), dt);

	return dt;
}

std::unique_ptr<ISimulation> ISimulation::CreateInstance(void* d_particles, size_t particlesCount, float particleRadius)
{
	return std::make_unique<CSimulation>(d_particles, particlesCount, particleRadius);
}