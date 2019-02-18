#include <device_launch_parameters.h>

#include "CCollisionDetector.hpp"
#include "CSimulation.hpp"
//#include <cuda_runtime.h>

static inline __device__ __host__ float sqr(float x)
{
	return x * x;
}

__global__ void predictParticleParticleCollisionsKernel(const SParticleSOA particles, SObjectsCollision* __restrict__ out)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particles.count)
		return;

	auto selfPos = particles.pos[threadId];
	auto selfVel = particles.vel[threadId];
	auto selfRad = particles.radius[threadId];

	SObjectsCollision earliestCollision;

	for (size_t i = threadId + 1; i < particles.count; ++i)
	{
		//if (i == threadId) continue;
		auto otherPos = particles.pos[i];
		auto otherVel = particles.vel[i];
		auto otherRad = particles.radius[i];

		//Let's solve a quadratic equation to predict the exact collision time.
		//The quadric equation can be get from the following vector equation:
		//(R1 + V1 * dt) - (R2 + V2 * dt) = rad1 + rad2  : the distance between new positions equals the sum of two radii
		//where R1 and R2 are radius vectors of the current particles position
		//      V1 and V2 are velocity vectors
		//      rad1 and rad2 are particles' radii
		//      dt is the unknown variable
		//Vector dot product satisfies a distributive law.

		float3 deltaR = selfPos - otherPos;
		float3 deltaV = selfVel - otherVel;

		//Quadratic equation coefficients
		float a = dot(deltaV, deltaV);
		float b = 2.0f * dot(deltaR, deltaV);
		float c = dot(deltaR, deltaR) - sqr(selfRad + otherRad);
		float discriminant = sqr(b) - 4.0f * a * c;

		//if particles don't move relatively each other (deltaV = 0)
		if (fabsf(a) <= CSimulation::kSimPrecision)
			continue;

		//if particles are flying away, don't check collision
		if (b > 0.0f)
			continue;

		//if particles somehow have already penetrated each other (e.g. due to incorrect position generation), don't check collision.
		//It's just a check for invalid states
		if (c < 0.0f)
		{
			earliestCollision.AnalyzeAndApply(threadId, i, 0.0f, SObjectsCollision::CollisionType::ParticleToParticle);
			break;
			//continue;
		}

		//if particles ways never intersect
		if (discriminant < 0.0f)
			continue;

		float sqrtD = sqrtf(discriminant);
		//Here is a tricky part.
		//You might think, why we even need to compute dt2 if it definitely is greater than dt1?
		//The answer is these two values can be negative, which means two contacts has already been somewhere in the past.
		float dt1 = (-b - sqrtD) / (2.0f * a);
		float dt2 = (-b + sqrtD) / (2.0f * a);

		earliestCollision.AnalyzeAndApply(threadId, i, dt1, SObjectsCollision::CollisionType::ParticleToParticle);
		earliestCollision.AnalyzeAndApply(threadId, i, dt2, SObjectsCollision::CollisionType::ParticleToParticle);
	}

	out[threadId] = earliestCollision;
}

__global__ void predictParticlePlaneCollisionsKernel(
	const SParticleSOA particles,
	const SPlane* __restrict__ planes,
	const size_t planesCount,
	SObjectsCollision* __restrict__ inOut)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particles.count)
		return;

	auto pos = particles.pos[threadId];
	auto vel = particles.vel[threadId];
	auto rad = particles.radius[threadId];

	SObjectsCollision earliestCollision = inOut[threadId];

	for (size_t i = 0; i < planesCount; ++i)
	{
		SPlane plane = planes[i];

		auto velProjection = dot(plane.normal, vel);

		if (velProjection >= 0.0f)
			continue;

		auto time = max(-plane.Distance(pos, rad) / velProjection, 0.0f);
		earliestCollision.AnalyzeAndApply(threadId, i, time, SObjectsCollision::CollisionType::ParticleToPlane);
	}

	inOut[threadId] = earliestCollision;
}

__global__ void extractCollisionsTimeKernel(const SObjectsCollision* __restrict__ collisions, const size_t count, float* __restrict__ out)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= count)
		return;

	out[threadId] = collisions[threadId].predictedTime;
}

__global__ void copyResultKernel(
	const SObjectsCollision* __restrict__ collisions,
	const cub::KeyValuePair<int, float>* reductionResult,
	SObjectsCollision* __restrict__ collisionResult)
{
	*collisionResult = collisions[reductionResult->key];
}

CCollisionDetector::CCollisionDetector(const SParticleSOA d_particles, const thrust::host_vector<SPlane>& worldBoundaries) :
	m_deviceParticles(d_particles),
	m_devicePlanes(worldBoundaries)
{
	m_collisions.resize(d_particles.count);
	m_mappedCollisionTimes.resize(d_particles.count);
	m_reductionResult = thrust::device_malloc<cub::KeyValuePair<int, float>>(1);
	m_collisionResult = thrust::device_malloc<SObjectsCollision>(1);

	size_t tempStorageBytesSize = 0;
	auto in = m_mappedCollisionTimes.data().get();
	auto out = m_reductionResult.get();

	auto status = cub::DeviceReduce::ArgMin(nullptr, tempStorageBytesSize, in, out, int(d_particles.count));
	assert(status == cudaSuccess);
	m_cubTemporaryStorage.resize(tempStorageBytesSize);
}

SObjectsCollision* CCollisionDetector::FindEarliestCollision()
{
	dim3 blockDim(64);
	dim3 gridDim(divCeil(unsigned(m_deviceParticles.count), blockDim.x));

	auto collisions = m_collisions.data().get();
	auto planes = m_devicePlanes.data().get();
	auto timeArray = m_mappedCollisionTimes.data().get();
	auto reductionResult = m_reductionResult.get();
	auto result = m_collisionResult.get();

	predictParticleParticleCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, collisions);
	predictParticlePlaneCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, planes, m_devicePlanes.size(), collisions);
	extractCollisionsTimeKernel <<<gridDim, blockDim >>> (collisions, m_deviceParticles.count, timeArray);

	size_t tempStorageBytesSize = m_cubTemporaryStorage.size();
	auto status = cub::DeviceReduce::ArgMin(m_cubTemporaryStorage.data().get(), tempStorageBytesSize, timeArray, reductionResult, int(m_deviceParticles.count));
	assert(status == cudaSuccess);
	copyResultKernel <<<1, 1 >>> (collisions, reductionResult, result);

	return result;
}
