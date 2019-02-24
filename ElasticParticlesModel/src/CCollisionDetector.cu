#include <device_launch_parameters.h>

#include "CCollisionDetector.hpp"
#include "CSimulation.hpp"
//#include <cuda_runtime.h>

constexpr size_t kTileSize = 32;

template<typename T>
static inline __device__ __host__ T sqr(T x)
{
	return x * x;
}


__global__ void copyResultKernel(
	const cub::KeyValuePair<int, float>* __restrict__ timeReductionResult,
	SObjectsCollision* __restrict__ collisions,
	SObjectsCollision* __restrict__ copyResult)
{
	*copyResult = collisions[timeReductionResult->key];
}

__device__ float predictCollision(float3 selfPos, float3 selfVel, float3 otherPos, float3 otherVel, float rad)
{
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
	float c = dot(deltaR, deltaR) - sqr(rad * 2.0f);
	float discriminant = sqr(b) - 4.0f * a * c;

	//if particles don't move relatively each other (deltaV = 0)
	if (fabsf(a) <= 1e-6f)
		return INFINITY;

	//if particles are flying away
	if (b > 0.0f)
		return INFINITY;

	//if particles somehow have already penetrated one each other (e.g. due to incorrect position generation or numerical errors)
	if (c < 0.0f)
		return 0.0f;

	//if particles ways never intersect
	if (discriminant < 0.0f)
		return INFINITY;

	float sqrtD = sqrtf(discriminant);
	//Here is a tricky part.
	//You might think, why we even need to compute dt2 if it definitely is greater than dt1?
	//The answer is these two values can be negative, which means two contacts has already been somewhere in the past.
	float dt1 = (-b - sqrtD) / (2.0f * a);
	float dt2 = (-b + sqrtD) / (2.0f * a);

	float result = INFINITY;

	if (dt2 >= 0.0f)
		result = dt2;
	if (dt1 >= 0.0f)
		result = dt1;

	return result;
}

__global__ void predictParticleParticleCollisionsKernel(const SParticleSOA particles, float* __restrict__ resultTime, SObjectsCollision* __restrict__ resultCollisions)
{
	__shared__ float3 cachedPos[kTileSize];
	__shared__ float3 cachedVel[kTileSize];

	const auto threadId = threadIdx.y;
	const auto tileVerticalIdx = blockIdx.y * kTileSize;
	const auto verticalIdx = tileVerticalIdx + threadId;

	float3 selfPos = make_float3(0.0f);
	float3 selfVel = make_float3(0.0f);
	SObjectsCollision result;

	if (verticalIdx < particles.count)
	{
		selfPos = make_float3(particles.pos[verticalIdx]);
		selfVel = make_float3(particles.vel[verticalIdx]);
	}

	for (auto tileHorizontalIdx = tileVerticalIdx; tileHorizontalIdx < particles.count; tileHorizontalIdx += kTileSize)
	{
		const auto horizontalIdx = tileHorizontalIdx + threadId;

		if (horizontalIdx < particles.count)
		{
			cachedPos[threadId] = make_float3(particles.pos[horizontalIdx]);
			cachedVel[threadId] = make_float3(particles.vel[horizontalIdx]);
		}

		if (kTileSize > 32)
			__syncthreads();

		for (auto x = 0; x < kTileSize; ++x)
		{
			auto otherIdx = tileHorizontalIdx + x;
			if (verticalIdx >= particles.count)
				break;
			if (otherIdx >= particles.count)
				break;
			if (verticalIdx >= otherIdx)
				continue;


			const auto otherPos = cachedPos[x];
			const auto otherVel = cachedVel[x];

			float time = predictCollision(selfPos, selfVel, otherPos, otherVel, particles.radius);
			result.AnalyzeAndApply(verticalIdx, otherIdx, time, SObjectsCollision::CollisionType::ParticleToParticle);
		}

		if (kTileSize > 32)
			__syncthreads();
	}

	if (verticalIdx < particles.count)
	{
		resultTime[verticalIdx] = result.predictedTime;
		resultCollisions[verticalIdx] = result;
	}
}

__global__ void predictParticlePlaneCollisionsKernel(
	const SParticleSOA particles,
	const SPlane* __restrict__ planes,
	const size_t planesCount,
	float* __restrict__ resultTime,
	SObjectsCollision* __restrict__ resultCollisions)
{
	extern __shared__ SPlane cachedPlanes[];

	const auto threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < planesCount)
		cachedPlanes[threadIdx.x] = planes[threadIdx.x];

	__syncthreads();

	if (threadId >= particles.count)
		return;

	auto pos = make_float3(particles.pos[threadId]);
	auto vel = make_float3(particles.vel[threadId]);
	auto result = resultCollisions[threadId];

	for (size_t i = 0; i < planesCount; ++i)
	{
		auto plane = cachedPlanes[i];
		auto velProjection = dot(plane.normal(), vel);
		if (velProjection < 0.0f)
		{
			auto distance = plane.Distance(pos, particles.radius);
			float time = 0.0f;
			if (distance >= 0.0f)
				time = distance / -velProjection;

			result.AnalyzeAndApply(threadId, i, time, SObjectsCollision::CollisionType::ParticleToPlane);
		}
	}

	if (result.collisionType == SObjectsCollision::CollisionType::ParticleToPlane)
	{
		resultTime[threadId] = result.predictedTime;
		resultCollisions[threadId] = result;
	}
}

CCollisionDetector::ArgMinReduction::ArgMinReduction(size_t elementsCount)
{
	m_timeReductionResult = thrust::device_malloc<cub::KeyValuePair<int, float>>(1);
	m_timeValues.resize(elementsCount);
	m_collisionResult.resize(elementsCount);

	size_t tempStorageBytesSize = 0;
	auto status = cub::DeviceReduce::ArgMin(nullptr, tempStorageBytesSize, m_timeValues.data().get(), m_timeReductionResult.get(), int(m_timeValues.size()));
	assert(status == cudaSuccess);
	m_cubTemporaryStorage.resize(tempStorageBytesSize);
}

void CCollisionDetector::ArgMinReduction::Reduce()
{
	size_t tempStorageBytesSize = m_cubTemporaryStorage.size();
	auto status = cub::DeviceReduce::ArgMin(m_cubTemporaryStorage.data().get(), tempStorageBytesSize, m_timeValues.data().get(), m_timeReductionResult.get(), int(m_timeValues.size()));
	assert(status == cudaSuccess);
}

CCollisionDetector::CCollisionDetector(const SParticleSOA d_particles, const thrust::host_vector<SPlane>& worldBoundaries) :
	m_deviceParticles(d_particles),
	m_devicePlanes(worldBoundaries),
	m_reduction(d_particles.count)
{
	m_collisionResult = thrust::device_malloc<SObjectsCollision>(1);
}


SObjectsCollision* CCollisionDetector::FindEarliestCollision()
{
	cudaError_t status;
	auto particles = unsigned(m_deviceParticles.count);
	auto walls = unsigned(m_devicePlanes.size());

	dim3 blockDim;
	dim3 gridDim;

	blockDim = dim3(1, unsigned(kTileSize));
	gridDim = dim3(1, divCeil(particles, blockDim.y));
	predictParticleParticleCollisionsKernel << <gridDim, blockDim >> > (m_deviceParticles, m_reduction.m_timeValues.data().get(), m_reduction.m_collisionResult.data().get());
	status = cudaGetLastError();
	assert(status == cudaSuccess);

	blockDim = dim3(1024u);
	gridDim = dim3(divCeil(particles, blockDim.x));
	predictParticlePlaneCollisionsKernel << <gridDim, blockDim, sizeof(SPlane) * m_devicePlanes.size() >> > (
		m_deviceParticles,
		m_devicePlanes.data().get(), walls,
		m_reduction.m_timeValues.data().get(), m_reduction.m_collisionResult.data().get());
	status = cudaGetLastError();
	assert(status == cudaSuccess);

	m_reduction.Reduce();

	copyResultKernel <<<1, 1 >>> (m_reduction.m_timeReductionResult.get(), m_reduction.m_collisionResult.data().get(), m_collisionResult.get());

	return m_collisionResult.get();
}