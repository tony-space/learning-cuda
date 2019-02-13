#include <device_launch_parameters.h>

#include "CCollisionDetector.hpp"
#include "CSimulation.hpp"
//#include <cuda_runtime.h>

constexpr size_t kMaxReductionBlockSize = 256;
constexpr SObjectsCollision kDefaultValue = SObjectsCollision();

static inline __device__ __host__ float sqr(float x)
{
	return x * x;
}

static inline __device__ SObjectsCollision warpReduce(SObjectsCollision val)
{
	for (auto offset = warpSize >> 1; offset > 0; offset >>= 1)
	{
		auto neighbourTime = __shfl_down_sync(0xFFFFFFFF, val.predictedTime, offset);
		auto neighbourIndex1 = __shfl_down_sync(0xFFFFFFFF, val.object1, offset);
		auto neighbourIndex2 = __shfl_down_sync(0xFFFFFFFF, val.object2, offset);
		auto neighbourType = (SObjectsCollision::CollisionType)__shfl_down_sync(0xFFFFFFFF, (int)val.collisionType, offset);

		if (neighbourTime < val.predictedTime)
		{
			val.predictedTime = neighbourTime;
			val.object1 = neighbourIndex1;
			val.object2 = neighbourIndex2;
			val.collisionType = neighbourType;
		}
	}
	return val;
}

__global__ void reduceKernel(const SObjectsCollision* __restrict__ values, size_t size, SObjectsCollision* __restrict__ output)
{
	SObjectsCollision val;

	auto threadId = blockDim.x * blockIdx.x + threadIdx.x;
	auto warpId = threadIdx.x / unsigned(warpSize);
	auto laneId = threadIdx.x % warpSize;
	auto gridSize = blockDim.x * gridDim.x;

	extern __shared__ SObjectsCollision cache[];
	auto cacheSize = divCeil(blockDim.x, unsigned(warpSize)); //equals to amount of warps in blocks

	if (threadId < size)
		val = values[threadId];
	if (threadId + gridSize < size)
		val = SObjectsCollision::min(val,values[threadId + gridSize]);

	val = warpReduce(val);
	if (laneId == 0)
		cache[warpId] = val;

	if (warpId > 0)
		return;

	__syncthreads();

	val = laneId < cacheSize ? cache[laneId] : kDefaultValue;
	val = warpReduce(val);

	if (laneId == 0)
		output[blockIdx.x] = val;
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

CCollisionDetector::CCollisionDetector(const SParticleSOA d_particles, const thrust::host_vector<SPlane>& worldBoundaries) :
	m_deviceParticles(d_particles),
	m_devicePlanes(worldBoundaries)
{

	auto intermediateSize = divCeil(divCeil(d_particles.count, size_t(2)), kMaxReductionBlockSize);
	m_collisions.resize(d_particles.count);
	m_intermediate.resize(intermediateSize);
}

SObjectsCollision* CCollisionDetector::FindEarliestCollision()
{
	dim3 blockDim(64);
	dim3 gridDim(divCeil(unsigned(m_deviceParticles.count), blockDim.x));

	auto collisions = m_collisions.data().get();
	auto buffer1 = collisions;
	auto buffer2 = m_intermediate.data().get();

	predictParticleParticleCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, collisions);
	predictParticlePlaneCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_devicePlanes.data().get(), m_devicePlanes.size(), collisions);

	for (size_t particles = m_deviceParticles.count; particles > 1;)
	{
		size_t pairs = divCeil(particles, size_t(2));
		size_t warps = divCeil(pairs, size_t(32));
		dim3 blockSize(unsigned(min(kMaxReductionBlockSize, warps * size_t(32))));
		dim3 gridSize(unsigned(divCeil(pairs, size_t(blockSize.x))));

		reduceKernel <<<gridSize, blockSize, blockSize.x / 32 * sizeof(SObjectsCollision) >>> (buffer1, particles, buffer2);
		std::swap(buffer1, buffer2);
		particles = gridSize.x;
	}

	return buffer1;
}
