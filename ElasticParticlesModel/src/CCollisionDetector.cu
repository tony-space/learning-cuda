#include <device_launch_parameters.h>

#include "CCollisionDetector.hpp"
#include "CSimulation.hpp"
//#include <cuda_runtime.h>

template<typename T>
static inline __device__ __host__ T sqr(T x)
{
	return x * x;
}


__global__ void copyResultKernel(
	const cub::KeyValuePair<int, float>* particlesReductionResult,
	const cub::KeyValuePair<int, float>* wallsReductionResult,
	size_t particles,
	size_t walls,
	SObjectsCollision* __restrict__ collisionResult)
{
	auto particlesResult = *particlesReductionResult;
	auto wallsResult = *wallsReductionResult;

	SObjectsCollision result;

	if (particlesResult.value < wallsResult.value)
	{
		const size_t matIndex = size_t(particlesResult.key);
		auto i = matIndex / particles;
		auto j = matIndex % particles;

		result.collisionType = SObjectsCollision::CollisionType::ParticleToParticle;
		result.object1 = i;
		result.object2 = j;
		result.predictedTime = particlesResult.value;
	}
	else
	{
		const size_t matIndex = size_t(wallsResult.key);
		auto wall = matIndex / particles;
		auto particle = matIndex % particles;

		result.collisionType = SObjectsCollision::CollisionType::ParticleToPlane;
		result.object1 = particle;
		result.object2 = wall;
		result.predictedTime = wallsResult.value;
	}

	*collisionResult = result;
}

__global__ void predictParticleParticleCollisionsKernel(const SParticleSOA particles, float* __restrict__ matrix)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	auto threadId = i * particles.count + j;

	if (i >= particles.count || j >= particles.count)
		return;

	float result = INFINITY;

	if (i > j)
	{
		auto selfPos = particles.pos[i];
		auto selfVel = particles.vel[i];
		auto selfRad = particles.radius[i];

		auto otherPos = particles.pos[j];
		auto otherVel = particles.vel[j];
		auto otherRad = particles.radius[j];

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
		if (fabsf(a) <= 0.0f)
		{
			matrix[threadId] = INFINITY;
			return;
		}
		//if particles are flying away
		if (b > 0.0f)
		{
			matrix[threadId] = INFINITY;
			return;
		}
		//if particles somehow have already penetrated one each other (e.g. due to incorrect position generation or numerical errors)
		if (c < 0.0f)
		{
			matrix[threadId] = 0.0f;
			return;
		}
		//if particles ways never intersect
		if (discriminant < 0.0f)
		{
			matrix[threadId] = INFINITY;
			return;
		}

		float sqrtD = sqrtf(discriminant);
		//Here is a tricky part.
		//You might think, why we even need to compute dt2 if it definitely is greater than dt1?
		//The answer is these two values can be negative, which means two contacts has already been somewhere in the past.
		float dt1 = (-b - sqrtD) / (2.0f * a);
		float dt2 = (-b + sqrtD) / (2.0f * a);

		if (dt2 >= 0.0f)
			result = dt2;
		if (dt1 >= 0.0f)
			result = dt1;
	}
	matrix[threadId] = result;
}

__global__ void predictParticlePlaneCollisionsKernel(
	const SParticleSOA particles,
	const SPlane* __restrict__ planes,
	const size_t planesCount,
	float* __restrict__ matrix)
{
	auto planeId = blockIdx.y * blockDim.y + threadIdx.y;
	auto particleId = blockIdx.x * blockDim.x + threadIdx.x;
	auto threadId = planeId * particles.count + particleId;

	if (planeId >= planesCount || particleId >= particles.count)
		return;

	auto pos = particles.pos[particleId];
	auto vel = particles.vel[particleId];
	auto rad = particles.radius[particleId];
	auto plane = planes[planeId];
	float result = INFINITY;

	auto velProjection = dot(plane.normal, vel);
	if (velProjection < 0.0f)
		result = plane.Distance(pos, rad) / -velProjection;

	matrix[threadId] = result;
}

CCollisionDetector::ArgMinReduction::ArgMinReduction(size_t rows, size_t columns)
{
	m_reductionResult = thrust::device_malloc<cub::KeyValuePair<int, float>>(1);

	m_matrix.resize(rows * columns);

	size_t tempStorageBytesSize = 0;
	auto status = cub::DeviceReduce::ArgMin(nullptr, tempStorageBytesSize, m_matrix.data().get(), m_reductionResult.get(), int(m_matrix.size()));
	assert(status == cudaSuccess);
	m_cubTemporaryStorage.resize(tempStorageBytesSize);
}

void CCollisionDetector::ArgMinReduction::Reduce()
{
	//thrust::host_vector<float> debugVec = m_matrix;
	//for (auto f : debugVec)
	//	printf("%.2f ", f);
	//printf("\n");

	size_t tempStorageBytesSize = m_cubTemporaryStorage.size();
	auto status = cub::DeviceReduce::ArgMin(m_cubTemporaryStorage.data().get(), tempStorageBytesSize, m_matrix.data().get(), m_reductionResult.get(), int(m_matrix.size()));
	assert(status == cudaSuccess);

	//cub::KeyValuePair<int, float> debug;
	//status = cudaMemcpy(&debug, m_reductionResult.get(), sizeof(debug), cudaMemcpyDeviceToHost);
	//assert(status == cudaSuccess);
	//printf("%d %f\n", debug.key, debug.value);
	//printf("\n");
}

CCollisionDetector::CCollisionDetector(const SParticleSOA d_particles, const thrust::host_vector<SPlane>& worldBoundaries) :
	m_deviceParticles(d_particles),
	m_devicePlanes(worldBoundaries),
	m_particle2particleReduction(d_particles.count, d_particles.count),
	m_particle2planeReduction(worldBoundaries.size(), d_particles.count)
{
	m_collisionResult = thrust::device_malloc<SObjectsCollision>(1);
}

SObjectsCollision* CCollisionDetector::FindEarliestCollision()
{
	auto particles = unsigned(m_deviceParticles.count);
	auto walls = unsigned(m_devicePlanes.size());

	dim3 blockDim;
	dim3 gridDim;

	blockDim = dim3(32u, 32u);
	gridDim = dim3(divCeil(particles, blockDim.x), divCeil(particles, blockDim.y));
	predictParticleParticleCollisionsKernel << <gridDim, blockDim >> > (m_deviceParticles, m_particle2particleReduction.m_matrix.data().get());

	blockDim = dim3(1024u, 1u);
	gridDim = dim3(divCeil(particles, blockDim.x), divCeil(walls, blockDim.y));
	predictParticlePlaneCollisionsKernel << <gridDim, blockDim >> > (m_deviceParticles, m_devicePlanes.data().get(), walls, m_particle2planeReduction.m_matrix.data().get());

	m_particle2particleReduction.Reduce();
	m_particle2planeReduction.Reduce();

	copyResultKernel << <1, 1 >> > (m_particle2particleReduction.m_reductionResult.get(), m_particle2planeReduction.m_reductionResult.get(), particles, walls, m_collisionResult.get());

	//SObjectsCollision debug;
	//auto status = cudaMemcpy(&debug, m_collisionResult.get(), sizeof(debug), cudaMemcpyDeviceToHost);
	//assert(status == cudaSuccess);
	//printf("%f\n", debug.predictedTime);
	//printf("\n");

	return m_collisionResult.get();
}