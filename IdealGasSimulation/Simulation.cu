#include <GL/glew.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <cassert>
#include <helper_math.h>
#include <thrust/device_vector.h>

#include "Simulation.hpp"

static constexpr float kSimPrecision = 1e-16f;

static inline __device__ __host__ float sqr(float x)
{
	return x * x;
}

struct SParticle
{
	float3 pos;
	float3 vel;
};

struct SPlane
{
	float3 normal;
	float planeDistance;
	SPlane(float3 _normal, float _dist) : normal(normalize(_normal)), planeDistance(_dist) {	}
	inline __device__ __host__ float Distance(const SParticle& p, float radius)
	{
		return dot(p.pos, normal) - planeDistance - radius;
	}
};

struct SObjectsCollision
{
	enum class CollisionType
	{
		None,
		ParticleToParticle,
		ParticleToPlane
	};

	size_t object1 = size_t(-1);
	size_t object2 = size_t(-1);
	//predicted time interval when collision will happen
	float predictedTime = INFINITY;

	CollisionType collisionType = CollisionType::None;

	struct Comparator
	{
		__device__ inline SObjectsCollision operator()(const SObjectsCollision& x, const SObjectsCollision& y)
		{
			return x.predictedTime < y.predictedTime ? x : y;
		}
	};

	__device__ inline void AnalyzeAndApply(const size_t obj1, const size_t obj2, float time, CollisionType type)
	{
		if (time < 0.0f) return;
		if (time > predictedTime) return;

		object1 = obj1;
		object2 = obj2;
		predictedTime = time;
		collisionType = type;
	}
};

__global__ void predictParticleParticleCollisionsKernel(const SParticle* __restrict__ particles, const size_t particlesCount, const float particlesRadius, SObjectsCollision* __restrict__ out)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];

	SObjectsCollision earliestCollision;

	for (size_t i = threadId + 1; i < particlesCount; ++i)
	{
		//if (i == threadId) continue;
		SParticle other = particles[i];

		//Let's solve a quadratic equation to predict the exact collision time.
		//The quadric equation can be get from the following vector equation:
		//(R1 + V1 * dt) - (R2 + V2 * dt) = rad1 + rad2  : the distance between new positions equals the sum of two radii
		//where R1 and R2 are radius vectors of the current particles position
		//      V1 and V2 are velocity vectors
		//      rad1 and rad2 are particles' radii
		//      dt is the unknown variable
		//Vector dot product satisfies a distributive law.

		float3 deltaR = self.pos - other.pos;
		float3 deltaV = self.vel - other.vel;

		//Quadratic equation coefficients
		float a = dot(deltaV, deltaV);
		float b = 2.0f * dot(deltaR, deltaV);
		float c = dot(deltaR, deltaR) - sqr(particlesRadius + particlesRadius);
		float discriminant = sqr(b) - 4.0f * a * c;

		//if particles don't move relatively each other (deltaV = 0)
		if (fabsf(a) <= kSimPrecision)
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
	const SParticle* __restrict__ particles,
	const size_t particlesCount,
	const float particlesRadius,
	const SPlane* __restrict__ planes,
	const size_t planesCount,
	SObjectsCollision* __restrict__ inOut)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];


	SObjectsCollision earliestCollision = inOut[threadId];

	for (size_t i = 0; i < planesCount; ++i)
	{
		SPlane plane = planes[i];

		auto velProjection = dot(plane.normal, self.vel);

		if (velProjection >= 0.0f)
			continue;

		auto time = max(-plane.Distance(self, particlesRadius) / velProjection, 0.0f);
		earliestCollision.AnalyzeAndApply(threadId, i, time, SObjectsCollision::CollisionType::ParticleToPlane);
	}

	inOut[threadId] = earliestCollision;
}

__global__ void moveParticlesKernel(SParticle* __restrict__ particles, const size_t particlesCount, const float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];
	self.pos += self.vel * dt;
	self.vel.y -= 1.0f * dt;

	particles[threadId] = self;
}

__global__ void resolveParticle2ParticleCollision(SParticle* p1, SParticle* p2)
{
	if (p1 == nullptr)
		return;
	if (p2 == nullptr)
		return;

	SParticle a = *p1;
	SParticle b = *p2;

	auto centerOfMassVel = (a.vel + b.vel) / 2.0f;
	auto v1 = a.vel - centerOfMassVel;
	auto v2 = b.vel - centerOfMassVel;

	auto planeNormal = normalize(b.pos - a.pos);

	v1 = reflect(v1, planeNormal) * 0.98f;
	v2 = reflect(v2, planeNormal) * 0.98f;

	a.vel = v1 + centerOfMassVel;
	b.vel = v2 + centerOfMassVel;

	*p1 = a;
	*p2 = b;
}

__global__ void resolveParticle2PlaneCollision(SParticle* particle, SPlane* plane)
{
	if (particle == nullptr)
		return;
	if (plane == nullptr)
		return;

	particle->vel = reflect(particle->vel, plane->normal) * 0.98f;
}

class CSimulation : public ISimulation
{
private:
	const GLuint m_stateVBO;
	const size_t m_particlesCount;
	const float m_particleRadius;

	cudaGraphicsResource_t m_resource;
	SParticle* m_deviceParticles;
	thrust::device_vector<SPlane> m_devicePlanes;

	thrust::device_vector<SObjectsCollision> m_collisions;

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

		thrust::host_vector<SPlane> hostPlanes;
		hostPlanes.push_back(SPlane(make_float3(1.0, 0.0, 0.0), -0.5));
		hostPlanes.push_back(SPlane(make_float3(-1.0, 0.0, 0.0), -0.5));
		hostPlanes.push_back(SPlane(make_float3(0.0, 1.0, 0.0), -0.5));
		hostPlanes.push_back(SPlane(make_float3(0.0, -1.0, 0.0), -0.5));
		hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, 1.0), -0.5));
		hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, -1.0), -0.5));
		m_devicePlanes = hostPlanes;

		m_collisions.resize(m_particlesCount);
	}

	virtual ~CSimulation() override
	{
		cudaError_t error;
		error = cudaGraphicsUnmapResources(1, &m_resource);
		assert(error == cudaSuccess);

		error = cudaGraphicsUnregisterResource(m_resource);
		assert(error == cudaSuccess);
	}

	virtual float UpdateState(float dt) override
	{
#pragma warning(push)
#pragma warning(disable : 4244)
		dim3 blockDim(32 * 32);
		dim3 gridDim((unsigned(m_particlesCount) - 1) / blockDim.x + 1);

		predictParticleParticleCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, m_particleRadius, thrust::raw_pointer_cast(m_collisions.data()));
		predictParticlePlaneCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, m_particleRadius, thrust::raw_pointer_cast(m_devicePlanes.data()), m_devicePlanes.size(), thrust::raw_pointer_cast(m_collisions.data()));

		auto earilestCollision = thrust::reduce(m_collisions.begin(), m_collisions.end(), SObjectsCollision(), SObjectsCollision::Comparator());

		bool detected = false;
		if (earilestCollision.predictedTime < dt)
		{
			dt = earilestCollision.predictedTime;
			detected = true;
		}

		moveParticlesKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, dt);

		if (detected)
			switch (earilestCollision.collisionType)
			{
			case SObjectsCollision::CollisionType::ParticleToParticle:
				resolveParticle2ParticleCollision << <1, 1 >> > (m_deviceParticles + earilestCollision.object1, m_deviceParticles + earilestCollision.object2);
				break;
			case SObjectsCollision::CollisionType::ParticleToPlane:
				resolveParticle2PlaneCollision << <1, 1 >> > (m_deviceParticles + earilestCollision.object1, thrust::raw_pointer_cast(m_devicePlanes.data()) + earilestCollision.object2);
				break;
			}

		return dt;
#pragma warning(pop)
	}
};

std::unique_ptr<ISimulation> ISimulation::CreateInstance(GLuint stateVBO, size_t particlesCount, float particleRadius)
{
	return std::make_unique<CSimulation>(stateVBO, particlesCount, particleRadius);
}