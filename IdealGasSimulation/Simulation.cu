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

struct SParticleParticleCollision
{
	//first particle index
	size_t particle1 = size_t(-1);
	//second particle index
	size_t particle2 = size_t(-1);
	//predicted time interval when collision will happen
	float predictedTime = INFINITY;

	struct Comparator
	{
		__device__ inline SParticleParticleCollision operator()(const SParticleParticleCollision& x, const SParticleParticleCollision& y)
		{
			return x.predictedTime < y.predictedTime ? x : y;
		}
	};

	__device__ inline void AnalyzeAndApply(size_t a, size_t b, float time)
	{
		if (time < 0.0f) return;
		if (time > predictedTime) return;

		particle1 = a;
		particle2 = b;
		predictedTime = time;
	}
};

struct SParticlePlaneCollision
{
	//first particle index
	size_t particle = size_t(-1);
	//second particle index
	size_t plane = size_t(-1);
	//predicted time interval when collision will happen
	float predictedTime = INFINITY;

	struct Comparator
	{
		__device__ inline SParticlePlaneCollision operator()(const SParticlePlaneCollision& x, const SParticlePlaneCollision& y)
		{
			return x.predictedTime < y.predictedTime ? x : y;
		}
	};

	__device__ inline void AnalyzeAndApply(size_t _particle, size_t _plane, float time)
	{
		if (time < 0.0f) return;
		if (time > predictedTime) return;

		particle = _particle;
		plane = _plane;
		predictedTime = time;
	}
};

__global__ void predictParticleParticleCollisionsKernel(const SParticle particles[], const size_t particlesCount, const float particlesRadius, SParticleParticleCollision out[])
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];


	SParticleParticleCollision earliestCollision;

	for (size_t i = 0; i < particlesCount; ++i)
	{
		if (i == threadId) continue;
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
			continue;

		//if particles ways never intersect
		if (discriminant < 0.0f)
			continue;

		float sqrtD = sqrtf(discriminant);
		//Here is a tricky part.
		//You might think, why we even need to compute dt2 if it definitely is greater than dt1?
		//The answer is these two values can be negative, which means two contacts has already been somewhere in the past.
		float dt1 = (-b - sqrtD) / (2.0f * a);
		float dt2 = (-b + sqrtD) / (2.0f * a);

		earliestCollision.AnalyzeAndApply(threadId, i, dt1);
		earliestCollision.AnalyzeAndApply(threadId, i, dt2);
	}

	out[threadId] = earliestCollision;
}

__global__ void predictParticlePlaneCollisionsKernel(
	const SParticle particles[],
	const size_t particlesCount,
	const float particlesRadius,
	const SPlane planes[],
	const size_t planesCount,
	SParticlePlaneCollision out[])
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];


	SParticlePlaneCollision earliestCollision;

	for (size_t i = 0; i < planesCount; ++i)
	{
		SPlane plane = planes[i];

		auto velProjection = dot(plane.normal, self.vel);

		if (velProjection >= 0.0f)
			continue;

		auto time = -plane.Distance(self, particlesRadius) / velProjection;
		earliestCollision.AnalyzeAndApply(threadId, i, time);
	}

	out[threadId] = earliestCollision;
}

__global__ void moveParticlesKernel(SParticle particles[], const size_t particlesCount, const float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particlesCount)
		return;

	SParticle self = particles[threadId];
	self.pos += self.vel * dt;

	particles[threadId] = self;
}

__global__ void resolveParticle2ParticleCollision(SParticle particles[], const size_t particlesCount, const size_t p1, const size_t p2)
{
	if (p1 >= particlesCount)
		return;
	if (p2 >= particlesCount)
		return;

	SParticle a = particles[p1];
	SParticle b = particles[p2];

	auto centerOfMassVel = (a.vel + b.vel) / 2.0f;
	auto v1 = a.vel - centerOfMassVel;
	auto v2 = b.vel - centerOfMassVel;

	auto planeNormal = normalize(b.pos - a.pos);

	v1 = reflect(v1, planeNormal);
	v2 = reflect(v2, planeNormal);

	a.vel = v1 + centerOfMassVel;
	b.vel = v2 + centerOfMassVel;

	particles[p1] = a;
	particles[p2] = b;
}

__global__ void resolveParticle2PlaneCollision(SParticle particles[], const size_t particlesCount, const SPlane planes[], const size_t planesCount, const size_t particle, const size_t plane)
{
	if (particle >= particlesCount)
		return;
	if (plane >= planesCount)
		return;

	particles[particle].vel = reflect(particles[particle].vel, planes[plane].normal);
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

	thrust::device_vector<SParticleParticleCollision> m_part2PartCollisions;
	thrust::device_vector<SParticlePlaneCollision> m_part2PlaneCollisions;

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

		m_part2PartCollisions.resize(m_particlesCount);
		m_part2PlaneCollisions.resize(m_particlesCount);
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
#pragma warning(push)
#pragma warning(disable : 4244)
		dim3 blockDim(32 * 32);
		dim3 gridDim((unsigned(m_particlesCount) - 1) / blockDim.x + 1);

		predictParticleParticleCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, m_particleRadius, thrust::raw_pointer_cast(m_part2PartCollisions.data()));
		predictParticlePlaneCollisionsKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, m_particleRadius, thrust::raw_pointer_cast(m_devicePlanes.data()), m_devicePlanes.size(), thrust::raw_pointer_cast(m_part2PlaneCollisions.data()));
		
		auto earilestPart2PartCollision = thrust::reduce(m_part2PartCollisions.begin(), m_part2PartCollisions.end(), SParticleParticleCollision(), SParticleParticleCollision::Comparator());
		auto earilestPart2PlaneCollision = thrust::reduce(m_part2PlaneCollisions.begin(), m_part2PlaneCollisions.end(), SParticlePlaneCollision(), SParticlePlaneCollision::Comparator());

		enum PredictionResult
		{
			NoCollisions,
			Particle2Particle,
			Particle2Plane
		} predResult;

		if (dt < earilestPart2PartCollision.predictedTime && dt < earilestPart2PlaneCollision.predictedTime)
			predResult = NoCollisions;
		else if (earilestPart2PartCollision.predictedTime < dt && earilestPart2PartCollision.predictedTime < earilestPart2PlaneCollision.predictedTime)
		{
			predResult = Particle2Particle;
			dt = earilestPart2PartCollision.predictedTime;
		}
		else if (earilestPart2PlaneCollision.predictedTime < dt && earilestPart2PlaneCollision.predictedTime < earilestPart2PartCollision.predictedTime)
		{
			predResult = Particle2Plane;
			dt = earilestPart2PlaneCollision.predictedTime;
		}
		else
			throw std::exception();

		moveParticlesKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_particlesCount, dt);
		switch (predResult)
		{
		case Particle2Particle:
			resolveParticle2ParticleCollision <<<1, 1 >>> (m_deviceParticles, m_particlesCount, earilestPart2PartCollision.particle1, earilestPart2PartCollision.particle2);
			break;
		case Particle2Plane:
			resolveParticle2PlaneCollision <<<1, 1 >>> (m_deviceParticles, m_particlesCount, thrust::raw_pointer_cast(m_devicePlanes.data()), m_devicePlanes.size(), earilestPart2PlaneCollision.particle, earilestPart2PlaneCollision.plane);
			break;
		}

		

#pragma warning(pop)
	}
};

std::unique_ptr<ISimulation> ISimulation::CreateInstance(GLuint stateVBO, size_t particlesCount, float particleRadius)
{
	return std::make_unique<CSimulation>(stateVBO, particlesCount, particleRadius);
}