#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "CSimulation.hpp"
#include <cub/device/device_segmented_reduce.cuh>

__device__ float3 getHeatMapColor(float value)
{
	value = fminf(fmaxf(value, 0.0f), 1.0f);

	static const size_t stages = 7;
	static const float3 heatMap[stages] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f} };
	value *= stages - 1;
	int idx1 = int(value);
	int idx2 = idx1 + 1;
	float fract1 = value - float(idx1);
	return heatMap[idx1] + fract1 * (heatMap[idx2] - heatMap[idx1]);
}


__global__ void rebuildSprings(const SParticleSOA particles, bool* __restrict__ springsMat)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particles.count || j >= particles.count)
		return;

	bool result = false;

	auto matIndex = i * particles.count + j;

	if (i != j)
	{
		auto connected = springsMat[matIndex];

		auto r = particles.pos[i] - particles.pos[j];
		auto magnitude = length(r);
		auto diameter = particles.radius * 2.0f;
		//if (!connected)
		//{
		//	result = magnitude < diameter;
		//}
		//else
		if (connected)
		{
			result = magnitude <= diameter * particles.maxDiameterFactor;
		}

		//result = magnitude < diameter;
	}

	springsMat[matIndex] = result;
}

__global__ void computeForcesMatrix(const SParticleSOA particles, const bool* __restrict__ springsMat, float3* __restrict__ forcesMat)
{
	auto i = blockIdx.y * blockDim.y + threadIdx.y;
	auto j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= particles.count || j >= particles.count)
		return;

	auto matIndex = i * particles.count + j;

	float3 result = make_float3(0.0f);

	if (i != j)
	{
		bool connected = springsMat[matIndex];
		auto pos1 = particles.pos[i];
		auto pos2 = particles.pos[j];

		auto r = pos1 - pos2;
		auto magnitude = length(r);
		auto diameter = particles.radius * 2.0f;

		//if (!connected && magnitude > diameter)
		//{
		//	constexpr float k = 8e-7f;
		//	result = -k * particles.mass * particles.mass * r / (magnitude * magnitude * magnitude);
		//}
		//else
		if (connected || magnitude < diameter)
		{
			constexpr float stiffness = 7000.0f;
			constexpr float damp = 25.0f;

			auto delta = magnitude - diameter;

			auto vel1 = particles.vel[i];
			auto vel2 = particles.vel[j];

			auto v = dot(vel1 - vel2, r / magnitude);

			result = (-r / magnitude) * (delta * stiffness + v * damp);
		}
	}

	forcesMat[matIndex] = result;
}

__global__ void moveParticlesKernel(SParticleSOA particles, const SPlane* __restrict__ planes, const size_t planesCount, const float dt)
{
	auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadId >= particles.count)
		return;

	auto pos = particles.pos[threadId];
	auto vel = particles.vel[threadId];
	auto force = particles.force[threadId];

	pos += vel * dt;
	vel += force * dt / particles.mass;

	for (size_t i = 0; i < planesCount; ++i)
	{
		SPlane p = planes[i];
		if (p.Distance(pos, particles.radius) >= 0.0f)
			continue;

		if (dot(p.normal, vel) >= 0.0f)
			continue;

		vel = reflect(vel, p.normal);
		break;
	}

	particles.pos[threadId] = pos;
	particles.vel[threadId] = vel;
	particles.color[threadId] = getHeatMapColor(logf(length(force) + 1.0f) / 8.0f + 0.15f);
}

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

CSimulation::CSimulation(SParticleSOA d_particles) : m_deviceParticles(d_particles)
{
	thrust::host_vector<SPlane> hostPlanes;
	hostPlanes.push_back(SPlane(make_float3(1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(-1.0, 0.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, -1.0, 0.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, 1.0), -0.5));
	hostPlanes.push_back(SPlane(make_float3(0.0, 0.0, -1.0), -0.5));
	m_devicePlanes = hostPlanes;

	auto matSize = m_deviceParticles.count * m_deviceParticles.count;

	m_deviceForcesMatrix.resize(matSize, make_float3(0.0f));
	m_deviceSpringsMatrix.resize(matSize, true);

	thrust::host_vector<size_t> hostSegments(m_deviceParticles.count + 1);
	for (size_t i = 0; i < m_deviceParticles.count + 1; ++i)
		hostSegments[i] = i * m_deviceParticles.count;

	m_deviceReductionSegments = hostSegments;
}

float CSimulation::UpdateState(float dt)
{
	cudaError_t cudaStatus;
	auto blocks = unsigned((m_deviceParticles.count - 1) / 32 + 1);

	dim3 blockDim(32, 32);
	dim3 gridDim(blocks, blocks);

	rebuildSprings <<<gridDim, blockDim >>> (m_deviceParticles, m_deviceSpringsMatrix.data().get());

	computeForcesMatrix <<<gridDim, blockDim >>> (m_deviceParticles, m_deviceSpringsMatrix.data().get(), m_deviceForcesMatrix.data().get());

	size_t storageBytes = 0;
	cudaStatus = cub::DeviceSegmentedReduce::Sum(
		nullptr, storageBytes,
		m_deviceForcesMatrix.data().get(), m_deviceParticles.force,
		int(m_deviceParticles.count), m_deviceReductionSegments.data().get(), m_deviceReductionSegments.data().get() + 1);
	assert(cudaStatus == cudaSuccess);

	if (m_segmentedReductionStorage.size() != storageBytes)
		m_segmentedReductionStorage.resize(storageBytes);

	cudaStatus = cub::DeviceSegmentedReduce::Sum(
		m_segmentedReductionStorage.data().get(), storageBytes,
		m_deviceForcesMatrix.data().get(), m_deviceParticles.force,
		int(m_deviceParticles.count), m_deviceReductionSegments.data().get(), m_deviceReductionSegments.data().get() + 1);
	assert(cudaStatus == cudaSuccess);

	blockDim = dim3(64);
	gridDim = dim3((unsigned(m_deviceParticles.count) - 1) / blockDim.x + 1);

	moveParticlesKernel <<<gridDim, blockDim >>> (m_deviceParticles, m_devicePlanes.data().get(), m_devicePlanes.size(), dt);

	return dt;
}

std::unique_ptr<ISimulation> ISimulation::CreateInstance(SParticleSOA d_particles)
{
	return std::make_unique<CSimulation>(d_particles);
}