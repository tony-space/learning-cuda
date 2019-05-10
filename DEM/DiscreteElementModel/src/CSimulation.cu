#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include "CSimulation.hpp"

#include <cub/device/device_radix_sort.cuh>

constexpr unsigned kBlockDim = 128u;

__device__ float4 getHeatMapColor(float value)
{
	value = fminf(fmaxf(value, 0.0f), 1.0f);

	static const size_t stages = 7;
	static const float3 heatMap[stages] = { {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f} };
	value *= stages - 1;
	int idx1 = int(value);
	int idx2 = idx1 + 1;
	float fract1 = value - float(idx1);
	return make_float4(heatMap[idx1] + fract1 * (heatMap[idx2] - heatMap[idx1]), 1.0f);
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

__global__ void getCellIdKernel(const SParticleSOA particles, size_t* __restrict__ particleId, size_t* __restrict__ cellId)
{
	auto threadId = size_t(blockDim.x * blockIdx.x + threadIdx.x);
	if (threadId >= particles.count)
		return;

	auto pos = particles.pos[threadId];

	constexpr auto worldDim = 2.0f; //from -1.0 to 1.0
	pos += make_float4(1.0f);//make each coordinate to be from 0 to 2.0f
	
	auto cellDim = particles.radius * 2.0f;
	auto cellsPerDim = size_t(ceilf(worldDim / cellDim));

	//compute integer indices of the current cell
	auto idx = size_t(floorf(pos.x / cellDim));
	auto idy = size_t(floorf(pos.y / cellDim));
	auto idz = size_t(floorf(pos.z / cellDim));

	particleId[threadId] = threadId;
	cellId[threadId] = idz * cellsPerDim * cellsPerDim + idy * cellsPerDim + idx;
}

__global__ void reorderParticlesKernel(const size_t* __restrict__ particleId, SParticleSOA particles)
{
	auto threadId = size_t(blockDim.x * blockIdx.x + threadIdx.x);
	if (threadId >= particles.count)
		return;

	float4 buf;
	size_t idx = particleId[threadId];

	buf = particles.pos[idx];
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

	m_deviceParticleIdx.resize(d_particles.count);
	m_deviceCellIdx.resize(d_particles.count);
	m_deviceParticleIdxAlt.resize(d_particles.count);
	m_deviceCellIdxAlt.resize(d_particles.count);
}

float CSimulation::UpdateState(float dt)
{
	dim3 blockDim(kBlockDim);
	dim3 gridDim(unsigned((m_deviceParticles.count - 1) / kBlockDim + 1));

	auto d_particleIdx = m_deviceParticleIdx.data().get();
	auto d_cellIdx = m_deviceCellIdx.data().get();

	getCellIdKernel <<<gridDim, blockDim >>> (m_deviceParticles, d_particleIdx, d_cellIdx);

	SortAndReorder();

	return dt;
}

void CSimulation::SortAndReorder()
{
	cudaError status;
	dim3 blockDim(kBlockDim);
	dim3 gridDim(unsigned((m_deviceParticles.count - 1) / kBlockDim + 1));

	auto d_particleIdx = m_deviceParticleIdx.data().get();
	auto d_particleIdxAlt = m_deviceParticleIdxAlt.data().get();
	auto d_cellIdx = m_deviceCellIdx.data().get();
	auto d_cellIdxAlt = m_deviceCellIdxAlt.data().get();

	cub::DoubleBuffer<size_t> d_keys(d_cellIdx, d_cellIdxAlt);
	cub::DoubleBuffer<size_t> d_values(d_particleIdx, d_particleIdxAlt);

	void* d_tempStorage = nullptr;
	size_t tempStorageSize = 0;
	int elements = int(m_deviceParticles.count);

	status = cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageSize, d_keys, d_values, elements);
	assert(status == cudaSuccess);
	if (tempStorageSize > m_cubDataStorage.size())
		m_cubDataStorage.resize(tempStorageSize);

	d_tempStorage = m_cubDataStorage.data().get();
	status = cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageSize, d_keys, d_values, elements);
	assert(status == cudaSuccess);

	if (d_keys.Current() == d_cellIdxAlt)
	{
		typedef thrust::device_ptr<size_t> ptr;
		thrust::copy(ptr(d_cellIdxAlt), ptr(d_cellIdxAlt + m_deviceParticles.count), ptr(d_cellIdx));
	}

	reorderParticlesKernel <<<gridDim, blockDim >>> (d_values.Current(), m_deviceParticles);
}

std::unique_ptr<ISimulation> ISimulation::CreateInstance(SParticleSOA d_particles)
{
	return std::make_unique<CSimulation>(d_particles);
}