#pragma once
#include <memory>

#include<thrust/device_vector.h>

#include "../include/ISimulation.hpp"
#include "SimulationTypes.hpp"

class CSimulation : public ISimulation
{
public:
	static constexpr float kSimPrecision = 1e-16f;

	CSimulation(SParticleSOA d_particles);
	virtual float UpdateState(float dt) override;

private:
	SParticleSOA m_deviceParticles; //N particles
	thrust::device_vector<SPlane> m_devicePlanes;
	
	thrust::device_vector<uint8_t> m_cubDataStorage;//used by CUB library for the internal purposes

	//sorting part. CUB library switches between these buffers during radix sort
	thrust::device_vector<size_t> m_deviceParticleIdx;
	thrust::device_vector<size_t> m_deviceCellIdx;
	thrust::device_vector<size_t> m_deviceParticleIdxAlt;
	thrust::device_vector<size_t> m_deviceCellIdxAlt;

	void SortAndReorder();
};