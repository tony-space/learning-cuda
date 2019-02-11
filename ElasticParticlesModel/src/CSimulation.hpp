#pragma once
#include <memory>

#include<thrust/device_vector.h>

#include "../include/ISimulation.hpp"
#include "SimulationTypes.hpp"
#include "CCollisionDetector.hpp"

class CSimulation : public ISimulation
{
public:
	static constexpr float kSimPrecision = 1e-16f;

	CSimulation(void* d_particles, size_t particlesCount, float particleRadius);
	virtual float UpdateState(float dt) override;

private:
	std::unique_ptr<CCollisionDetector> m_collisionDetector;

	const size_t m_particlesCount; //N value
	const float m_particleRadius;
	
	SParticle* m_deviceParticles; //N particles

	thrust::device_vector<SPlane> m_devicePlanes;

	thrust::device_vector<float3> m_deviceForcesMatrix; //NxN elements
	thrust::device_vector<float3> m_deviceForces; //N elements
	thrust::device_vector<size_t> m_deviceReductionSegments; // used by CUB library
	thrust::device_vector<uint8_t> m_segmentedReductionStorage; //an intermediate storage, used by CUB library for reduction puproses

	thrust::device_vector<bool> m_deviceSpringsMatrix; //NxN elements
};