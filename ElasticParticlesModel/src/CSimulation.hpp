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
	const size_t m_particlesCount;
	const float m_particleRadius;
	SParticle* m_deviceParticles;
	std::unique_ptr<CCollisionDetector> m_collisionDetector;
};