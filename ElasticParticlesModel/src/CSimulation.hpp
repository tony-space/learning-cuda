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

	CSimulation(SParticleSOA d_particles);
	virtual float UpdateState(float dt) override;

private:
	std::unique_ptr<CCollisionDetector> m_collisionDetector;
	SParticleSOA m_deviceParticles; //N particles
};