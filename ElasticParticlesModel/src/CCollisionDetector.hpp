#pragma once

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "SimulationTypes.hpp"

class CCollisionDetector
{
public:
	CCollisionDetector(const SParticle* d_particles, size_t particlesCount, float particleRadius, const thrust::host_vector<SPlane>& worldBoundaries);
	SObjectsCollision* FindEarliestCollision();
	const SPlane* GetPlanes() const { return m_devicePlanes.data().get(); }
private:
	const SParticle* m_deviceParticles;
	const size_t m_particlesCount;
	const float m_particleRadius;

	thrust::device_vector<SPlane> m_devicePlanes;

	thrust::device_vector<SObjectsCollision> m_collisions;
	thrust::device_vector<SObjectsCollision> m_intermediate;
};