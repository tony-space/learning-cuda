#pragma once

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>

#include "SimulationTypes.hpp"

class CCollisionDetector
{
public:
	CCollisionDetector(const SParticleSOA d_particles, const thrust::host_vector<SPlane>& worldBoundaries);
	SObjectsCollision* FindEarliestCollision();
	const SPlane* GetPlanes() const { return m_devicePlanes.data().get(); }
private:
	const SParticleSOA m_deviceParticles;
	thrust::device_vector<SPlane> m_devicePlanes;
	thrust::device_vector<SObjectsCollision> m_collisions;
	thrust::device_ptr<SObjectsCollision> m_collisionResult;

	thrust::device_vector<float> m_mappedCollisionTimes;
	thrust::device_ptr<cub::KeyValuePair<int, float>> m_reductionResult;
	thrust::device_vector<uint8_t> m_cubTemporaryStorage;
};