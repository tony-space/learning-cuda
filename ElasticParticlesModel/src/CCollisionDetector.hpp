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
	//cudaStream_t stream

	const SParticleSOA m_deviceParticles;
	thrust::device_vector<SPlane> m_devicePlanes;

	struct ArgMinReduction
	{
		thrust::device_vector<float> m_matrix;
		thrust::device_vector<uint8_t> m_cubTemporaryStorage;
		thrust::device_ptr<cub::KeyValuePair<int, float>> m_reductionResult;

		ArgMinReduction(size_t rows, size_t columns);

		void Reduce();
	};

	ArgMinReduction m_particle2particleReduction;
	ArgMinReduction m_particle2planeReduction;

	thrust::device_ptr<SObjectsCollision> m_collisionResult;
};