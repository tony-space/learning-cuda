#pragma once
#include "../include/ISimulation.hpp"

struct SPlane
{
	float3 normal;
	float planeDistance;
	SPlane(float3 _normal, float _dist) : normal(normalize(_normal)), planeDistance(_dist) {	}
	inline __device__ __host__ float Distance(const float3& pos, float radius)
	{
		return dot(pos, normal) - planeDistance - radius;
	}
};

struct SObjectsCollision
{
	enum class CollisionType : int
	{
		None,
		ParticleToParticle,
		ParticleToPlane
	};

	size_t object1 = size_t(-1);
	size_t object2 = size_t(-1);
	//predicted time interval when collision will happen
	float predictedTime = INFINITY;

	CollisionType collisionType = CollisionType::None;

	__device__ inline void AnalyzeAndApply(const size_t obj1, const size_t obj2, float time, CollisionType type)
	{
		if (time < 0.0f) return;
		if (time > predictedTime) return;

		object1 = obj1;
		object2 = obj2;
		predictedTime = time;
		collisionType = type;
	}
};

template<typename T>
static inline __device__ __host__ T divCeil(T a, T b)
{
	return (a - 1) / b + 1;
}