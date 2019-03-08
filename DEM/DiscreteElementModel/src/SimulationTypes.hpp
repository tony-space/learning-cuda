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

template<typename T>
static inline __device__ __host__ T divCeil(T a, T b)
{
	return (a - 1) / b + 1;
}