#pragma once
#include <memory>
#include <helper_math.h>

//Pattern 'Structure of arrays (SOA)'
//SOA doesn't waste bandwidth and provides higher memory coalescing
struct SParticleSOA
{
	size_t count;

	//vector types
	float3* __restrict__ pos;
	float3* __restrict__ vel;
	float3* __restrict__ color;
	float3* __restrict__ force;
	
	//scalar types
	float radius;
	float mass;
	float maxDiameterFactor;
};

class __declspec(novtable) ISimulation
{
public:
	virtual ~ISimulation() {}
	virtual float UpdateState(float dt) = 0;

	static std::unique_ptr<ISimulation> CreateInstance(SParticleSOA d_particles);
};