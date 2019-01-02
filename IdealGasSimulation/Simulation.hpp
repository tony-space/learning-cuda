#pragma once
#include <memory>

class __declspec(novtable) ISimulation
{
public:
	virtual ~ISimulation() {}
	virtual float UpdateState(float dt) = 0;

	static std::unique_ptr<ISimulation> CreateInstance(void* particles, size_t particlesCount, float particleRadius);
};