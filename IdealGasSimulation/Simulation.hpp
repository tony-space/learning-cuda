#pragma once
#include <gl\glew.h>
#include <cuda_runtime.h>
#include <memory>

class __declspec(novtable) ISimulation
{
public:
	virtual ~ISimulation() {}
	virtual void UpdateState(float dt) = 0;

	static std::unique_ptr<ISimulation> CreateInstance(GLuint stateVBO, size_t particlesCount, float particleRadius);
};