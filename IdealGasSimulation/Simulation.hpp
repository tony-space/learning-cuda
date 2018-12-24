#pragma once
#include <gl\glew.h>
#include <memory>

class __declspec(novtable) ISimulation
{
public:
	virtual ~ISimulation() {}
	virtual float UpdateState(float dt) = 0;

	static std::unique_ptr<ISimulation> CreateInstance(GLuint stateVBO, size_t particlesCount, float particleRadius);
};