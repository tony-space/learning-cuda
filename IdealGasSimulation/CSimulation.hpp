#pragma once
#include <gl\glew.h>
#include <cuda_runtime.h>

class CSimulation
{
public:
	CSimulation(GLuint stateVBO, size_t particles);
	~CSimulation();
	void UpdateState(float dt);
private:
	const GLuint m_stateVBO;
	const size_t m_particles;

	cudaGraphicsResource_t m_resource;
	float3* m_devicePositions;
	float3* m_deviceVelocities;
};