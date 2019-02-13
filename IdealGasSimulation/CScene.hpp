#pragma once

#include "CShaderProgram.hpp"
#include "ISimulation.hpp"

#include <memory>
#include <cuda_gl_interop.h>

class CScene
{
public:
	CScene();
	~CScene();

	void UpdateState(float dt);
	void Render(float windowHeight, float fov, glm::mat4 mvm);
	float GetParticleRadius() const;
private:
	CShaderProgram m_spriteShader;
	GLuint m_moleculesVBO;

	std::unique_ptr<ISimulation> m_cudaSim;

	//state arrays:
	cudaGraphicsResource_t m_resource; // OpenGL VBO object with positions
	SParticleSOA m_state;
};
