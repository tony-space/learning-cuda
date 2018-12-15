#pragma once

#include "CShaderProgram.hpp"
#include "Simulation.hpp"
#include <memory>

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

	//std::unique_ptr<CSimulation> m_cudaSim;
	std::unique_ptr<ISimulation> m_cudaSim;
};
