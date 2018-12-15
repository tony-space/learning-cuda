#pragma once

#include "CShaderProgram.hpp"

class CScene
{
public:
	CScene();
	~CScene();

	void Render(float windowHeight, float fov, glm::mat4 mvm);
private:
	CShaderProgram m_spriteShader;
	GLuint m_moleculesVBO;
};
