#pragma once
#include <GL/glew.h>
#include <vector_types.h>

class CScene
{
public:
	CScene(GLsizei width, GLsizei height, float aspectRatio);
	~CScene();

	void Render(float dt);
private:
	GLuint m_texture;
	GLsizei m_width;
	GLsizei m_height;
	float m_aspectRatio;
	float m_time;

	void UpdateTexture(float dt);
};
