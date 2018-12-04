#include "CElectricField.hpp"
#include <cassert>

#include <glm/glm.hpp>

CElectricField::CElectricField(GLsizei width, GLsizei height, float aspectRatio) : m_width(width), m_height(height), m_aspectRatio(aspectRatio)
{
	GLenum error;

	glGenTextures(1, &m_texture);
	error = glGetError();
	assert(!error);

	GLint oldTex;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTex);
	error = glGetError();
	assert(!error);

	glBindTexture(GL_TEXTURE_2D, m_texture);
	error = glGetError();
	assert(!error);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	error = glGetError();
	assert(!error);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	error = glGetError();
	assert(!error);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_width, m_width, 0, GL_RGBA, GL_FLOAT, nullptr);
	error = glGetError();
	assert(!error);

	glBindTexture(GL_TEXTURE_2D, oldTex);
	error = glGetError();
	assert(!error);
}

CElectricField::~CElectricField()
{
	GLenum error;
	glBindTexture(GL_TEXTURE_2D, m_texture);
	error = glGetError();
	assert(!error);

	glDeleteTextures(1, &m_texture);
	error = glGetError();
	assert(!error);
}

void CElectricField::AddParticle(const SParticle& p)
{
	m_hostParticles.push_back(p);
	m_deviceParticles = m_hostParticles;
}

void CElectricField::Render(float dt)
{
	UpdateState(0.005f);
	UpdateTexture();

	GLenum error;
	GLint oldTex;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTex);
	error = glGetError();
	assert(!error);

	glBindTexture(GL_TEXTURE_2D, m_texture);
	error = glGetError();
	assert(!error);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(-1.0f, -1.0f);

	glTexCoord2f(1.0, 0.0);
	glVertex2f(1.0f, -1.0f);

	glTexCoord2f(1.0, 1.0);
	glVertex2f(1.0f, 1.0f);

	glTexCoord2f(0.0, 1.0);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, oldTex);
	error = glGetError();
	assert(!error);
}
