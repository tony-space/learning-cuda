#include "CElectricField.hpp"
#include <cassert>

#include <glm/glm.hpp>

CElectricField::CElectricField(GLsizei width, GLsizei height) : m_width(width), m_height(height)
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
	//assert(!error);

	glDeleteTextures(1, &m_texture);
	error = glGetError();
	//assert(!error);
}

void CElectricField::AddParticle(const SParticle& p)
{
	m_hostParticles.push_back(p);
	m_deviceParticles = m_hostParticles;
}

void CElectricField::Render(float dt)
{
	UpdateState(0.005);

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

	//auto posA = glm::vec2(m_particles[0].position.x, m_particles[0].position.y);
	//auto posB = glm::vec2(m_particles[1].position.x, m_particles[1].position.y);

	//auto r = posB - posA;
	//auto distance = glm::length(r);
	//auto force = m_particles[0].charge * m_particles[1].charge * r / (distance * distance * distance);

	//auto accelA = -force / m_particles[0].mass;
	//auto accelB = force / m_particles[1].mass;

	//auto velA = glm::vec2(m_particles[0].velocity.x, m_particles[0].velocity.y);
	//auto velB = glm::vec2(m_particles[1].velocity.x, m_particles[1].velocity.y);

	//velA += accelA * dt;
	//velB += accelB * dt;

	//posA += velA * dt;
	//posB += velB * dt;

	//m_particles[0].position = { posA.x, posA.y };
	//m_particles[1].position = { posB.x, posB.y };

	//m_particles[0].velocity = { velA.x, velA.y };
	//m_particles[1].velocity = { velB.x, velB.y };
}
