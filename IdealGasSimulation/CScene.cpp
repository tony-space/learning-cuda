#define _USE_MATH_DEFINES
#define GLM_FORCE_SWIZZLE

#include "CScene.hpp"
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <vector>
#include <cmath>


static const size_t kMolecules = 128;
static const float kParticleRad = 0.025f;

struct SParticle
{
	glm::vec3 pos;
	glm::vec3 vel;
};

CScene::CScene() : m_spriteShader("shaders\\vertex.glsl", "shaders\\fragment.glsl")
{
	std::vector<SParticle> particles(kMolecules);
	std::vector<glm::vec3> colors(kMolecules);

	for (auto& p : particles)
	{
		p.pos = glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f) + kParticleRad, glm::vec3(0.5f, 0.5f, 0.5f) - kParticleRad);
		p.vel = glm::sphericalRand(1.0f) * glm::linearRand(0.0f, 0.4f);
	}

	for (auto& c : colors)
		c = glm::linearRand(glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 1.0f, 1.0f));

	std::vector<glm::vec3> bufferData;
	bufferData.reserve(kMolecules * 3);
	for (const auto& p : particles)
	{
		bufferData.push_back(p.pos);
		bufferData.push_back(p.vel);
	}

	bufferData.insert(bufferData.end(), colors.begin(), colors.end());


	glGenBuffers(1, &m_moleculesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	glBufferData(GL_ARRAY_BUFFER, bufferData.size() * sizeof(bufferData[0]), bufferData.data(), GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	auto err = glGetError();
	assert(err == GL_NO_ERROR);

	m_cudaSim = ISimulation::CreateInstance(m_moleculesVBO, kMolecules, kParticleRad);
}

CScene::~CScene()
{
	m_cudaSim.reset();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &m_moleculesVBO);
	auto err = glGetError();
	assert(err == GL_NO_ERROR);
}

void CScene::UpdateState(float dt)
{
	m_cudaSim->UpdateState(dt);
}

void CScene::Render(float windowHeight, float fov, glm::mat4 mvm)
{
	auto smartSwitcher = m_spriteShader.Activate();

	static const glm::vec4 lightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
	m_spriteShader.SetUniform("pointRadius", kParticleRad);
	m_spriteShader.SetUniform("pointScale", windowHeight / tanf(fov / 2.0f *  float(M_PI) / 180.0f));
	m_spriteShader.SetUniform("lightDir", (mvm * lightDirection).xyz);

	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_FLOAT, sizeof(SParticle), (void*)0);
	glColorPointer(3, GL_FLOAT, 0, (void*)(kMolecules * 2 * sizeof(glm::vec3)));
	glDrawArrays(GL_POINTS, 0, kMolecules);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	auto err = glGetError();
	assert(err == GL_NO_ERROR);
}

float CScene::GetParticleRadius() const
{
	return 0.0f;
}
