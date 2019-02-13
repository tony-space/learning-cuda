#define _USE_MATH_DEFINES
#define GLM_FORCE_SWIZZLE

#include "CScene.hpp"
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <vector>
#include <cmath>


static const size_t kMolecules = 8192;
static const float kParticleRad = 0.004f;

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
		//p.pos = glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f) + kParticleRad, glm::vec3(0.5f, 0.5f, 0.5f) - kParticleRad);
		//p.vel = glm::sphericalRand(1.0f) * glm::linearRand(0.0f, 0.3f);

		p.pos = glm::sphericalRand(0.5f) * glm::linearRand(0.5f, 1.0f);
		p.pos *= 0.5f;

		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
			p.pos.x += 0.25f;
		else
			p.pos.x -= 0.25f;

		p.vel.x += p.pos.z * 0.5f;
		p.vel.z += -p.pos.x * 0.5f;
	}

	/*for (auto& c : colors)
		c = glm::linearRand(glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 1.0f, 1.0f));*/

	for (size_t i = 0; i < kMolecules; ++i)
		colors[i] = particles[i].pos + glm::vec3(0.6f);

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

	cudaError_t error;

	error = cudaGraphicsGLRegisterBuffer(&m_resource, m_moleculesVBO, cudaGraphicsRegisterFlagsNone);
	assert(error == cudaSuccess);

	error = cudaGraphicsMapResources(1, &m_resource);
	assert(error == cudaSuccess);

	void* d_stateVector;
	size_t stateSize;
	error = cudaGraphicsResourceGetMappedPointer(&d_stateVector, &stateSize, m_resource);
	assert(error == cudaSuccess);

	m_cudaSim = ISimulation::CreateInstance(d_stateVector, kMolecules, kParticleRad);
}

CScene::~CScene()
{
	m_cudaSim.reset();

	cudaError_t error;
	error = cudaGraphicsUnmapResources(1, &m_resource);
	assert(error == cudaSuccess);

	error = cudaGraphicsUnregisterResource(m_resource);
	assert(error == cudaSuccess);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &m_moleculesVBO);
	auto err = glGetError();
	assert(err == GL_NO_ERROR);
}

void CScene::UpdateState(float dt)
{
	//int counter = 0;
	//while (dt > 0 && counter++ < 32)
	//	dt -= m_cudaSim->UpdateState(dt);

	m_cudaSim->UpdateState(1.0f / 1000.0f);
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
