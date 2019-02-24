#define _USE_MATH_DEFINES
#define GLM_FORCE_SWIZZLE

#include "CScene.hpp"
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <vector>
#include <cmath>


static const size_t kMolecules = 16384;
static const float kParticleRad = 0.001f;

//static const size_t kMolecules = 32;
//static const float kParticleRad = 0.03f;

CScene::CScene() : m_spriteShader("shaders\\vertex.glsl", "shaders\\fragment.glsl"), m_state()
{
	GLenum glError;
	cudaError_t error;

	std::vector<glm::vec4> pos(kMolecules);
	std::vector<glm::vec4> vel(kMolecules);
	std::vector<glm::vec4> color(kMolecules);

	for (size_t i = 0; i < kMolecules; ++i)
	{
		pos[i] = glm::vec4(glm::sphericalRand(0.4f) * glm::linearRand(0.25f, 1.0f), 1.0f);
		pos[i] *= 0.5f;

		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
			pos[i].x += 0.25f;
		else
			pos[i].x -= 0.25f;

		vel[i].xyz = glm::sphericalRand(0.05f);

		vel[i].x += pos[i].z * 0.5f;
		vel[i].z -= pos[i].x * 0.5f;

		color[i] = pos[i] + glm::vec4(0.6f, 0.6f, 0.6f, 0.0f);
	}

	std::vector<float> bufferData;
	bufferData.insert(bufferData.end(), (float*)pos.data(), (float*)pos.data() + pos.size() * 4);
	bufferData.insert(bufferData.end(), (float*)color.data(), (float*)color.data() + color.size() * 4);


	glGenBuffers(1, &m_moleculesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	glBufferData(GL_ARRAY_BUFFER, bufferData.size() * sizeof(bufferData[0]), bufferData.data(), GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glError = glGetError();
	assert(glError == GL_NO_ERROR);
	
	//filling state
	m_state.count = kMolecules;
	m_state.radius = kParticleRad;

	error = cudaGraphicsGLRegisterBuffer(&m_resource, m_moleculesVBO, cudaGraphicsRegisterFlagsNone);
	assert(error == cudaSuccess);

	error = cudaGraphicsMapResources(1, &m_resource);
	assert(error == cudaSuccess);

	size_t stateSize;
	void* pVboData = nullptr;
	error = cudaGraphicsResourceGetMappedPointer(&pVboData, &stateSize, m_resource);
	assert(error == cudaSuccess);
	m_state.pos = (float4*)pVboData;
	m_state.color = m_state.pos + kMolecules;

	error = cudaMalloc(&m_state.vel, kMolecules * sizeof(float4));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.vel, vel.data(), kMolecules * sizeof(float4), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);

	m_cudaSim = ISimulation::CreateInstance(m_state);
}

CScene::~CScene()
{
	m_cudaSim.reset();

	cudaError_t error;

	error = cudaFree(m_state.vel);
	assert(error == cudaSuccess);

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

	m_cudaSim->UpdateState(dt);
}

void CScene::Render(float windowHeight, float fov, glm::mat4 mvm)
{
	auto smartSwitcher = m_spriteShader.Activate();

	static const glm::vec4 lightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
	m_spriteShader.SetUniform("pointScale", windowHeight / tanf(fov / 2.0f *  float(M_PI) / 180.0f));
	m_spriteShader.SetUniform("radius", kParticleRad);
	m_spriteShader.SetUniform("lightDir", (mvm * lightDirection).xyz);

	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	auto posLoc = m_spriteShader.GetAttributeLocation("pos");
	auto colorLoc = m_spriteShader.GetAttributeLocation("color");

	glEnableVertexAttribArray(posLoc);
	glEnableVertexAttribArray(colorLoc);

	glVertexAttribPointer(posLoc, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(colorLoc, 4, GL_FLOAT, GL_FALSE, 0, (void*)(kMolecules * sizeof(glm::vec4)));

	glDrawArrays(GL_POINTS, 0, kMolecules);

	glDisableVertexAttribArray(colorLoc);
	glDisableVertexAttribArray(posLoc);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	auto err = glGetError();
	assert(err == GL_NO_ERROR);
}

float CScene::GetParticleRadius() const
{
	return 0.0f;
}
