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

CScene::CScene() : m_spriteShader("shaders\\vertex.glsl", "shaders\\fragment.glsl"), m_state()
{
	GLenum glError;
	cudaError_t error;

	std::vector<glm::vec3> pos(kMolecules);
	std::vector<glm::vec3> vel(kMolecules);
	std::vector<glm::vec3> color(kMolecules);
	std::vector<float> rad(kMolecules, kParticleRad);
	std::vector<float> mass(kMolecules, 1.0f);


	for (size_t i = 0; i < kMolecules; ++i)
	{
		pos[i] = glm::sphericalRand(0.5f) * glm::linearRand(0.5f, 1.0f);
		pos[i] *= 0.5f;

		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
			pos[i].x += 0.25f;
		else
			pos[i].x -= 0.25f;

		vel[i].x += pos[i].z * 0.5f;
		vel[i].z -= pos[i].x * 0.5f;

		color[i] = pos[i] + glm::vec3(0.6f);
	}

	std::vector<glm::vec3> bufferData;
	bufferData.reserve(kMolecules * 2);
	bufferData.insert(bufferData.end(), pos.begin(), pos.end());
	bufferData.insert(bufferData.end(), color.begin(), color.end());


	glGenBuffers(1, &m_moleculesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	glBufferData(GL_ARRAY_BUFFER, bufferData.size() * sizeof(bufferData[0]), bufferData.data(), GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glError = glGetError();
	assert(glError == GL_NO_ERROR);

	
	//filling state
	m_state.count = kMolecules;

	error = cudaGraphicsGLRegisterBuffer(&m_resource, m_moleculesVBO, cudaGraphicsRegisterFlagsNone);
	assert(error == cudaSuccess);

	error = cudaGraphicsMapResources(1, &m_resource);
	assert(error == cudaSuccess);

	size_t stateSize;
	error = cudaGraphicsResourceGetMappedPointer((void**)&m_state.pos, &stateSize, m_resource);
	assert(error == cudaSuccess);

	error = cudaMalloc(&m_state.vel, kMolecules * sizeof(float3));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.vel, vel.data(), kMolecules * sizeof(float3), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);

	//no need to copy forces
	error = cudaMalloc(&m_state.force, kMolecules * sizeof(float3));
	assert(error == cudaSuccess);

	error = cudaMalloc(&m_state.radius, kMolecules * sizeof(float));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.radius, rad.data(), kMolecules * sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);

	error = cudaMalloc(&m_state.mass, kMolecules * sizeof(float));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.mass, mass.data(), kMolecules * sizeof(float), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);


	m_cudaSim = ISimulation::CreateInstance(m_state);
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
	glVertexPointer(3, GL_FLOAT, 0, (void*)(0));
	glColorPointer(3, GL_FLOAT, 0, (void*)(kMolecules * sizeof(glm::vec3)));
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
