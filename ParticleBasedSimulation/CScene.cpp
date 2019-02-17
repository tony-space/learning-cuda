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


	//for (size_t i = 0; i < kMolecules; ++i)
	//{
	//	pos[i] = glm::sphericalRand(0.5f) * glm::linearRand(0.5f, 1.0f);
	//	pos[i] *= 0.5f;

	//	if (glm::linearRand(0.0f, 1.0f) > 0.5f)
	//		pos[i].x += 0.25f;
	//	else
	//		pos[i].x -= 0.25f;

	//	vel[i].x += pos[i].z * 0.5f;
	//	vel[i].z -= pos[i].x * 0.5f;

	//	color[i] = pos[i] + glm::vec3(0.6f);
	//	rad[i] = kParticleRad;
	//	mass[i] = 1.0f;
	//}

	for (size_t i = 0; i < kMolecules; ++i)
	{
		if (glm::linearRand(0.0f, 20.0f) < 19.0f)
		{
			pos[i] = glm::vec3
			(
				glm::linearRand(-0.01f, 0.01f),
				glm::linearRand(-0.1f, 0.1f),
				glm::linearRand(-0.1f, 0.1f)
			);

			pos[i].x -= 0.075f;
		}
		else
		{
			pos[i] = glm::sphericalRand(0.01f);
			pos[i].x += 0.35f;
			vel[i].x = -5.0f;
		}

		color[i] = pos[i] + glm::vec3(0.6f);
	}

	std::vector<float> bufferData;
	bufferData.insert(bufferData.end(), (float*)pos.data(), (float*)pos.data() + pos.size() * 3);
	bufferData.insert(bufferData.end(), (float*)color.data(), (float*)color.data() + color.size() * 3);
	bufferData.insert(bufferData.end(), rad.begin(), rad.end());


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
	void* pVboData = nullptr;
	error = cudaGraphicsResourceGetMappedPointer(&pVboData, &stateSize, m_resource);
	assert(error == cudaSuccess);
	m_state.pos = (float3*)pVboData;
	m_state.color = m_state.pos + kMolecules;
	m_state.radius = (float*)(m_state.color + kMolecules);


	error = cudaMalloc(&m_state.vel, kMolecules * sizeof(float3));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.vel, vel.data(), kMolecules * sizeof(float3), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);

	//no need to copy forces
	error = cudaMalloc(&m_state.force, kMolecules * sizeof(float3));
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

	error = cudaFree(m_state.mass);
	assert(error == cudaSuccess);
	
	error = cudaFree(m_state.force);
	assert(error == cudaSuccess);
	
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

	m_cudaSim->UpdateState(1.0f / 1000.0f);
}

void CScene::Render(float windowHeight, float fov, glm::mat4 mvm)
{
	auto smartSwitcher = m_spriteShader.Activate();

	static const glm::vec4 lightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
	m_spriteShader.SetUniform("pointScale", windowHeight / tanf(fov / 2.0f *  float(M_PI) / 180.0f));
	m_spriteShader.SetUniform("lightDir", (mvm * lightDirection).xyz);

	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	auto posLoc = m_spriteShader.GetAttributeLocation("pos");
	auto colorLoc = m_spriteShader.GetAttributeLocation("color");
	auto radLoc = m_spriteShader.GetAttributeLocation("radius");

	glEnableVertexAttribArray(posLoc);
	glEnableVertexAttribArray(colorLoc);
	glEnableVertexAttribArray(radLoc);

	glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(colorLoc, 3, GL_FLOAT, GL_FALSE, 0, (void*)(kMolecules * sizeof(glm::vec3)));
	glVertexAttribPointer(radLoc, 1, GL_FLOAT, GL_FALSE, 0, (void*)(2 * kMolecules * sizeof(glm::vec3)));

	glDrawArrays(GL_POINTS, 0, kMolecules);

	glDisableVertexAttribArray(radLoc);
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
