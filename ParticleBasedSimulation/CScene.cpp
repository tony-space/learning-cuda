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


	for (size_t i = 0; i < kMolecules; ++i)
	{
		/*pos[i] = glm::sphericalRand(0.1f) * glm::linearRand(0.25f, 1.0f);
		pos[i] *= 0.5f;*/

		pos[i].x = glm::linearRand(-0.15f, 0.15f);
		pos[i].yz = glm::circularRand(0.015f);

		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
		{
			glm::mat4 m = glm::identity<glm::mat4>();
			m = glm::rotate(m, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			pos[i] = m * glm::vec4(pos[i], 1.0f);

			pos[i].x += 0.4f;
			vel[i].x -= 1.0f;
		}
		else
		{
			glm::mat4 m = glm::identity<glm::mat4>();
			m = glm::rotate(m, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			pos[i] = m * glm::vec4(pos[i], 1.0f);

			pos[i].x -= 0.4f;
			vel[i].x += 1.0f;
		}
	}

	//for (size_t i = 0; i < kMolecules; ++i)
	//{
	//	if (glm::linearRand(0.0f, 20.0f) < 19.0f)
	//	{
	//		pos[i] = glm::vec3
	//		(
	//			glm::linearRand(-0.01f, 0.01f),
	//			glm::linearRand(-0.15f, 0.15f),
	//			glm::linearRand(-0.15f, 0.15f)
	//		);

	//		pos[i].x -= 0.35f;
	//	}
	//	else
	//	{
	//		//pos[i] = glm::sphericalRand(0.01f);
	//		pos[i].x = glm::linearRand(0.0f, 0.1f);
	//		pos[i].yz = glm::circularRand(0.01f);

	//		pos[i].x += 0.35f;
	//		vel[i].x = -2.5f;
	//	}
	//}

	std::vector<float> bufferData;
	bufferData.insert(bufferData.end(), (float*)pos.data(), (float*)pos.data() + pos.size() * 3);
	bufferData.insert(bufferData.end(), (float*)color.data(), (float*)color.data() + color.size() * 3);


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
	m_state.radius = kParticleRad;
	m_state.mass = 1.0f;
	m_state.maxDiameterFactor = 1.75f;


	error = cudaMalloc(&m_state.vel, kMolecules * sizeof(float3));
	assert(error == cudaSuccess);
	error = cudaMemcpy(m_state.vel, vel.data(), kMolecules * sizeof(float3), cudaMemcpyHostToDevice);
	assert(error == cudaSuccess);

	//no need to copy forces
	error = cudaMalloc(&m_state.force, kMolecules * sizeof(float3));
	assert(error == cudaSuccess);


	m_cudaSim = ISimulation::CreateInstance(m_state);
}

CScene::~CScene()
{
	m_cudaSim.reset();

	cudaError_t error;

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

	m_cudaSim->UpdateState(1.0f / 2000.0f);
}

void CScene::Render(float windowHeight, float fov, glm::mat4 mvm)
{
	auto smartSwitcher = m_spriteShader.Activate();

	static const glm::vec4 lightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
	m_spriteShader.SetUniform("pointScale", windowHeight / tanf(fov / 2.0f *  float(M_PI) / 180.0f));
	m_spriteShader.SetUniform("lightDir", (mvm * lightDirection).xyz);
	m_spriteShader.SetUniform("radius", kParticleRad);

	glBindBuffer(GL_ARRAY_BUFFER, m_moleculesVBO);
	auto posLoc = m_spriteShader.GetAttributeLocation("pos");
	auto colorLoc = m_spriteShader.GetAttributeLocation("color");

	glEnableVertexAttribArray(posLoc);
	glEnableVertexAttribArray(colorLoc);

	glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(colorLoc, 3, GL_FLOAT, GL_FALSE, 0, (void*)(kMolecules * sizeof(glm::vec3)));

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
