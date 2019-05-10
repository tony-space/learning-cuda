#define GLM_FORCE_SWIZZLE

#include <cstdio>
#include <ISimulation.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

constexpr size_t kParticles = 16;
constexpr float kParticleRad = 0.004f;
constexpr size_t kArrayBytesSize = sizeof(float4) * kParticles;
int main(int argc, char** argv)
{
	cudaError status;
	std::vector<glm::vec4> pos(kParticles);
	std::vector<glm::vec4> vel(kParticles);

	for (size_t i = 0; i < kParticles; ++i)
	{
		pos[i] = glm::linearRand(glm::vec4(-1.0f + kParticleRad), glm::vec4(1.0f - kParticleRad));
		vel[i] = glm::linearRand(glm::vec4(-1.0f), glm::vec4(1.0f));
	}

	SParticleSOA simState;
	simState.count = kParticles;
	simState.mass = 1.0f;
	simState.maxDiameterFactor = 1.75f;
	simState.radius = kParticleRad;

	status = cudaMalloc(&simState.pos, kArrayBytesSize);
	assert(status == cudaSuccess);
	status = cudaMalloc(&simState.vel, kArrayBytesSize);
	assert(status == cudaSuccess);
	status = cudaMalloc(&simState.color, kArrayBytesSize);
	assert(status == cudaSuccess);
	status = cudaMalloc(&simState.force, kArrayBytesSize);
	assert(status == cudaSuccess);

	status = cudaMemcpy(simState.pos, pos.data(), kArrayBytesSize, cudaMemcpyHostToDevice);
	assert(status == cudaSuccess);
	status = cudaMemcpy(simState.vel, vel.data(), kArrayBytesSize, cudaMemcpyHostToDevice);
	assert(status == cudaSuccess);

	{
		auto simulation = ISimulation::CreateInstance(simState);
		simulation->UpdateState(0.1f);
	}

	status = cudaFree(&simState.pos);
	assert(status == cudaSuccess);
	status = cudaFree(&simState.vel);
	assert(status == cudaSuccess);
	status = cudaFree(&simState.color);
	assert(status == cudaSuccess);
	status = cudaFree(&simState.force);
	assert(status == cudaSuccess);
	return 0;
}