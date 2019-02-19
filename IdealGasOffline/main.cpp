#include <cstdio>
#include <ISimulation.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

static const size_t kMolecules = 8;
static const float kParticleRad = 0.025f;

int main(int argc, char** argv)
{
	std::vector<glm::vec3> pos(kMolecules);
	std::vector<glm::vec3> vel(kMolecules);
	std::vector<float> rad(kMolecules, kParticleRad);


	for (size_t i = 0; i < kMolecules; ++i)
	{
		pos[i] = glm::sphericalRand(0.5f) * glm::linearRand(0.5f, 1.0f);
		pos[i] *= 0.5f;

		if (glm::linearRand(0.0f, 1.0f) > 0.5f)
			pos[i].x += 0.25f;
		else
			pos[i].x -= 0.25f;

		vel[i] = glm::sphericalRand(0.05f);

		vel[i].x += pos[i].z * 0.5f;
		vel[i].z -= pos[i].x * 0.5f;
	}

	std::vector<float> bufferData;
	bufferData.insert(bufferData.end(), (float*)pos.data(), (float*)pos.data() + kMolecules * 3);
	bufferData.insert(bufferData.end(), (float*)vel.data(), (float*)vel.data() + kMolecules * 3);
	bufferData.insert(bufferData.end(), rad.begin(), rad.end());

	void* d_state;
	auto status = cudaMalloc(&d_state, bufferData.size() * sizeof(float));
	assert(status == cudaSuccess);

	status = cudaMemcpy(d_state, bufferData.data(), bufferData.size() * sizeof(float), cudaMemcpyHostToDevice);
	assert(status == cudaSuccess);

	SParticleSOA state = SParticleSOA();
	state.count = kMolecules;
	state.pos = (float3*)d_state;
	state.vel = state.pos + kMolecules;
	state.radius = (float*)(state.vel + kMolecules);

	auto sim = ISimulation::CreateInstance(state);
	for (size_t i = 0; i < 100; ++i)
		sim->UpdateState(1.0f);


	cudaFree(d_state);
}