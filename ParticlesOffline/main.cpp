#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Simulation.hpp"

static const size_t kMolecules = 256;
static const float kParticleRad = 0.04f;

struct SParticle
{
	glm::vec3 pos;
	glm::vec3 vel;
};

int main(int argc, char** argv)
{
	thrust::host_vector<SParticle> particles(kMolecules);

	for (auto& p : particles)
	{
		p.pos = glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f) + kParticleRad, glm::vec3(0.5f, 0.5f, 0.5f) - kParticleRad);
		p.vel = glm::sphericalRand(1.0f) * glm::linearRand(0.0f, 0.4f);
	}

	thrust::device_vector<SParticle> d_particles(particles);
	auto sim = ISimulation::CreateInstance(thrust::raw_pointer_cast(d_particles.data()), kMolecules, kParticleRad);

	constexpr float kEndTime = 10.0f;
	for (float time = 0.0f; time <= kEndTime;)
	{
		float simulatedTime = sim->UpdateState(1.0f);
		time += simulatedTime;
	}

	return 0;
}