#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "ISimulation.hpp"

//static const size_t kMolecules = 16384;
static const size_t kMolecules = 128;
static const float kParticleRad = 0.002f;

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
	auto sim = ISimulation::CreateInstance(d_particles.data().get(), kMolecules, kParticleRad);

	sim->UpdateState(1.0f);
	sim->UpdateState(1.0f);
	sim->UpdateState(1.0f);
	sim->UpdateState(1.0f);
	sim->UpdateState(1.0f);

	//constexpr float kEndTime = 10.0f;
	//for (float time = 0.0f; time <= kEndTime;)
	//{
	//	auto simulatedTime = sim->UpdateState(1.0f);
	//	auto lastTime = int(std::round(time * 10.0f));
	//	time += simulatedTime;
	//	auto current = int(std::round(time * 10.0f));
	//	if (current > lastTime)
	//		printf("%.1f\n", time);
	//}

	return 0;
}