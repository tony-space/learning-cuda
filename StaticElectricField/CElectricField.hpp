#pragma once
#include <GL/glew.h>
#include <vector_types.h>
#include <thrust/device_vector.h>


class CElectricField
{
public:
	struct SParticle
	{
		float2 position;
		float2 velocity;
		float mass;
		float charge;
	};

	CElectricField(GLsizei width, GLsizei height);
	~CElectricField();

	void AddParticle(const SParticle& p);
	void Render(float dt);

private:
	GLuint m_texture;
	GLsizei m_width;
	GLsizei m_height;

	thrust::host_vector<SParticle> m_hostParticles;

	thrust::device_vector<SParticle> m_deviceParticles;
	thrust::device_vector<float3> m_deviceVectorField;

	void UpdateState(float dt);
};
