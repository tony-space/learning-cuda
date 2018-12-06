#pragma once
#include <gl\glew.h>
#include <map>
#include <string>

class CShaderProgram
{
public:
	class ProgramSwitcher
	{
	public:
		ProgramSwitcher(GLuint program);
		//ProgramSwitcher(const ProgramSwitcher&) = delete;
		//ProgramSwitcher(ProgramSwitcher&&) = delete;
		~ProgramSwitcher();
	private:
		GLuint m_previous = -1;
	};


	CShaderProgram(const std::string& vertexPath, const std::string& fragmentPath);
	~CShaderProgram();
	void SetUniform(const std::string& name, float value);
	ProgramSwitcher Activate() const { return ProgramSwitcher(m_program); }
private:
	GLuint m_program;
	std::map<std::string, GLint> m_uniformLocationCache;
};
