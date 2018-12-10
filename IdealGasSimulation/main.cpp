#define _USE_MATH_DEFINES
#define GLM_FORCE_SWIZZLE

#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <memory>
#include <chrono>
#include <sstream>
#include <cmath>
#include <vector>

#include "CShaderProgram.hpp"

static const float g_fov = 60.0f;
static float g_windowHeight = 1.0f;


std::unique_ptr<CShaderProgram> g_spriteShader;
glm::vec4 g_camera = glm::vec4(0.0f, 0.0f, 0.0f, 1.5f);
glm::vec4 g_cameraVelocity;
glm::vec2 g_prevMouseState;
glm::vec2 g_mouseDelta;

auto g_lastFrameTime = std::chrono::high_resolution_clock::now();
float g_deltaTime = 0.0f;

static const size_t kMolecules = 128;
GLuint g_moleculesVBO;

void InitScene()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SPRITE);

	//See the link below to read more  about GL_POINT_SPRITE_ARB
	//https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_point_sprite.txt
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);


	g_spriteShader = std::make_unique<CShaderProgram>("shaders\\vertex.glsl", "shaders\\fragment.glsl");

	std::vector<glm::vec3> molecules(kMolecules * kMolecules * 2);
	for (size_t y = 0; y < kMolecules; ++y)
		for (size_t x = 0; x < kMolecules; ++x)
		{
			size_t index = (x + y * kMolecules) * 2;
			molecules[index] = glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f));
			molecules[index + 1] = glm::linearRand(glm::vec3(0.5f, 0.5f, 0.5f), glm::vec3(1.0f, 1.0f, 1.0f));
		}

	glGenBuffers(1, &g_moleculesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, g_moleculesVBO);
	glBufferData(GL_ARRAY_BUFFER, molecules.size() * sizeof(molecules[0]), molecules.data(), GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void DisplayFunc()
{
	//
	// put computations here
	//
	auto now = std::chrono::high_resolution_clock::now();
	auto deltaTime = now - g_lastFrameTime;
	g_lastFrameTime = now;

	g_deltaTime = float(std::chrono::duration_cast<std::chrono::nanoseconds>(deltaTime).count()) * 1e-9f;

	static float __timeCounter = 0.0f;
	__timeCounter += g_deltaTime;
	if (__timeCounter > 1.0f)
	{
		std::stringstream str;
		str << "Ideal gas simulation FPS:" << int(1.0f / g_deltaTime);
		glutSetWindowTitle(str.str().c_str());
		__timeCounter = 0.0f;
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static const float mass = 1.0f;
	static const float stiffness = 2.0f;
	static const float damp = 5.0f;
	auto force = -glm::vec2(g_mouseDelta.y, g_mouseDelta.x) * stiffness - g_cameraVelocity.xy * damp;
	auto accel = force / mass;
	g_cameraVelocity += glm::vec4(accel, 0.0f, 0.0f) * g_deltaTime;
	g_camera += g_cameraVelocity * g_deltaTime;
	g_mouseDelta *= 0.0f;

	glm::mat4 modelView = glm::identity<glm::mat4>();
	modelView = glm::translate(modelView, glm::vec3(0.0f, 0.0f, -g_camera.w));
	modelView = glm::rotate(modelView, glm::radians(g_camera.x), glm::vec3(1.0f, 0.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(g_camera.y), glm::vec3(0.0f, 1.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(g_camera.z), glm::vec3(0.0f, 0.0f, 1.0f));


	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(modelView));

	glutWireCube(1.0f);

	{
		auto smartSwitcher = g_spriteShader->Activate();

		static const glm::vec4 lightDirection = glm::normalize(glm::vec4(1.0f, 1.0f, 1.0f, 0.0f));
		g_spriteShader->SetUniform("pointRadius", 0.01f);
		g_spriteShader->SetUniform("pointScale", g_windowHeight / tanf(g_fov / 2.0f *  float(M_PI) / 180.0f));
		g_spriteShader->SetUniform("lightDir", (modelView * lightDirection).xyz);

		glBindBuffer(GL_ARRAY_BUFFER, g_moleculesVBO);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glVertexPointer(3, GL_FLOAT, 24, nullptr);
		glColorPointer(3, GL_FLOAT, 24, (void*)12);
		glDrawArrays(GL_POINTS, 0, kMolecules * kMolecules);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	/*glBegin(GL_LINE_LOOP);
	glVertex2f(-0.5f, -0.5f);
	glVertex2f(0.5f, -0.5f);
	glVertex2f(0.5f, 0.5f);
	glVertex2f(-0.5f, 0.5f);
	glEnd();*/

	glutSwapBuffers();
	glutPostRedisplay();
	glutReportErrors();
}

void MouseFunc(int button, int state, int x, int y)
{
}

void MotionFunc(int x, int y)
{
	glm::vec2 mouseState(x, y);
	if (g_prevMouseState == glm::vec2(0.0f, 0.0f) || g_deltaTime == 0.0f)
	{
		g_prevMouseState = mouseState;
		return;
	}

	g_mouseDelta = mouseState - g_prevMouseState;
	if (glm::length(g_mouseDelta) > 100.0f)
		g_mouseDelta *= 0.0f;

	g_mouseDelta /= g_deltaTime;
	g_prevMouseState = mouseState;
}

void ReshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(g_fov, GLdouble(w) / h, 0.01, 100.0);
	g_windowHeight = float(h);
}

void CloseFunc()
{
	g_spriteShader.reset();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH /*| GLUT_BORDERLESS | GLUT_CAPTIONLESS*/);
	glutInitWindowSize(1366, 768);
	glutCreateWindow("Ideal gas simulation");

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);
	glutMotionFunc(MotionFunc);
	glutPassiveMotionFunc(MotionFunc);
	glutCloseFunc(CloseFunc);

	try
	{
		auto err = glewInit();
		wglSwapIntervalEXT(0);
		InitScene();

		glutMainLoop();
	}
	catch (std::exception& ex)
	{
		fprintf(stderr, "Exception handled: %s\r\n", ex.what());
	}

	return 0;
}