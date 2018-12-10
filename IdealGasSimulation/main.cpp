#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>
#define GLM_FORCE_SWIZZLE
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <memory>
#include <chrono>
#include <sstream>

#include "CShaderProgram.hpp"

std::unique_ptr<CShaderProgram> g_spriteShader;
glm::vec4 g_camera = glm::vec4(0.0f, 0.0f, 0.0f, 1.5f);
glm::vec4 g_cameraVelocity;
glm::vec2 g_prevMouseState;
glm::vec2 g_mouseDelta;

auto g_lastFrameTime = std::chrono::high_resolution_clock::now();
float g_deltaTime = 0.0f;

static const size_t kMolecules = 1024;
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

	glm::vec3 molecules[kMolecules];
	for (size_t i = 0; i < kMolecules; ++i)
		molecules[i] = glm::linearRand(glm::vec3(-0.5f, -0.5f, -0.5f), glm::vec3(0.5f, 0.5f, 0.5f));

	glGenBuffers(1, &g_moleculesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, g_moleculesVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(molecules), molecules, GL_STREAM_DRAW);
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
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	static const float mass = 1.0f;
	static const float stiffness = 2.0f;
	static const float damp = 5.0f;
	auto force = -glm::vec2(g_mouseDelta.y, g_mouseDelta.x) * stiffness - g_cameraVelocity.xy * damp;
	auto accel = force / mass;
	g_cameraVelocity += glm::vec4(accel, 0.0f, 0.0f) * g_deltaTime;
	g_camera += g_cameraVelocity * g_deltaTime;
	g_mouseDelta *= 0.0f;

	glTranslatef(0.0f, 0.0f, -g_camera.w);
	glRotatef(g_camera.x, 1.0f, 0.0f, 0.0f);
	glRotatef(g_camera.y, 0.0f, 1.0f, 0.0f);
	glRotatef(g_camera.z, 0.0f, 0.0f, 1.0f);

	glutWireCube(1.0f);

	{
		auto smartSwitcher = g_spriteShader->Activate();
		g_spriteShader->SetUniform("pointRadius", 20.0f);

		glBindBuffer(GL_ARRAY_BUFFER, g_moleculesVBO);
		
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, nullptr);
		glDrawArrays(GL_POINTS, 0, kMolecules);
		glDisableClientState(GL_VERTEX_ARRAY);
		
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

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
	gluPerspective(60, GLdouble(w) / h, 0.01, 100.0);
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