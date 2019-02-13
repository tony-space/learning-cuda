#define GLM_FORCE_SWIZZLE
#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>
#include <algorithm>
#include <memory>
#include <chrono>
#include <sstream>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\ext\matrix_transform.hpp>
#include "CScene.hpp"

static const float g_fov = 60.0f;
static float g_windowHeight = 1.0f;


std::unique_ptr<CScene> g_scene;
glm::vec4 g_camera = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
glm::vec4 g_cameraVelocity;
glm::vec2 g_prevMouseState;
glm::vec2 g_mouseDelta;
bool g_moveCamera = false;

auto g_lastFrameTime = std::chrono::high_resolution_clock::now();
float g_deltaTime = 0.0f;


void InitScene()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SPRITE);

	//See the link below to read more  about GL_POINT_SPRITE_ARB
	//https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_point_sprite.txt
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	g_scene = std::make_unique<CScene>();
}

void DisplayFunc()
{
	auto now = std::chrono::high_resolution_clock::now();
	auto deltaTime = now - g_lastFrameTime;
	g_lastFrameTime = now;

	g_deltaTime = float(std::chrono::duration_cast<std::chrono::nanoseconds>(deltaTime).count()) * 1e-9f;
	g_deltaTime = std::min(g_deltaTime, 1.0f);

	static float __timeCounter = 0.0f;
	__timeCounter += g_deltaTime;
	if (__timeCounter > 0.5f)
	{
		std::stringstream str;
		str << "Ideal gas simulation FPS: " << int(1.0f / g_deltaTime);
		glutSetWindowTitle(str.str().c_str());
		__timeCounter = 0.0f;
	}

	g_scene->UpdateState(g_deltaTime);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static const float mass = 1.0f;
	static const float stiffness = 0.05f;
	static const float damp = 5.0f;
	auto force = -glm::vec2(g_mouseDelta.y, g_mouseDelta.x) * stiffness - g_cameraVelocity.xy * damp;
	auto accel = force / mass;
	g_cameraVelocity += glm::vec4(accel, 0.0f, 0.0f) * g_deltaTime;
	g_camera += g_cameraVelocity * g_deltaTime;
	g_mouseDelta *= 0.0f;

	glm::mat4 modelView = glm::identity<glm::mat4>();
	modelView = glm::translate(modelView, glm::vec3(0.0f, 0.0f, -g_camera.w));
	modelView = glm::rotate(modelView, glm::radians(-g_camera.x), glm::vec3(1.0f, 0.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(-g_camera.y), glm::vec3(0.0f, 1.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(g_camera.z), glm::vec3(0.0f, 0.0f, 1.0f));


	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(modelView));

	glutWireCube(1.0f);

	g_scene->Render(g_windowHeight, g_fov, modelView);

	glutSwapBuffers();
	glutPostRedisplay();
	glutReportErrors();
}

void MouseFunc(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON)
	{
		if (state == GLUT_UP)
			g_moveCamera = false;
		else if (state == GLUT_DOWN)
			g_moveCamera = true;
	}
}

void MouseWheelFunc(int button, int dir, int x, int y)
{
	if (dir > 0)
	{
		g_camera.w *= 9.0f / 10.0f;
	}
	else
	{
		g_camera.w *= 10.0f / 9.0f;
	}
}

void MotionFunc(int x, int y)
{
	glm::vec2 mouseState(x, y);
	if (g_prevMouseState == glm::vec2(0.0f, 0.0f) || g_deltaTime == 0.0f || !g_moveCamera)
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
	g_scene.reset();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	//glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH /*| GLUT_BORDERLESS | GLUT_CAPTIONLESS*/);
	//glutInitWindowSize(1366, 768);
	
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_BORDERLESS | GLUT_CAPTIONLESS);
	glutInitWindowSize(1920, 1080);

	glutCreateWindow("Ideal gas simulation");

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);
	glutMouseWheelFunc(MouseWheelFunc);
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