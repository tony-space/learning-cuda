#define GLM_FORCE_SWIZZLE
#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>
#include <exception>
#include <cstdio>
#include <memory>

#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\ext\matrix_transform.hpp>

#include "CShaderProgram.hpp"

static const float g_fov = 60.0f;

glm::vec4 g_camera = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
bool g_cameraRotating = false;
glm::ivec2 g_prevMousePos;
glm::ivec2 g_windowSize;

std::unique_ptr<CShaderProgram> g_phongShading;

void InitScene()
{
	glEnable(GL_DEPTH_TEST);
	g_phongShading = std::make_unique<CShaderProgram>("shaders/vertex.glsl", "shaders/fragment.glsl");
}

void RenderScene(int w, int h, bool perspective)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	if(perspective)
		gluPerspective(g_fov, GLdouble(w) / h, 0.01, 100.0);
	else
	{
		float aspect = float(w) / float(h);
		glOrtho(-1.5f * aspect, 1.5f * aspect, -1.5f, 1.5f, 0.01f, 100.0f);
	}

	glm::mat4 modelView = glm::identity<glm::mat4>();
	modelView = glm::translate(modelView, glm::vec3(0.0f, 0.0f, -g_camera.w));
	modelView = glm::rotate(modelView, glm::radians(-g_camera.x), glm::vec3(1.0f, 0.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(-g_camera.y), glm::vec3(0.0f, 1.0f, 0.0f));
	modelView = glm::rotate(modelView, glm::radians(g_camera.z), glm::vec3(0.0f, 0.0f, 1.0f));


	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(modelView));

	glutWireCube(2.0f);
	{
		auto switcher = g_phongShading->Activate();
		g_phongShading->SetUniform("u_lightDir", (modelView * glm::vec4(1.0, 1.0, 1.0, 0.0)).xyz);
		//glutSolidTeaspoon(1.0f);
		glutSolidTeacup(1.0f);
		//glutSolidSphere(1.0f, 128, 128);
	}
}

void DisplayFunc()
{
	int w = g_windowSize.x;
	int h = g_windowSize.y;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(0, h/2, w/2, h/2);
	RenderScene(w / 2, h / 2, true);

	glViewport(w/2, h / 2, w / 2, h / 2);
	RenderScene(w / 2, h / 2, false);


	glutSwapBuffers();
	glutPostRedisplay();
	glutReportErrors();
}

void MouseFunc(int button, int state, int x, int y)
{
	g_cameraRotating = state == GLUT_DOWN;
	if (g_cameraRotating)
		g_prevMousePos = glm::ivec2(x, y);
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
	if (!g_cameraRotating) return;

	auto curPos = glm::ivec2(x, y);

	auto delta = curPos - g_prevMousePos;
	g_camera.x -= delta.y * 0.1f;
	g_camera.y -= delta.x * 0.1f;

	g_prevMousePos = curPos;
}

void ReshapeFunc(int w, int h)
{
	g_windowSize = glm::ivec2(w, h);
}

void CloseFunc()
{
	g_phongShading.reset();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH /*| GLUT_BORDERLESS | GLUT_CAPTIONLESS*/);
	glutInitWindowSize(1280, 768);

	glutCreateWindow("Voxelizer");

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