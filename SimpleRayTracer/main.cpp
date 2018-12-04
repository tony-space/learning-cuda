#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cassert>
#include <memory>
#include <chrono>

#include "kernel.hpp"
#include "CScene.hpp"

static const unsigned kTextureWidth = 8192;
static const unsigned kTextureHeight = 8192;

std::unique_ptr<CScene> g_scene;

void InitScene()
{
	GLenum error;
	glEnable(GL_TEXTURE_2D);
	error = glGetError();
	assert(!error);

	GLint texSize;
	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &texSize);
	error = glGetError();
	assert(!error);

	g_scene = std::make_unique<CScene>(kTextureWidth, kTextureHeight, 1.0f);
}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	static auto lastTime = std::chrono::system_clock::now();
	auto now = std::chrono::system_clock::now();
	auto delta = now - lastTime;
	auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(delta);
	float dt = microseconds.count() / 1000000.0f;
	lastTime = now;

	g_scene->Render(dt);
	//printf("FPS: %.1f\r\n", 1 / dt);

	glutSwapBuffers();
	glutPostRedisplay();
}

void MouseFunc(int button, int state, int x, int y)
{
}

void ReshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);
	/*glMatrixMode(GL_PROJECTION);
	glLoadIdentity();*/

	//double aspectRatio = double(w) / double(h);

	//glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH /*| GLUT_BORDERLESS | GLUT_CAPTIONLESS*/);
	glutInitWindowSize(900, 900);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("Simple raytracer");

	/*glutGameModeString("1920x1080");
	glutEnterGameMode();
	*/
	typedef bool (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)        (int interval);
	PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
	wglSwapIntervalEXT(1);


	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);

	auto err = glewInit();
	InitScene();

	glutMainLoop();

	return 0;
}