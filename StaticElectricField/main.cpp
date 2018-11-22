#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cassert>
#include <memory>
#include <chrono>

#include "kernel.hpp"
#include "CElectricField.hpp"

static const unsigned kTextureWidth = 8192;
static const unsigned kTextureHeight = 8192;

std::unique_ptr<CElectricField> g_electricField;

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

	g_electricField = std::make_unique<CElectricField>(kTextureWidth, kTextureHeight, 1920.0f / 1080.0f);
	CElectricField::SParticle electron = { {0.0f, -0.75f}, {0.5f, 0.0f}, 1, -1 };
	CElectricField::SParticle electron2 = { {0.5f, -0.75f}, {0.5f, 0.0f}, 1, -1 };
	CElectricField::SParticle electron3 = { {0.5f, 0.75f}, {0.0f, 0.0f}, 1, -1 };
	CElectricField::SParticle proton = { {0.0f, 0.0f}, {0.0f, 0.0f}, 1836, 1 };
	CElectricField::SParticle positron = { {-0.8f, -0.8f}, {0.0f, 0.25f}, 1, 1 };
	g_electricField->AddParticle(electron);
	g_electricField->AddParticle(electron2);
	g_electricField->AddParticle(electron3);
	g_electricField->AddParticle(proton);
	g_electricField->AddParticle(positron);
}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	static auto lastTime = std::chrono::system_clock::now();
	auto now = std::chrono::system_clock::now();
	auto delta = now - lastTime;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
	float dt = milliseconds.count() / 1000.0f;
	lastTime = now;

	g_electricField->Render(dt);
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
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_BORDERLESS | GLUT_CAPTIONLESS);
	glutInitWindowSize(1920, 1080);
	glutCreateWindow("Static electric field");

	/*glutGameModeString("1920x1080");
	glutEnterGameMode();
	*/
	typedef bool (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)        (int interval);
	PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
	wglSwapIntervalEXT(0);


	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);

	auto err = glewInit();
	InitScene();

	glutMainLoop();

	return 0;
}