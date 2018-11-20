#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cassert>
#include "kernel.hpp"

GLuint electricFieldTexture = -1;
static const unsigned kTextureSize = 8192;

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

	glGenTextures(1, &electricFieldTexture);
	error = glGetError();
	assert(!error);

	glBindTexture(GL_TEXTURE_2D, electricFieldTexture);
	error = glGetError();
	assert(!error);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	error = glGetError();
	assert(!error);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	error = glGetError();
	assert(!error);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, kTextureSize, kTextureSize, 0, GL_RGBA, GL_FLOAT, nullptr);
	error = glGetError();
	assert(!error);
}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	ProcessElectronField(electricFieldTexture, kTextureSize, kTextureSize);

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(-1.0f, -1.0f);

	glTexCoord2f(1.0, 0.0);
	glVertex2f(1.0f, -1.0f);

	glTexCoord2f(1.0, 1.0);
	glVertex2f(1.0f, 1.0f);

	glTexCoord2f(0.0, 1.0);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	glutSwapBuffers();
	//glutPostRedisplay();
}

void MouseFunc(int button, int state, int x, int y)
{
}

void ReshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(900, 900);
	glutCreateWindow("Static electric field");

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);

	auto err = glewInit();
	InitScene();

	glutMainLoop();

	return 0;
}