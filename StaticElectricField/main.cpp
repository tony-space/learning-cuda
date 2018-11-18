#include <GL\glew.h>
#include <GL\freeglut.h>
#include <cassert>
#include "kernel.hpp"

GLuint electricFieldTexture = -1;
//GLuint pboUnpackedBuffer = -1;

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

	//glGenBuffers(1, &pboUnpackedBuffer);
	//error = glGetError();
	//assert(!error);

	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboUnpackedBuffer);
	//error = glGetError();
	//assert(!error);

	//glBufferData(GL_PIXEL_UNPACK_BUFFER, kTextureSize * kTextureSize * 4 * sizeof(float), nullptr, GL_DYNAMIC_COPY);
	//error = glGetError();
	//assert(!error);

	//TextureFetchTest();
	//OpenGLTextureFetchTest(electricFieldTexture);
}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT);
	
	//GeneratePBO(pboUnpackedBuffer, kTextureSize, kTextureSize);
	ModifyTexture(electricFieldTexture, kTextureSize, kTextureSize);

	//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 2048, 2048, GL_RGBA, GL_FLOAT, nullptr);
	//error = glGetError();
	//assert(!error);

	glBegin(GL_QUADS);
	//glColor3f(0.0f, 0.0f, 0.0f);
	glTexCoord2f(0.0, 0.0);
	glVertex2f(-1.0f, -1.0f);

	//glColor3f(1.0f, 0.0f, 0.0f);
	glTexCoord2f(1.0, 0.0);
	glVertex2f(1.0f, -1.0f);

	//glColor3f(0.0f, 1.0f, 0.0f);
	glTexCoord2f(1.0, 1.0);
	glVertex2f(1.0f, 1.0f);

	//glColor3f(0.0f, 0.0f, 1.0f);
	glTexCoord2f(0.0, 1.0);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();
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