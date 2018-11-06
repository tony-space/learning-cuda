#include <GL\glew.h>
#include <GL\freeglut.h>
#include "kernel.hpp"

GLuint electricFieldTexture = -1;

void InitScene()
{
	GLenum error;
	glEnable(GL_TEXTURE_2D);
	error = glGetError();

	glGenTextures(1, &electricFieldTexture);
	error = glGetError();

	glBindTexture(GL_TEXTURE_2D, electricFieldTexture);
	error = glGetError();

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	error = glGetError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	error = glGetError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	error = glGetError();
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	error = glGetError();

	float pixels[] =
	{
		1.0f, 1.0f, 1.0f,
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	};

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 2, 2, 0, GL_RGB, GL_FLOAT, pixels);
	error = glGetError();

	//TextureFetchTest();
	OpenGLTextureFetchTest(electricFieldTexture);
}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT);

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
	glutInitWindowSize(1366, 768);
	glutCreateWindow("Static electric field");

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);

	auto err = glewInit();
	InitScene();

	glutMainLoop();

	return 0;
}