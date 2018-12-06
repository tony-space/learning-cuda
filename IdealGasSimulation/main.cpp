#include <GL\glew.h>
#include <GL\wglew.h>
#include <GL\freeglut.h>
#include <glm\glm.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <glm\gtc\random.hpp>
#include <memory>

#include "CShaderProgram.hpp"

std::unique_ptr<CShaderProgram> g_spriteShader;
//See the link below to read more  about GL_POINT_SPRITE_ARB
//https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_point_sprite.txt
void InitScene()
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glPointSize(10.0f);


	g_spriteShader = std::make_unique<CShaderProgram>("shaders\\vertex.glsl", "shaders\\fragment.glsl");

}

void DisplayFunc()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glTranslatef(0.0f, 0.0f, -2.0f);
	glutWireCube(1.0f);

	{
		auto smartSwitcher = g_spriteShader->Activate();
		g_spriteShader->SetUniform("pointRadius", 1.0f);
		g_spriteShader->SetUniform("pointScale", 50.0f);
		glBegin(GL_POINTS);
		//glVertex3f(0.0f, 0.0f, 0.0f);
		for (size_t i = 0; i < 100; ++i)
			glVertex3fv(glm::value_ptr(glm::linearRand(glm::vec3(-1.0f, -1.0f, -1.0f), glm::vec3(1.0f, 1.0f, 1.0f)) + glm::vec3(0.0f, 0.0f, -2.0f)));
		glEnd();
	}

	glutSwapBuffers();
	//glutPostRedisplay();
}

void MouseFunc(int button, int state, int x, int y)
{
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

	/*glutGameModeString("1920x1080");
	glutEnterGameMode();
	*/

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutMouseFunc(MouseFunc);
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