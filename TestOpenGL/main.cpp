#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <glm\vec3.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glu32.lib")
#pragma comment(lib, "winmm.lib")

struct Camera
{
	float fov;
	glm::vec3 position;
	glm::vec3 rotation;

	int width;
	int height;

	Camera() : width(800), height(600), fov(80), position(0, 0, -2), rotation()
	{

	}
};



Camera camera;
int windowDesc;
int xRotationState = 0;
int yRotationState = 0;
int camDistancingState = 0;
unsigned vertexShader;
unsigned fragmentShader;
unsigned shaderProgram;
std::chrono::high_resolution_clock::time_point lastTime;



void ReshapeFunc(int w, int h)
{
	glViewport(0, 0, w, h);
	camera.width = w;
	camera.height = h;
}

void Render(float dt)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::mat4 projection = glm::perspectiveFov(glm::radians(camera.fov), float(camera.width), float(camera.height), 0.01f, 100.0f);

	glm::mat4 view(1.0f);
	view = glm::translate(view, camera.position);
	view = glm::rotate(view, camera.rotation.x, glm::vec3(1, 0, 0));
	view = glm::rotate(view, camera.rotation.y, glm::vec3(0, 1, 0));

	glm::mat4 model(1.0);
	model = glm::translate(model, glm::vec3(0.5, 0.5, 0.5));

	auto projUniform = glGetUniformLocation(shaderProgram, "projection");
	auto viewUniform = glGetUniformLocation(shaderProgram, "view");
	auto modelUniform = glGetUniformLocation(shaderProgram, "model");
	glProgramUniformMatrix4fv(shaderProgram, projUniform, 1, false, glm::value_ptr(projection));
	glProgramUniformMatrix4fv(shaderProgram, viewUniform, 1, false, glm::value_ptr(view));
	glProgramUniformMatrix4fv(shaderProgram, modelUniform, 1, false, glm::value_ptr(model));

	glutSolidCube(1.0);
	/*glCullFace(GL_FRONT);
	glutSolidTeapot(1.0);
	glCullFace(GL_BACK);*/

	glutSwapBuffers();
}

void DisplayFunc()
{
	float delta = 0.0f;

	auto now = std::chrono::high_resolution_clock::now();

	if (lastTime.time_since_epoch().count() != 0)
	{
		auto duration = now - lastTime;
		delta = static_cast<float>(duration.count()) * 1e-9f;
	}
	lastTime = now;

	camera.rotation.x += delta * xRotationState;
	camera.rotation.y += delta * yRotationState;
	camera.position.z += delta * camDistancingState;

	Render(delta);
	glutPostRedisplay();
}

void SpecialFunc(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		xRotationState = 1;
		break;

	case GLUT_KEY_DOWN:
		xRotationState = -1;
		break;

	case GLUT_KEY_LEFT:
		yRotationState = 1;
		break;

	case GLUT_KEY_RIGHT:
		yRotationState = -1;
		break;

	case GLUT_KEY_PAGE_UP:
		camDistancingState = 1;
		break;

	case GLUT_KEY_PAGE_DOWN:
		camDistancingState = -1;
		break;
	}
}

void SpecialFuncUp(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
	case GLUT_KEY_DOWN:
		xRotationState = 0;
		break;

	case GLUT_KEY_LEFT:
	case GLUT_KEY_RIGHT:
		yRotationState = 0;
		break;

	case GLUT_KEY_PAGE_UP:
	case GLUT_KEY_PAGE_DOWN:
		camDistancingState = 0;
		break;
	}
}

void KeyboardFunc(unsigned char key, int x, int y)
{
}

void MouseFunc(int button, int state, int x, int y)
{
}

void InitScene()
{
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	shaderProgram = glCreateProgram();

	std::ifstream vertSourceFile("./shaders/vertex.glsl");
	std::string vertSource = std::string(std::istreambuf_iterator<char>(vertSourceFile), std::istreambuf_iterator<char>());
	const char* strings[1] = { vertSource.c_str() };
	int length = (int)vertSource.length();
	glShaderSource(vertexShader, 1, strings, &length);
	glCompileShader(vertexShader);

	GLint success = 0;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		GLint logSize = 0;
		glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logSize);

		std::string log;
		log.resize(logSize);
		glGetShaderInfoLog(vertexShader, logSize, &logSize, const_cast<char*>(log.data()));
		std::cout << log << std::endl;
		glDeleteShader(vertexShader);
		throw std::runtime_error("compilation failed");
	}

	std::ifstream fragSourceFile("./shaders/fragment.glsl");
	std::string fragSource = std::string(std::istreambuf_iterator<char>(fragSourceFile), std::istreambuf_iterator<char>());
	strings[0] = fragSource.c_str();
	length = (int)fragSource.length();
	glShaderSource(fragmentShader, 1, strings, &length);
	glCompileShader(fragmentShader);

	success = 0;
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		GLint logSize = 0;
		glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logSize);

		std::string log;
		log.resize(logSize);
		glGetShaderInfoLog(fragmentShader, logSize, &logSize, const_cast<char*>(log.data()));
		std::cout << log << std::endl;
		glDeleteShader(fragmentShader);
		throw std::runtime_error("compilation failed");
	}

	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	success = 0;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

	if (!success)
	{
		GLint logSize = 0;
		glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logSize);

		std::string log;
		log.resize(logSize);
		glGetProgramInfoLog(shaderProgram, logSize, &logSize, const_cast<char*>(log.data()));
		std::cout << log << std::endl;
		glDeleteShader(fragmentShader);
		throw std::runtime_error("compilation failed");
	}
	glUseProgram(shaderProgram);
}

void FreeScene()
{
	glDeleteProgram(shaderProgram);
	glDeleteShader(fragmentShader);
	glDeleteShader(vertexShader);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

	/*glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(1366, 768);
	windowsDesc = glutCreateWindow("Hello OpenGL");*/

	glutGameModeString("1920x1080");
	glutEnterGameMode();
	windowDesc = glutGetWindow();

	glutReshapeFunc(ReshapeFunc);
	glutDisplayFunc(DisplayFunc);
	glutSpecialFunc(SpecialFunc);
	glutSpecialUpFunc(SpecialFuncUp);
	glutKeyboardFunc(KeyboardFunc);
	glutMouseFunc(MouseFunc);

	auto err = glewInit();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	InitScene();

	glutMainLoop();

	FreeScene();

	return 0;
}