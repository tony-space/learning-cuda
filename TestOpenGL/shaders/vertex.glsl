#version 460
/*layout(location = 0)*/ in vec3 position;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;

out vec3 vertexColor;

void main()
{
	vec4 pos4 = vec4(position, 1);

	vertexColor = (model * pos4).xyz;

	gl_Position= projection * view * model * pos4;
}