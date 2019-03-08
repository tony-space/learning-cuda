uniform float pointScale;   // scale to calculate size in pixels
uniform float radius;

in vec3 pos;
in vec3 color;
//in float radius;

void main()
{
	// calculate window-space point size
	vec3 posEye = vec3(gl_ModelViewMatrix * vec4(pos, 1));
	float dist = length(posEye);
	gl_PointSize = radius * (pointScale / dist);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);

	gl_FrontColor = vec4(color, 1);
}