varying vec3 v_normal;
varying vec4 v_fragPos;

void main()
{
	v_normal = (gl_ModelViewMatrix * vec4(gl_Normal, 0.0)).xyz;
	v_fragPos = gl_ModelViewMatrix * gl_Vertex;
	gl_Position = gl_ProjectionMatrix * v_fragPos;
}