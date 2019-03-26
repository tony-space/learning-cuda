uniform vec3 u_lightDir;
varying vec3 v_normal;
varying vec4 v_fragPos;

void main()
{
	vec3 normal = normalize(v_normal);
	vec3 lightDir = normalize(u_lightDir);
	vec3 viewDir = normalize(-v_fragPos.xyz);
	vec3 reflectDir = reflect(-lightDir, normal);

	float ambient = 0.2;
	float diffuse = clamp(dot(normal, lightDir), 0.0, 1.0);
	float specular = pow(clamp(dot(viewDir, reflectDir), 0.0, 1.0), 32.0);

	vec3 color = vec3(0.0);
	color += vec3(1.0, 1.0, 1.0) * ambient;
	color += vec3(1.0, 1.0, 0.0) * diffuse;
	color += vec3(0.0, 0.0, 1.0) * specular;

	gl_FragColor = vec4(color, 1.0);
}