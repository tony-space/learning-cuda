uniform vec3 lightDir;
void main()
{
	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);

	if (mag > 1.0) discard;   // kill pixels outside circle

	N.z = sqrt(1.0 - mag);

	// calculate lighting
	float diffuse = max(0.1, dot(lightDir, N));

	gl_FragColor = gl_Color * diffuse;
}