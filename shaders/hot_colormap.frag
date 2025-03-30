#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D bhTexture;

void main() {
    // float intensity = texture(bhTexture, TexCoord).r;
    
    
    vec3 color = vec3(texture(bhTexture, TexCoord).r,
                      texture(bhTexture, TexCoord).g,
                      texture(bhTexture, TexCoord).b);

    FragColor = vec4(color, 1.0);
}