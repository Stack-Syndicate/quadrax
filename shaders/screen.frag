#version 460

layout(location = 0) out vec4 f_color;

void main() {
  float x_pos = gl_FragCoord.x;
  f_color = vec4(abs(sin(x_pos * 0.05)), abs(cos(x_pos * 0.05)), 0.0, 1.0);
}
