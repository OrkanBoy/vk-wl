struct VSInput {
  [[vk::location(0)]]
  uint32_t surface;
};

namespace lighting {
struct Uniform {
  Light light;
  Camera camera;
};

struct Light {
  vector<float32_t, directions + 1> affine[directions];
  vector<float32_t, directions> direction;
  uint32_t padding;
}

struct Camera {
  vector<float32_t, directions> position;

  float32_t near;

  vector<float32_t, directions - 1> cos;
  vector<float32_t, directions - 1> sin;
  vector<float32_t, directions - 1> scale;
};

[vk_binding(0, 0)]
ConstantBuffer<Uniform> uniform;

struct VSOutput {
  vector<float32_t, directions + 1> clip_position : SV_Position;
  vector<float32_t, directions> world_position;
  vector<float32_t, directions> light_view_position;

  nointerpolation float32_t diffuse;
  nointerpolation float32_t ambient;
  nointerpolation vector<float32_t, 3> color;
  nointerpolation uint surface_direction;
  nointerpolation uint surface_direction_sign;
};

static const float diffuse_strength = 0.60;
static const float specular_strength = 0.20;
static const float ambient_strength = 1.0 - diffuse_strength - specular_strength;

[shader("vertex")]
VSOutput vertexLighting(VSInput input, uint vertex_id: SV_VertexID) {
  VSOutput output;

  vector<float32_t, 3> position;

  uint surface = input.surface;
  for (uint direction = 0; direction != directions; direction += 1) {
    position[directions - direction - 1] =
        float(surface & ((1 << log_side_len) - 1));
    surface >>= log_side_len;
  }
  uint surface_direction = surface & ((1 << log_directions) - 1);
  surface >>= log_directions;
  uint surface_direction_sign = surface & 1;
  surface >>= 1;
  uint surface_surrounding_air = surface & ((1 << log_surrounding_air) - 1);

  output.surface_direction = surface_direction;
  output.surface_direction_sign = surface_direction_sign;

  float ambient = float(surface_surrounding_air) / float(max_surrounding_air);
  ambient *= ambient_strength;


  float diffuse = uniform.light.direction[surface_direction];
  if (surface_direction_sign == 1) {
    diffuse = -uniform.light.direction[surface_direction];
  }
  diffuse = max(0.0, diffuse);
  diffuse *= diffuse_strength;


  position[surface_direction] += float(surface_direction_sign);
  if (surface_direction_sign == 1) {
    switch (vertex_id) {
    case 1:
      vertex_id = 2;
      break;
    case 2:
      vertex_id = 1;
    }
  }
  position[(surface_direction + 2) % directions] += float((vertex_id & 0b01) >> 0);
  position[(surface_direction + 1) % directions] += float((vertex_id & 0b10) >> 1);

  output.world_position = position;

  for (uint direction_i = 0; direction_i != directions; direction_i += 1) {
    output.color[direction_i] =
        float(surface_direction_sign ^ (direction_i != surface_direction));
  }
  output.ambient = ambient;
  output.diffuse = diffuse;

  output.light_view_position = applyAffine(position, uniform.light.affine);

  position -= uniform.camera.position;

  for (uint direction = 0; direction != directions - 1; direction += 1) {
    float adj = position[2];
    float opp = position[direction];
    position[directions - 1] = 
      adj * uniform.camera.cos[direction] + 
      opp * uniform.camera.sin[direction];
    position[direction] = 
      opp * uniform.camera.cos[direction] - 
      adj * uniform.camera.sin[direction];
  }

  position.z += uniform.camera.near;

  for (uint direction = 0; direction != directions - 1; direction += 1) {
    position[direction] *= uniform.camera.scale[direction];
  }

  output.clip_position = {
    position.x,
    position.y,
    uniform.camera.near,
    position.z,
  };
  return output;
}
[vk_binding(1, 0)]
Sampler2DShadow shadow;

[shader("fragment")]
vector<float32_t, 4> fragmentLighting(VSOutput input) {
  float32_t x = input.light_view_position.x / 2.0 + 0.5;
  float32_t y = input.light_view_position.y / 2.0 + 0.5;
  float32_t z = input.light_view_position.z;

  float light = shadow.SampleCmp(float2(x, y), z);

  float3 camera_direction = normalize(input.world_position - uniform.camera.position);
  float3 halfway = normalize(uniform.light.direction + camera_direction);
  float specular = halfway[input.surface_direction] / length(halfway);
  if (input.surface_direction_sign == 1) {
    specular = -specular;
  }
  specular = pow(max(0.0, specular), 2);
  specular *= specular_strength;

  return { input.color * ((input.diffuse + specular) * light + input.ambient), 1.0 };
}
}

namespace shadow {
struct Uniform {
  vector<float32_t, directions + 1> light_affine[directions];
  vector<float32_t, directions> light_direction;
};
[vk_binding(0,0)] 
ConstantBuffer<Uniform> uniform;

struct VSOutput {
  nointerpolation vector<float32_t, 4> position : SV_Position;
};

[shader("vertex")]
VSOutput vertexShadow(VSInput input, uint vertex_id: SV_VertexID) {
  VSOutput output;

  vector<float32_t, 3> position;

  uint surface = input.surface;
  for (uint _direction_i = 0; _direction_i != directions; _direction_i += 1) {
    uint direction_i = directions - 1 - _direction_i;
    position[direction_i] = float(surface & ((1 << log_side_len) - 1));
    surface >>= log_side_len;
  }
  uint surface_direction = surface & ((1 << log_directions) - 1);
  surface >>= log_directions;
  uint surface_direction_sign = surface & 1;
  surface >>= 1;

  position[surface_direction] += float(surface_direction_sign);

  if (surface_direction_sign == 1) {
    switch (vertex_id) {
    case 1:
      vertex_id = 2;
      break;
    case 2:
      vertex_id = 1;
    }
  }
  position[(surface_direction + 2) % directions] += float((vertex_id & 0b01) >> 0);
  position[(surface_direction + 1) % directions] += float((vertex_id & 0b10) >> 1);

  position = applyAffine(position, uniform.light_affine);
  output.position = { position[0], position[1], position[2], 1.0 };
  return output;
}
}


vector<T, N> applyAffine<T : __BuiltinFloatingPointType, let N : int>(
    vector<T, N> input, vector<T, N + 1> affine[N], ) {
  vector<T, N> output;
  for (uint i = 0; i != N; i += 1) {
    output[i] = affine[i][N];
    for (uint j = 0; j != N; j += 1) {
      output[i] += affine[i][j] * input[j];
    }
  }
  return output;
}

static const uint log_side_len = 0x5;

static const uint directions = 3;
static const uint log_directions = uint(ceil(log2(float(directions))));

static const uint max_surrounding_air = (directions << 1) - 1;
static const uint log_surrounding_air = uint(ceil(log2(float(max_surrounding_air))));


