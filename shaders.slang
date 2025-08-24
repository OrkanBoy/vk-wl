struct VSInput {
  [[vk::location(0)]]
  uint32_t surface;
};
struct Uniform {
  float4 light_affine[3];

  float3 light_direction;
  uint padding;

  float3 camera_position;
  float  camera_near;

  float2 camera_cos;
  float2 camera_sin;
  float2 camera_scale;
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

namespace shadow {
[vk_binding(0,0)] 
ConstantBuffer<Uniform> uniform;

struct VSOutput {
  noperspective float4 position : SV_Position;
};

[shader("vertex")]
VSOutput vertexShadow(VSInput input, uint vertex_id: SV_VertexID) {
  VSOutput output;

  float3 position;

  var surface = input.surface;
  for (uint _direction = 0; _direction != directions; _direction += 1) {
    uint direction = directions - 1 - _direction;
    position[direction] = float(surface & ((1 << log_side_len) - 1));
    surface >>= log_side_len;
  }
  let surface_direction_unsigned = surface & ((1 << log_directions) - 1);
  surface >>= log_directions;
  let surface_direction_sign = surface & 1;
  surface >>= 1;

  position[surface_direction_unsigned] += float(surface_direction_sign);

  if (surface_direction_sign == 1) {
    switch (vertex_id) {
    case 1:
      vertex_id = 2;
      break;
    case 2:
      vertex_id = 1;
    }
  }
  position[(surface_direction_unsigned + 2) % directions] += float((vertex_id & 0b01) >> 0);
  position[(surface_direction_unsigned + 1) % directions] += float((vertex_id & 0b10) >> 1);

  position = applyAffine(position, uniform.light_affine);
  output.position = { 
    position[0], 
    position[1], 
    position[2], 
    1.0,
  };
  return output;
}
}


namespace geometry {
[vk_binding(0,0)]
ConstantBuffer<Uniform> uniform;

struct VSOutput {
  float4 clip_position: SV_Position;
  uint32_t surface_direction_signed;
};


[shader("vertex")]
VSOutput vertexGeometry(VSInput input, uint vertex_id: SV_VertexID) {
  VSOutput output;

  float3 position;

  var surface = input.surface;
  for (uint _direction = 0; _direction != directions; _direction += 1) {
    uint direction = directions - _direction - 1;
    position[direction] =
        float(surface & ((1 << log_side_len) - 1));
    surface >>= log_side_len;
  }
  output.surface_direction_signed = surface & ((1 << (log_directions + 1)) - 1);
  let surface_direction_unsigned = surface & ((1 << log_directions) - 1);
  surface >>= log_directions;
  let surface_direction_sign = surface & 1;
  surface >>= 1;

  position[surface_direction_unsigned] += float(surface_direction_sign);
  if (surface_direction_sign == 1) {
    switch (vertex_id) {
    case 1:
      vertex_id = 2;
      break;
    case 2:
      vertex_id = 1;
    }
  }
  position[(surface_direction_unsigned + 2) % directions] += float((vertex_id & 0b01) >> 0);
  position[(surface_direction_unsigned + 1) % directions] += float((vertex_id & 0b10) >> 1);

  position -= uniform.camera_position;

  for (var direction = 0; direction != directions - 1; direction += 1) {
    let adj = position[directions - 1];
    let opp = position[direction];
    position[directions - 1] = 
      adj * uniform.camera_cos[direction] + 
      opp * uniform.camera_sin[direction];
    position[direction] = 
      opp * uniform.camera_cos[direction] - 
      adj * uniform.camera_sin[direction];
  }

  position[directions - 1] += uniform.camera_near;

  for (var direction = 0; direction != directions - 1; direction += 1) {
    position[direction] *= uniform.camera_scale[direction];
  }

  output.clip_position = {
    position[0],
    position[1],
    uniform.camera_near,
    position[2],
  };
  return output;
}

[shader("fragment")]
uint fragmentGeometry(VSOutput input) {
  return input.surface_direction_signed;
}
}

namespace lighting {
struct VSOutput {
  noperspective float4 clip_position: SV_Position;
  noperspective float2 sample_position;
}
static const float2 positions[3] = {
  float2(-3,  1),
  float2( 1,  1),
  float2( 1, -3),
};

[shader("vertex")]
VSOutput vertexLighting(uint vertex_id: SV_VertexID) {
  VSOutput output;
  output.clip_position = {positions[vertex_id], 0.0, 1.0};
  output.sample_position = positions[vertex_id] * 0.5 + 0.5;
  return output;
}

[vk_binding(0,0)]
ConstantBuffer<Uniform> uniform;
[vk_binding(1,0)]
Sampler2DShadow shadow_sampler;
[vk_binding(2,0)]
Sampler2DShadow depth_sampler;
[vk_binding(3,0)]
Sampler2D<uint> surface_direction_sampler;

[shader("fragment")]
float3 fragmentLighting(VSOutput input) {
  let depth = depth_sampler.Sample(input.sample_position);
  if (depth == 0.0) {
    return {0.0, 0.0, 0.0};
  }

  float3 output; 

  var direction_signed = surface_direction_sampler.Sample(input.sample_position);
  let direction_unsigned = direction_signed & ((1 << log_directions) - 1);
  direction_signed >>= log_directions;
  let direction_sign = direction_signed & 1;

  for (var direction = 0; direction != directions; direction += 1) {
    output[direction] = float(direction_sign ^ (direction != direction_unsigned));
  }
  
  return output;
}

}
