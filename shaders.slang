struct VSInput {
  [[vk::location(0)]]
  uint surface;
};
struct Uniform {
  float4 light_affine[3];

  float3 light_direction;
  uint padding;

  float3 camera_position;
  float camera_near;

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

static const uint log_side_len = 0x4;

static const uint directions = 3;

static const uint log_directions = uint(ceil(log2(float(directions))));
static const uint log_ambient = directions - 1;
static const uint log_ambients = (1 << (directions - 1)) * (directions - 1);
static const uint ambient_mask = (1 << log_ambient) - 1;

namespace shadow {
[vk_binding(0)]
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
    case 0b01:
      vertex_id = 0b10;
      break;
    case 0b10:
      vertex_id = 0b01;
    }
  }
  position[(surface_direction_unsigned + 0b01) % directions] +=
      float((vertex_id & 0b01) >> 0);
  position[(surface_direction_unsigned + 0b10) % directions] +=
      float((vertex_id & 0b10) >> 1);

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
[vk_binding(0)]
ConstantBuffer<Uniform> uniform;

struct VSOutput {
  float4 clip_position : SV_Position;
  nointerpolation uint surface_direction_signed;
  float ambient;
};

[shader("vertex")]
VSOutput vertexGeometry(VSInput input, uint vertex_id: SV_VertexID) {
  var output : VSOutput;

  var indices : uint3;

  var surface = input.surface;
  for (var _direction = 0; _direction != directions; _direction += 1) {
    let direction = directions - _direction - 1;
    indices[direction] = surface & ((1 << log_side_len) - 1);
    surface >>= log_side_len;
  }

  output.surface_direction_signed = surface & ((1 << (log_directions + 1)) - 1);
  let surface_direction_unsigned = surface & ((1 << log_directions) - 1);
  surface >>= log_directions;
  let surface_direction_sign = surface & 1;
  surface >>= 1;
  let ambients = surface & ((1 << log_ambients) - 1);
  surface >>= log_ambients;

  if (surface_direction_sign == 1) {
    switch (vertex_id) {
    case 0b01:
      vertex_id = 0b10;
      break;
    case 0b10:
      vertex_id = 0b01;
      break;
    }
  }

  if (((ambients >> (0b00 * log_ambient)) & ambient_mask) +
          ((ambients >> (0b11 * log_ambient)) & ambient_mask) >
      ((ambients >> (0b01 * log_ambient)) & ambient_mask) +
          ((ambients >> (0b10 * log_ambient)) & ambient_mask)) {
    switch (vertex_id) {
    case 0b00:
      vertex_id = 0b01;
      break;
    case 0b01:
      vertex_id = 0b11;
      break;
    case 0b11:
      vertex_id = 0b10;
      break;
    case 0b10:
      vertex_id = 0b00;
      break;
    }
  }

  var ambient_sample_indices = indices;
  ambient_sample_indices[surface_direction_unsigned] +=
      surface_direction_sign == 1 ? +1 : -1;

  indices[surface_direction_unsigned] += surface_direction_sign;
  indices[(surface_direction_unsigned + 0b01) % directions] +=
      (vertex_id & 0b01) >> 0;
  indices[(surface_direction_unsigned + 0b10) % directions] +=
      (vertex_id & 0b10) >> 1;

  var position = float3(indices);

  position -= uniform.camera_position;

  for (var direction = 0; direction != directions - 1; direction += 1) {
    let adj = position[directions - 1];
    let opp = position[direction];
    position[directions - 1] = adj * uniform.camera_cos[direction] +
                               opp * uniform.camera_sin[direction];
    position[direction] = opp * uniform.camera_cos[direction] -
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

  uint ambient = (ambients >> ((directions - 1) * vertex_id)) &
                 ((1 << (directions - 1)) - 1);
  output.ambient = float(ambient + 1) / float(1 << (directions - 1));
  return output;
}

struct FSOutput {
  [vk_location(0)]
  uint2 geometry;
}

[shader("fragment")]
uint2 fragmentGeometry(VSOutput input) {
  return uint2(input.surface_direction_signed, bit_cast<uint>(input.ambient));
}
}

struct FullScreenVSOutput {
  noperspective float4 clip_position : SV_Position;
  noperspective float2 position;
}

[shader("vertex")]
FullScreenVSOutput vertexFullScreen(uint vertex_id: SV_VertexID) {
  FullScreenVSOutput output;
  static const float2 positions[3] = {
    float2(-3, 1),
    float2(1, 1),
    float2(1, -3),
  };

  output.clip_position = { positions[vertex_id], 0.0, 1.0 };
  output.position = positions[vertex_id];
  return output;
}

namespace color {

[vk_binding(0)]
ConstantBuffer<Uniform> uniform;
[vk_binding(1)]
Sampler2DShadow shadow_combined_image_sampler;
[vk_input_attachment_index(0), vk_binding(2)]
SubpassInput<float4> depth_input;
[vk_input_attachment_index(1), vk_binding(3)]
SubpassInput<uint4> geometry_input;

struct FSOutput {
  [vk_location(1)]
  float3 color;
  [vk_location(2)]
  float luminance;
};

[shader("fragment")]
FSOutput fragmentColor(FullScreenVSOutput input) {
  const float luminance_epsilon = 0.0001;

  var output : FSOutput;

  let depth = depth_input.SubpassLoad()[0];
  if (depth == 0.0) {
    output.color = float3(0.0, 0.0, 0.0);
    output.luminance = luminance_epsilon;
    return output;
  }
  let geometry = geometry_input.SubpassLoad();

  float3 position;
  position[directions - 1] = uniform.camera_near / depth;

  for (var direction = 0; direction != directions - 1; direction += 1) {
    position[direction] = input.position[direction];
    position[direction] *= position[directions - 1];
    position[direction] /= uniform.camera_scale[direction];
  }

  position[directions - 1] -= uniform.camera_near;

  for (var _direction = 1; _direction != directions; _direction += 1) {
    let direction = directions - _direction - 1;
    let adj = position[directions - 1];
    let opp = position[direction];

    position[directions - 1] = uniform.camera_cos[direction] * adj -
                               uniform.camera_sin[direction] * opp;
    position[direction] = uniform.camera_sin[direction] * adj +
                          uniform.camera_cos[direction] * opp;
  }

  position += uniform.camera_position;

  let light_view_position = applyAffine(position, uniform.light_affine);
  let light_view_sample_position = light_view_position.xy / 2.0 + 0.5;

  let diffuse = shadow_combined_image_sampler.SampleCmp(
      light_view_sample_position, light_view_position.z, );
  let ambient = bit_cast<float>(geometry[1]);

  var direction_signed = geometry[0];
  let direction_unsigned = direction_signed & ((1 << log_directions) - 1);
  direction_signed >>= log_directions;
  let direction_sign = direction_signed & 1;

  for (var direction = 0; direction != directions; direction += 1) {
    output.color[direction] =
        float(direction_sign ^ (direction != direction_unsigned));
  }
  const let diffuse_strength : float = 0.8;
  const let ambient_strength : float = 0.4;

  output.color *= diffuse_strength * diffuse + ambient_strength * ambient;
  output.luminance = luminance_epsilon + calculateLuminance(output.color);

  return output;
}

}

namespace exposure_average {
[vk_binding(0)]
Texture2D<float> luminance_new;
[vk_binding(1)]
RWStructuredBuffer<float> exposure;

[shader("compute"), numthreads(1, 1, 1)]
void computeExposureAverage() {
  static const float alpha = 0.1;
  let exposure_new = 0.18 / luminance_new.Load(int3(0,0,0));

  exposure[0] = (alpha * exposure_new) + ((1.0 - alpha) * exposure[0]);

}
}

// TODO: improve tonemapping, this is very non physical
namespace tonemap {
[vk_binding(0)]
ConstantBuffer<float> exposure;
[vk_binding(1)]
Texture2D<float3> color_image;

[shader("fragment")]
float3 fragmentTonemap(FullScreenVSOutput input) {
  let sample_position = int3(int2(input.clip_position.xy), 0);
  let color = color_image.Load(sample_position);
  //
  // let luminance_original = calculateLuminance(color);
  // let luminance_scaled = exposure[0] * luminance_original;
  // let luminance_mapped = luminance_scaled / (luminance_scaled + 1.0);
  //
  // let sum = color[0] + color[1] + color[2];
  return (color * exposure) / (color * exposure + 1.0);
}
}

float calculateLuminance(float3 color) {
  return dot(color, float3(0.2126, 0.7152, 0.0722));
}
