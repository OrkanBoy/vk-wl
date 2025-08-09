struct Camera {
	vector<float32_t, 3> position;

	float32_t z_near;

	float32_t czx;
	float32_t szx;
	float32_t czy;
	float32_t szy;

	vector<float32_t, 2> scale;
};

[[vk::binding(0,0)]]
ConstantBuffer<Camera> camera;

struct VSOutput {
	vector<float32_t, 4> position: SV_Position;
	nointerpolation vector<float32_t, 3> color;
};

static const vector<int8_t, 3> positions[] = {
	{0,0,0},
	{0,1,0},
	{1,0,0},
	{1,1,0},
};

static const vector<float32_t, 3> colors[] = {
	{1.0, 0.0, 0.0},
	{0.0, 1.0, 0.0},
	{0.0, 0.0, 1.0},
};

struct VSInput {
	[[vk::location(0)]]
	uint32_t surface;
};

static const uint log_side_len = 0x5;
static const uint dimensions = 3;
static const uint log_dimensions_len = uint(ceil(log(float(dimensions))));

[shader("vertex")]
VSOutput vertexMain(VSInput input, uint32_t vertex_id: SV_VertexID) {
	VSOutput output;

	vector<float32_t, 3> position;

	uint surface = input.surface;
	uint direction_sign = surface & 1;
	surface >>= 1;
	uint direction = surface & ((1 << log_dimensions_len) - 1);
	surface >>= log_dimensions_len;
	for (int dimension_i = dimensions - 1; dimension_i != -1; dimension_i -= 1) {
		position[dimension_i] = float(surface & ((1 << log_side_len) - 1));
		surface >>= log_side_len;
	}
	position[direction] += direction_sign;

	if (direction_sign == 1) {
		switch (vertex_id) {
			case 1:
				vertex_id = 2;
				break;
			case 2: 
				vertex_id = 1;
		}
	}
	position[(direction + 2) % dimensions] += float((vertex_id & 0b01) >> 0);
	position[(direction + 1) % dimensions] += float((vertex_id & 0b10) >> 1);

	position -= camera.position;

	for (uint dimension_i = 0; dimension_i != dimensions; dimension_i += 1) {
		output.color[dimension_i] = float(direction_sign ^ (dimension_i != direction));
	}

	float32_t x, y, z;

	z = position.z;
	x = position.x;
	position.z = z * camera.czx + x * camera.szx;
	position.x = x * camera.czx - z * camera.szx;

	z = position.z;
	y = position.y;
	position.z = z * camera.czy + y * camera.szy;
	position.y = y * camera.czy - z * camera.szy;

	position.z += camera.z_near;

	position.x *= camera.scale.x;
	position.y *= camera.scale.y;

	output.position = {
		position.x,
		position.y,
		camera.z_near,
		position.z,
	};
	return output;
}

[shader("fragment")]
vector<float32_t, 4> fragmentMain(VSOutput input) {
	return float4(input.color, 1.0);
}
