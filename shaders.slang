struct Camera {
	vector<float32_t, 4> light_affine[3];
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

struct ColorVSOutput {
	vector<float32_t, 4> position: SV_Position;
	vector<float32_t, 3> light_position;
	nointerpolation vector<float32_t, 3> color;
};

struct VSInput {
	[[vk::location(0)]]
	uint32_t surface;
};

static const uint log_side_len = 0x5;

static const uint dimensions = 3;
static const uint log_dimensions = uint(ceil(log2(float(dimensions))));

static const float max_surrounding_air = float((dimensions << 1) - 1);
static const uint log_surrounding_air = uint(ceil(log2(max_surrounding_air)));

static const uint log_direction_hints = (dimensions - 1) * 2;

static const float diffuse_strength = 5.0 / 8.0;
static const float ambient_strength = 1.0 - diffuse_strength;

[shader("vertex")]
ColorVSOutput vertexColor(VSInput input, uint vertex_id: SV_VertexID) {
	ColorVSOutput output;

	vector<float32_t, 3> position;

	uint surface = input.surface;
	for (uint direction_i = 0; direction_i != dimensions; direction_i += 1) {
		position[dimensions - direction_i - 1] = float(surface & ((1 << log_side_len) - 1));
		surface >>= log_side_len;
	}	
	uint direction = surface & ((1 << log_dimensions) - 1);
	surface >>= log_dimensions;
	uint direction_sign = surface & 1;
	surface >>= 1;
	float ambient = float(surface & ((1 << log_surrounding_air) - 1)) / max_surrounding_air;
	surface >>= log_surrounding_air;
	uint direction_hints = surface & ((1 << log_direction_hints) - 1);

	// camera is light source
	float3 camera_direction = {
		camera.szx * camera.czy,
		camera.szy,
		camera.czx * camera.czy,
	};
	
	float diffuse;
	if (direction_sign == 0) {
		diffuse = camera_direction[direction];
	} else {
		diffuse = -camera_direction[direction];
	}
	uint non_zero_diffuse_contributions = 1;
	for (uint _direction_i = 0; _direction_i != dimensions; _direction_i += 1) {
		uint direction_i = dimensions - 1 - _direction_i;	
		
		if (direction_i != direction) {
			switch(direction_hints & 0b11) {
				case 0b00:
					diffuse += camera_direction[direction_i];
					non_zero_diffuse_contributions += 1;
					break;
				case 0b10:
					diffuse -= camera_direction[direction_i];
					non_zero_diffuse_contributions += 1;
					break;
			}
			direction_hints >>= 2;
		}
	}
	diffuse /= sqrt(float(non_zero_diffuse_contributions));
	diffuse = max(0.0, diffuse);

	position[direction] += float(direction_sign);
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

	for (uint dimension_i = 0; dimension_i != dimensions; dimension_i += 1) {
		output.color[dimension_i] = float(direction_sign ^ (dimension_i != direction));
	}
	output.color *= (ambient * ambient_strength) + (diffuse * diffuse_strength);

	output.light_position = applyAffine(position, camera.light_affine);

	position -= camera.position;

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
vector<float32_t, 4> fragmentColor(ColorVSOutput input) {
	return float4(input.color, 1.0);
}

struct Light {
	vector<float32_t, 4> affine[3];
};
[[vk::binding(0,0)]]
ConstantBuffer<Light> light;

struct ShadowCastVSOutput {
	noperspective vector<float32_t, 3> position: SV_Position;
};

[shader("vertex")]
ShadowCastVSOutput vertexShadowCast(VSInput input, uint vertex_id: SV_VertexID) {
	ShadowCastVSOutput output;

	vector<float32_t, 3> position;

	uint surface = input.surface;
	for (uint _direction_i = 0; _direction_i != dimensions; _direction_i += 1) {
		uint direction_i = dimensions - 1 - _direction_i;
		position[direction_i] = float(surface & ((1 << log_side_len) - 1));
		surface >>= log_side_len;
	}	
	uint direction = surface & ((1 << log_dimensions) - 1);
	surface >>= log_dimensions;
	uint direction_sign = surface & 1;
	surface >>= 1;

	position[direction] += float(direction_sign);

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

	output.position = applyAffine(position, light.affine);
	return output;
}

vector<T, N> applyAffine<T:__BuiltinFloatingPointType, let N : int>(
	vector<T, N> input,
	vector<T, N + 1> affine[N],
) {
	vector<T, N> output;
	for (uint i = 0; i != N; i += 1) {
		output[i] = affine[i][N];
		for (uint j = 0; j != N; j += 1) {
			output[i] += affine[i][j] * input[j];
		}
	}
	return output;
}
