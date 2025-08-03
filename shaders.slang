struct Camera {
	float x;
	float y;
	float z;

	float z_near;

	float cos_z_x;
	float sin_z_x;
	float cos_z_y;
	float sin_z_y;
};
[[vk::binding(0,0)]]
ConstantBuffer<Camera> camera;

struct VSOutput {
	float4 position: SV_Position;
	float3 color;
};

static const float2 positions[3] = {
	float2( 1.0,  0.5),
	float2( 0.0, -1.0),
	float2(-1.0,  1.0),
};

static const float3 colors[3] = {
	float3(1.0, 1.0, 0.0),
	float3(0.0, 1.0, 0.0),
	float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VSOutput vertexMain(uint id: SV_VertexID) {
	VSOutput output;
	output.position = float4(
		positions[id].x - camera.x, 
		positions[id].y - camera.y, 
		0.0, 
		1.0,
	);
	output.color = colors[id];
	return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) {
	return float4(input.color, 1.0);
}
