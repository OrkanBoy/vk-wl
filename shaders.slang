struct Camera {
	float3 position;

	float z_near;

	float czx;
	float szx;
	float czy;
	float szy;

	float2 scale;
};
[[vk::binding(0,0)]]
ConstantBuffer<Camera> camera;

struct VSOutput {
	float4 position: SV_Position;
	float3 color;
};

static const float3 positions[3] = {
	float3( 6.0,  2.0, 2.0),
	float3( 0.0, -2.0, 1.0),
	float3(-6.0,  2.0, 2.0),
};

static const float3 colors[3] = {
	float3(1.0, 1.0, 0.0),
	float3(0.0, 1.0, 0.0),
	float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VSOutput vertexMain(uint id: SV_VertexID) {
	VSOutput output;
	float3 position = positions[id] - camera.position;

	float x, y, z;

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

	output.position = float4(
		position.x,
		position.y,
		0.0,
		position.z,
	);
	output.color = colors[id];
	return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) {
	return float4(input.color, 1.0);
}
