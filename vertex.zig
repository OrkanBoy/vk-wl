const std = @import("std");
const gpu = std.gpu;

extern var v_color: @Vector(3, f32) addrspace(.output);

export fn main() callconv(.spirv_vertex) void {
    const positions = [_]@Vector(2, f32){
        .{ 1.0, 1.0 },
        .{ 0.0, -1.0 },
        .{ -1.0, 1.0 },
    };
    const color = [_]@Vector(3, f32){
        .{ 1.0, 0.0, 0.0 },
        .{ 0.0, 1.0, 0.0 },
        .{ 0.0, 0.0, 1.0 },
    };
    gpu.location(&v_color, 0);

    v_color = color[gpu.vertex_index];
    gpu.position_out.* = .{ positions[gpu.vertex_index][0], positions[gpu.vertex_index][1], 0.0, 1.0 };
}
