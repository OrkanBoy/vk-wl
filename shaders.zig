const std = @import("std");
const gpu = std.gpu;

const vertex = struct {
    extern var v_color: @Vector(3, f32) addrspace(.output);
    const Camera = extern struct {
        x: f32,
        y: f32,
        z: f32,

        z_near: f32,

        cos_x_z: f32,
        sin_x_z: f32,
        cos_z_y: f32,
        sin_z_y: f32,
    };
    extern var u_camera: Camera addrspace(.uniform);

    export fn vertexMain() callconv(.spirv_vertex) void {
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
        gpu.binding(&u_camera, 0, 0);

        v_color = color[gpu.vertex_index];
        gpu.position_out.* = .{ positions[gpu.vertex_index][0], positions[gpu.vertex_index][1], 0.0, 1.0 };
    }
};

const fragment = struct {
    extern const v_color: @Vector(3, f32) addrspace(.input);
    extern var f_color: @Vector(4, f32) addrspace(.output);

    export fn fragmentMain() callconv(.spirv_fragment) void {
        gpu.location(&v_color, 0);
        gpu.location(&f_color, 0);

        f_color = .{ v_color[0], v_color[1], v_color[2], 1.0 };
    }
};

comptime {
    _ = vertex.vertexMain;
    _ = fragment.fragmentMain;
}
