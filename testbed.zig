const std = @import("std");

pub fn main() !void {
    const dimensions = 0x2;

    const log_side_len = 0x5;
    const side_len = 1 << log_side_len;

    const log_volume_len = log_side_len * dimensions;
    const volume_len = 1 << log_volume_len;

    const morton_mask: usize = comptime blk: {
        var value: usize = 0;
        for (0..log_side_len) |_| {
            value <<= dimensions;
            value |= 1;
        }
        break :blk value;
    };

    const Voxel = enum {
        Path,
        Empty,
        Hit,
    };
    var voxels: [volume_len]Voxel = [_]Voxel{.Empty} ** volume_len;

    const origin: [dimensions]f32 = [_]f32{
        18.7,
        12.8,
    };
    var indices: [dimensions]usize = undefined;
    var combined_indices: usize = 0;

    const direction: [dimensions]f32 = [_]f32{
        -0.9,
        0.7,
    };

    var planes: [dimensions]f32 = undefined;
    for (0..dimensions) |dimension_i| {
        planes[dimension_i] =
            if (direction[dimension_i] > 0.0)
                @ceil(origin[dimension_i])
            else
                @floor(origin[dimension_i]);

        indices[dimension_i] = @intFromFloat(origin[dimension_i]);
        combined_indices |= pdep(usize, indices[dimension_i], morton_mask << @truncate(dimension_i));
    }

    while (true) {
        var t_min: f32 = std.math.inf(f32);
        var dimension_i_min: usize = undefined;
        for (0..dimensions) |dimension_i| {
            const t = (planes[dimension_i] - origin[dimension_i]) / direction[dimension_i];
            if (t < t_min) {
                t_min = t;
                dimension_i_min = dimension_i;
            }
        }
        if (direction[dimension_i_min] > 0.0) {
            planes[dimension_i_min] += 1.0;
            if (indices[dimension_i_min] == side_len - 1) break;
            indices[dimension_i_min] += 1;
        } else {
            planes[dimension_i_min] -= 1.0;
            if (indices[dimension_i_min] == 0) break;
            indices[dimension_i_min] -= 1;
        }
        const mask = morton_mask << @truncate(dimension_i_min);

        combined_indices &= ~mask;
        combined_indices |= pdep(usize, indices[dimension_i_min], mask);
        voxels[combined_indices] = Voxel.Path;
    }
    voxels[combined_indices] = Voxel.Hit;
    var buffer: [volume_len + side_len]u8 = undefined;
    var buffer_i: usize = 0;
    for (0..side_len) |y| {
        for (0..side_len) |x| {
            const voxel_i: usize = pdep(usize, y, morton_mask << 1) | pdep(usize, x, morton_mask);
            buffer[buffer_i] = switch (voxels[voxel_i]) {
                Voxel.Empty => ' ',
                Voxel.Path => '~',
                Voxel.Hit => '#',
            };
            buffer_i += 1;
        }
        buffer[buffer_i] = '\n';
        buffer_i += 1;
    }
    var stdout = std.io.getStdOut();
    try stdout.writeAll(&buffer);
}
inline fn pdep(comptime T: type, src: T, mask: T) T {
    if (T != u64 and T != u32 and T != usize) {
        @compileError("pdep only implemented for u32 and u64");
    }
    return asm volatile ("pdep %[mask], %[src], %[dst]"
        // Map to actual registers:
        : [dst] "=r" (-> T),
        : [src] "r" (src),
          [mask] "r" (mask),
    );
}
