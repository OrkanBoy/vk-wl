const std = @import("std");
const Scanner = @import("wayland").Scanner;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const scanner = Scanner.create(b, .{});
    const wayland = b.createModule(.{ .root_source_file = scanner.result });
    scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");
    scanner.generate("wl_compositor", 1);
    scanner.generate("wl_shm", 1);
    scanner.generate("xdg_wm_base", 1);

    const vulkan = b.dependency("vulkan", .{
        .registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml"),
    }).module("vulkan-zig");

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = optimize,
    });
    exe_mod.addImport("wayland", wayland);
    exe_mod.addImport("vulkan", vulkan);

    const spirv_target = b.resolveTargetQuery(.{
        .cpu_arch = .spirv,
        .os_tag = .vulkan,
        .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
        .cpu_features_add = std.Target.spirv.featureSet(&.{.int64}),
        .ofmt = .spirv,
    });

    exe_mod.addAnonymousImport("vertex_spv", .{
        .root_source_file = b.addObject(.{
            .name = "vertex_spv",
            .root_source_file = b.path("vertex.zig"),
            .target = spirv_target,
            .use_llvm = false,
        }).getEmittedBin(),
    });

    exe_mod.addAnonymousImport("fragment_spv", .{
        .root_source_file = b.addObject(.{
            .name = "fragment_spv",
            .root_source_file = b.path("fragment.zig"),
            .target = spirv_target,
            .use_llvm = false,
        }).getEmittedBin(),
    });

    const exe = b.addExecutable(.{
        .name = "vk-wl",
        .root_module = exe_mod,
        .use_llvm = false,
        .use_lld = false,
    });
    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");

    const run = b.addRunArtifact(exe);
    run.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "run vk-wl");
    run_step.dependOn(&run.step);
}
