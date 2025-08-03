const std = @import("std");
const Scanner = @import("wayland").Scanner;

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // const optimize = b.standardOptimizeOption(.{});

    const scanner = Scanner.create(b, .{});
    const wayland = b.createModule(.{ .root_source_file = scanner.result });
    scanner.addSystemProtocol("stable/xdg-shell/xdg-shell.xml");
    scanner.generate("wl_compositor", 6);
    scanner.generate("wl_seat", 9);
    scanner.generate("xdg_wm_base", 7);

    const vulkan = b.dependency("vulkan", .{
        .registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml"),
    }).module("vulkan-zig");

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("main.zig"),
        .target = target,
        .optimize = .Debug,
        .omit_frame_pointer = false,
    });
    exe_mod.addImport("wayland", wayland);
    exe_mod.addImport("vulkan", vulkan);

    const ShaderLanguage = enum {
        zig,
        slang,
    };
    const shader_language: ShaderLanguage = .slang;

    switch (shader_language) {
        .zig => {
            const spirv_target = b.resolveTargetQuery(.{
                .cpu_arch = .spirv64,
                .os_tag = .vulkan,
                .cpu_model = .{ .explicit = &std.Target.spirv.cpu.vulkan_v1_2 },
                .cpu_features_add = std.Target.spirv.featureSet(&.{
                    .int64,
                    .v1_6,
                }),
                .ofmt = .spirv,
            });
            exe_mod.addAnonymousImport("shaders", .{
                .root_source_file = b.addObject(.{
                    .name = "shaders",
                    .root_source_file = b.path("shaders.zig"),
                    .target = spirv_target,
                    .use_llvm = false,
                }).getEmittedBin(),
            });
        },
        .slang => {
            const cmd = b.addSystemCommand(&.{
                "slangc",
                "-profile",
                "glsl_450+SPV_KHR_vulkan_memory_model",
                "-target",
                "spirv",
                "-o",
            });
            const shaders = cmd.addOutputFileArg("shaders.spv");
            cmd.addFileArg(b.path("shaders.slang"));
            exe_mod.addAnonymousImport("shaders", .{ .root_source_file = shaders });
        },
    }

    const exe = b.addExecutable(.{
        .name = "vk-wl",
        .root_module = exe_mod,
        .use_llvm = true,
        .use_lld = false,
    });
    exe.linkLibC();
    exe.linkSystemLibrary("wayland-client");

    const run = b.addRunArtifact(exe);

    const run_step = b.step("run", "run vk-wl");
    run_step.dependOn(&run.step);

    b.installArtifact(exe);
}
