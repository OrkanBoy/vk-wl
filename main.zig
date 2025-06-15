const std = @import("std");
const wayland = @import("wayland");
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;
const vk = @import("vulkan");

// const vk = @import("vulkan");

const RegistryListenerData = struct {
    compositor: ?*wl.Compositor,
    wm_base: ?*xdg.WmBase,
};

pub fn main() !void {
    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();
    var registry_listener_data = RegistryListenerData{
        .compositor = null,
        .wm_base = null,
    };
    registry.setListener(*RegistryListenerData, registryListener, &registry_listener_data);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const compositor = registry_listener_data.compositor orelse return error.NoWlCompositor;
    const wm_base = registry_listener_data.wm_base orelse return error.NoXdgWmBase;

    const wl_surface = try compositor.createSurface();
    defer wl_surface.destroy();

    const xdg_surface = try wm_base.getXdgSurface(wl_surface);
    defer xdg_surface.destroy();
    xdg_surface.setListener(*const void, xdgSurfaceListener, &{});

    const toplevel = try xdg_surface.getToplevel();
    defer toplevel.destroy();

    var toplevel_listener_data: ?xdg.Toplevel.Event = null;
    toplevel.setListener(*?xdg.Toplevel.Event, xdgToplevelListener, &toplevel_listener_data);

    wl_surface.commit();

    var lib = try std.DynLib.open("libvulkan.so");
    defer lib.close();
    const vkGetInstanceProcAddr = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr").?;

    const extenisons = [_][*:0]const u8{
        "VK_KHR_surface",
        "VK_KHR_wayland_surface",
        "VK_EXT_debug_utils",
    };

    const layers = [_][*:0]const u8{
        "VK_LAYER_KHRONOS_validation",
    };

    const debug_info = vk.DebugUtilsMessengerCreateInfoEXT{
        .message_severity = .{
            .verbose_bit_ext = true,
            .info_bit_ext = true,
            .warning_bit_ext = true,
            .error_bit_ext = true,
        },
        .message_type = .{
            .general_bit_ext = true,
            .validation_bit_ext = true,
            .performance_bit_ext = true,
            .device_address_binding_bit_ext = false,
        },
        .pfn_user_callback = &debugCallback,
    };

    const report_flags = [_][*:0]const u8{ "error", "warn", "info", "perf", "verbose" };

    const layer_settings = [_]vk.LayerSettingEXT{
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_core",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &@as(vk.Bool32, vk.TRUE),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_sync",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &@as(vk.Bool32, vk.TRUE),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "validate_best_practices",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &@as(vk.Bool32, vk.TRUE),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "thread_safety",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &@as(vk.Bool32, vk.TRUE),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "debug_action",
            .type = .string_ext,
            .value_count = 1,
            .p_values = @ptrCast(&"VK_DBG_LAYER_ACTION_LOG_MSG"),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "report_flags",
            .type = .string_ext,
            .value_count = report_flags.len,
            .p_values = @ptrCast(&report_flags),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "enable_message_limit",
            .type = .bool32_ext,
            .value_count = 1,
            .p_values = &@as(vk.Bool32, vk.TRUE),
        },
        vk.LayerSettingEXT{
            .p_layer_name = "VK_LAYER_KHRONOS_validation",
            .p_setting_name = "duplicate_message_limit",
            .type = .uint32_ext,
            .value_count = 1,
            .p_values = &@as(u32, 0),
        },
    };

    const layer_settings_info = vk.LayerSettingsCreateInfoEXT{
        .setting_count = layer_settings.len,
        .p_settings = &layer_settings,
        .p_next = &debug_info,
    };

    const vkb = vk.BaseWrapper.load(vkGetInstanceProcAddr);
    const instance = try vkb.createInstance(&.{
        .p_application_info = &vk.ApplicationInfo{
            .api_version = @bitCast(vk.API_VERSION_1_4),
            .application_version = 0,
            .engine_version = 0,
        },
        .enabled_extension_count = extenisons.len,
        .pp_enabled_extension_names = &extenisons,
        .enabled_layer_count = layers.len,
        .pp_enabled_layer_names = &layers,
        .p_next = &layer_settings_info,
    }, null);
    const vki = vk.InstanceWrapper.load(instance, vkGetInstanceProcAddr);
    defer vki.destroyInstance(instance, null);

    const debug_messenger = try vki.createDebugUtilsMessengerEXT(instance, &debug_info, null);
    defer vki.destroyDebugUtilsMessengerEXT(instance, debug_messenger, null);

    var physical_devices_count: u32 = 1;
    var physical_device: vk.PhysicalDevice = undefined;
    _ = try vki.enumeratePhysicalDevices(instance, &physical_devices_count, @ptrCast(&physical_device));
    _ = vki.getPhysicalDeviceFeatures(physical_device);

    var features_1_3 = vk.PhysicalDeviceVulkan13Features{
        .dynamic_rendering = vk.TRUE,
        .synchronization_2 = vk.TRUE,
    };
    var features_1_2 = vk.PhysicalDeviceVulkan12Features{
        .buffer_device_address = vk.TRUE,
        .descriptor_indexing = vk.TRUE,
        .shader_int_8 = vk.TRUE,
        .p_next = @ptrCast(&features_1_3),
    };

    const device_extensions = [_][*:0]const u8{"VK_KHR_swapchain"};

    const device = try vki.createDevice(
        physical_device,
        &vk.DeviceCreateInfo{
            .queue_create_info_count = 1,
            .p_queue_create_infos = @ptrCast(&vk.DeviceQueueCreateInfo{
                .queue_family_index = 0,
                .queue_count = 1,
                .p_queue_priorities = &[_]f32{1.0},
            }),
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = &device_extensions,
            .p_enabled_features = &vk.PhysicalDeviceFeatures{
                .shader_int_64 = vk.TRUE,
                .shader_int_16 = vk.TRUE,
            },
            .p_next = @ptrCast(&features_1_2),
        },
        null,
    );
    const vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
    defer vkd.destroyDevice(device, null);

    const queue = vkd.getDeviceQueue2(device, &vk.DeviceQueueInfo2{
        .queue_family_index = 0,
        .queue_index = 0,
    });

    const vk_surface = try vki.createWaylandSurfaceKHR(
        instance,
        &.{
            .display = @ptrCast(display),
            .surface = @ptrCast(wl_surface),
        },
        null,
    );
    defer vki.destroySurfaceKHR(instance, vk_surface, null);

    var surface_formats_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(physical_device, vk_surface, &surface_formats_count, null);
    var surface_format: vk.SurfaceFormatKHR = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(physical_device, vk_surface, &surface_formats_count, @ptrCast(&surface_format));

    var swapchain: vk.SwapchainKHR = .null_handle;
    var swapchain_extent: vk.Extent2D = .{
        .width = ~@as(u32, 0),
        .height = ~@as(u32, 0),
    };
    var swapchain_images: [4]vk.Image = undefined;
    var swapchain_images_count: u32 = undefined;
    var swapchain_image_views: [4]vk.ImageView = undefined;

    const pipeline_layout = try vkd.createPipelineLayout(device, &vk.PipelineLayoutCreateInfo{}, null);
    defer vkd.destroyPipelineLayout(device, pipeline_layout, null);

    var pipeline: vk.Pipeline = undefined;
    const vertex_spv align(@alignOf(u32)) = @embedFile("vertex_spv").*;
    const fragment_spv align(@alignOf(u32)) = @embedFile("fragment_spv").*;

    const vertex_shader = try vkd.createShaderModule(device, &vk.ShaderModuleCreateInfo{
        .p_code = @ptrCast(&vertex_spv),
        .code_size = vertex_spv.len,
    }, null);
    defer vkd.destroyShaderModule(device, vertex_shader, null);

    const fragment_shader = try vkd.createShaderModule(device, &vk.ShaderModuleCreateInfo{
        .p_code = @ptrCast(&fragment_spv),
        .code_size = fragment_spv.len,
    }, null);
    defer vkd.destroyShaderModule(device, fragment_shader, null);

    _ = try vkd.createGraphicsPipelines(
        device,
        .null_handle,
        1,
        @ptrCast(&vk.GraphicsPipelineCreateInfo{
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = undefined,
            .layout = pipeline_layout,
            .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                .topology = .triangle_list,
                .primitive_restart_enable = vk.FALSE,
            },
            .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
                .attachment_count = 1,
                .blend_constants = [_]f32{
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                },
                .logic_op = .copy,
                .logic_op_enable = vk.FALSE,
                .p_attachments = @ptrCast(&vk.PipelineColorBlendAttachmentState{
                    .blend_enable = vk.FALSE,
                    .src_color_blend_factor = .one,
                    .dst_color_blend_factor = .zero,
                    .color_blend_op = .add,
                    .src_alpha_blend_factor = .one,
                    .dst_alpha_blend_factor = .zero,
                    .alpha_blend_op = .add,
                    .color_write_mask = .{
                        .r_bit = true,
                        .b_bit = true,
                        .g_bit = true,
                        .a_bit = true,
                    },
                }),
            },
            .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{},
            .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
                .cull_mode = .{ .back_bit = true },
                .front_face = .counter_clockwise,
                .polygon_mode = .fill,
                .depth_bias_enable = vk.FALSE,
                .depth_clamp_enable = vk.FALSE,
                .depth_bias_clamp = 0.0,
                .depth_bias_constant_factor = 0.0,
                .depth_bias_slope_factor = 0.0,
                .line_width = 1.0,
                .rasterizer_discard_enable = vk.FALSE,
            },
            .p_multisample_state = &vk.PipelineMultisampleStateCreateInfo{
                .rasterization_samples = .{ .@"1_bit" = true },
                .sample_shading_enable = vk.FALSE,
                .min_sample_shading = 1,
                .alpha_to_coverage_enable = vk.FALSE,
                .alpha_to_one_enable = vk.FALSE,
            },
            .p_viewport_state = &vk.PipelineViewportStateCreateInfo{
                .scissor_count = 1,
                .p_scissors = null,
                .viewport_count = 1,
                .p_viewports = null,
            },
            .p_dynamic_state = &vk.PipelineDynamicStateCreateInfo{
                .dynamic_state_count = 2,
                .p_dynamic_states = @ptrCast(&[_]vk.DynamicState{ .viewport, .scissor }),
            },
            .stage_count = 2,
            .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
                vk.PipelineShaderStageCreateInfo{
                    .p_name = "main",
                    .stage = .{ .vertex_bit = true },
                    .module = vertex_shader,
                },
                vk.PipelineShaderStageCreateInfo{
                    .p_name = "main",
                    .stage = .{ .fragment_bit = true },
                    .module = fragment_shader,
                },
            },
            .render_pass = .null_handle,
            .subpass = undefined,
            .p_next = &vk.PipelineRenderingCreateInfo{
                .color_attachment_count = 1,
                .p_color_attachment_formats = @ptrCast(&surface_format.format),
                .depth_attachment_format = .undefined,
                .stencil_attachment_format = .undefined,
                .view_mask = 0,
            },
        }),
        null,
        @ptrCast(&pipeline),
    );
    defer vkd.destroyPipeline(device, pipeline, null);

    defer vkd.destroySwapchainKHR(device, swapchain, null);
    defer for (0..swapchain_images_count) |i| {
        vkd.destroyImageView(device, swapchain_image_views[i], null);
    };

    const surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, vk_surface);

    var command_pools: [2]vk.CommandPool = undefined;
    for (0..2) |i| {
        command_pools[i] = try vkd.createCommandPool(
            device,
            &.{ .queue_family_index = 0 },
            null,
        );
    }

    defer for (0..2) |i| {
        vkd.destroyCommandPool(device, command_pools[i], null);
    };

    var command_buffers: [2]vk.CommandBuffer = undefined;
    for (0..2) |i| {
        _ = try vkd.allocateCommandBuffers(
            device,
            &.{
                .level = .primary,
                .command_buffer_count = 1,
                .command_pool = command_pools[i],
            },
            @ptrCast(&command_buffers[i]),
        );
    }
    var swapchain_image_semaphores: [2]vk.Semaphore = undefined;
    var command_buffer_fences: [2]vk.Fence = undefined;

    var render_end_semaphores: [4]vk.Semaphore = [_]vk.Semaphore{.null_handle} ** 4;
    defer for (0..swapchain_images_count) |i| {
        vkd.destroySemaphore(device, render_end_semaphores[i], null);
    };

    for (0..2) |i| {
        swapchain_image_semaphores[i] = try vkd.createSemaphore(device, &.{}, null);
        command_buffer_fences[i] = try vkd.createFence(
            device,
            &.{ .flags = .{ .signaled_bit = true } },
            null,
        );
    }
    defer for (0..2) |i| {
        vkd.destroySemaphore(device, swapchain_image_semaphores[i], null);
        vkd.destroyFence(device, command_buffer_fences[i], null);
    };

    var frame: u1 = 0;
    while (true) {
        if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;
        if (toplevel_listener_data != null) {
            switch (toplevel_listener_data.?) {
                .close => {
                    try vkd.deviceWaitIdle(device);
                    return;
                },
                .configure => |configure| if (swapchain_extent.width != @as(u32, @bitCast(configure.width)) or
                    swapchain_extent.height != @as(u32, @bitCast(configure.height)))
                {
                    if (swapchain != .null_handle) {
                        try vkd.deviceWaitIdle(device);
                        for (0..swapchain_images_count) |i| {
                            vkd.destroyImageView(device, swapchain_image_views[i], null);
                        }
                        vkd.destroySwapchainKHR(device, swapchain, null);
                    }
                    swapchain_extent = .{
                        .width = @bitCast(configure.width),
                        .height = @bitCast(configure.height),
                    };

                    swapchain = try vkd.createSwapchainKHR(
                        device,
                        &.{
                            .surface = vk_surface,
                            .min_image_count = surface_capabilities.min_image_count,
                            .image_format = surface_format.format,
                            .image_color_space = surface_format.color_space,
                            .image_extent = vk.Extent2D{
                                .width = @bitCast(configure.width),
                                .height = @bitCast(configure.height),
                            },
                            .image_array_layers = 1,
                            .image_usage = vk.ImageUsageFlags{
                                .color_attachment_bit = true,
                            },
                            .image_sharing_mode = vk.SharingMode.exclusive,
                            .queue_family_index_count = 1,
                            .p_queue_family_indices = &[_]u32{0},
                            .composite_alpha = .{
                                .opaque_bit_khr = true,
                            },
                            .present_mode = .fifo_khr,
                            .clipped = vk.TRUE,
                            .pre_transform = surface_capabilities.current_transform,
                        },
                        null,
                    );

                    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, null);
                    _ = try vkd.getSwapchainImagesKHR(device, swapchain, &swapchain_images_count, &swapchain_images);
                    for (0..swapchain_images_count) |i| {
                        swapchain_image_views[i] = try vkd.createImageView(
                            device,
                            &.{
                                .image = swapchain_images[i],
                                .components = .{
                                    .a = .identity,
                                    .r = .identity,
                                    .g = .identity,
                                    .b = .identity,
                                },
                                .subresource_range = .{
                                    .aspect_mask = .{
                                        .color_bit = true,
                                    },
                                    .base_mip_level = 0,
                                    .level_count = 1,
                                    .base_array_layer = 0,
                                    .layer_count = 1,
                                },
                                .format = surface_format.format,
                                .view_type = .@"2d",
                            },
                            null,
                        );
                        if (render_end_semaphores[i] == .null_handle) {
                            render_end_semaphores[i] = try vkd.createSemaphore(device, &.{}, null);
                        }
                    }

                    wl_surface.commit();
                },
            }
            toplevel_listener_data = null;
        }

        _ = try vkd.waitForFences(device, 1, @ptrCast(&command_buffer_fences[frame]), vk.TRUE, std.math.maxInt(u64));
        _ = try vkd.resetFences(device, 1, @ptrCast(&command_buffer_fences[frame]));

        const swapchain_image_index = (try vkd.acquireNextImageKHR(device, swapchain, std.math.maxInt(u64), swapchain_image_semaphores[frame], .null_handle)).image_index;

        try vkd.resetCommandPool(device, command_pools[frame], .{});
        {
            try vkd.beginCommandBuffer(command_buffers[frame], &.{ .flags = .{ .one_time_submit_bit = true } });

            vkd.cmdPipelineBarrier2(
                command_buffers[frame],
                &vk.DependencyInfo{
                    .image_memory_barrier_count = 1,
                    .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2{
                        .src_queue_family_index = 0,
                        .dst_queue_family_index = 0,
                        .subresource_range = .{
                            .aspect_mask = .{ .color_bit = true },
                            .base_array_layer = 0,
                            .base_mip_level = 0,
                            .layer_count = vk.REMAINING_ARRAY_LAYERS,
                            .level_count = vk.REMAINING_MIP_LEVELS,
                        },
                        .old_layout = .undefined,
                        .new_layout = .color_attachment_optimal,
                        .image = swapchain_images[swapchain_image_index],
                        .src_stage_mask = .{
                            .color_attachment_output_bit = true,
                        },
                        .src_access_mask = .{},
                        .dst_stage_mask = .{ .color_attachment_output_bit = true },
                        .dst_access_mask = .{
                            .color_attachment_read_bit = true,
                            .color_attachment_write_bit = true,
                        },
                    }),
                },
            );

            {
                vkd.cmdBeginRendering(command_buffers[frame], &vk.RenderingInfo{
                    .view_mask = 0,
                    .render_area = .{
                        .offset = .{
                            .x = 0,
                            .y = 0,
                        },
                        .extent = swapchain_extent,
                    },
                    .layer_count = 1,
                    .color_attachment_count = 1,
                    .p_color_attachments = @ptrCast(&vk.RenderingAttachmentInfo{
                        .image_view = swapchain_image_views[swapchain_image_index],
                        .clear_value = vk.ClearValue{ .color = .{ .float_32 = [4]f32{
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        } } },
                        .image_layout = .rendering_local_read,
                        .resolve_mode = .{},
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_image_layout = .undefined,
                    }),
                });

                vkd.cmdSetViewport(command_buffers[frame], 0, 1, @ptrCast(&vk.Viewport{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @floatFromInt(swapchain_extent.width),
                    .height = @floatFromInt(swapchain_extent.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                }));

                vkd.cmdSetScissor(command_buffers[frame], 0, 1, @ptrCast(&vk.Rect2D{
                    .extent = swapchain_extent,
                    .offset = .{ .x = 0, .y = 0 },
                }));

                vkd.cmdBindPipeline(command_buffers[frame], .graphics, pipeline);
                vkd.cmdDraw(command_buffers[frame], 3, 1, 0, 0);

                vkd.cmdEndRendering(command_buffers[frame]);
            }

            vkd.cmdPipelineBarrier2(
                command_buffers[frame],
                &vk.DependencyInfo{
                    .image_memory_barrier_count = 1,
                    .p_image_memory_barriers = @ptrCast(&vk.ImageMemoryBarrier2{
                        .src_queue_family_index = 0,
                        .dst_queue_family_index = 0,
                        .subresource_range = .{
                            .aspect_mask = .{ .color_bit = true },
                            .base_array_layer = 0,
                            .base_mip_level = 0,
                            .layer_count = vk.REMAINING_ARRAY_LAYERS,
                            .level_count = vk.REMAINING_MIP_LEVELS,
                        },
                        .old_layout = .color_attachment_optimal,
                        .new_layout = .present_src_khr,
                        .image = swapchain_images[swapchain_image_index],
                        .src_stage_mask = .{
                            .color_attachment_output_bit = true,
                        },
                        .src_access_mask = .{
                            .color_attachment_read_bit = true,
                            .color_attachment_write_bit = true,
                        },
                        .dst_stage_mask = .{
                            .color_attachment_output_bit = true,
                        },
                        .dst_access_mask = .{},
                    }),
                },
            );

            try vkd.endCommandBuffer(command_buffers[frame]);
        }

        _ = try vkd.queueSubmit2(queue, 1, @ptrCast(&vk.SubmitInfo2{
            .wait_semaphore_info_count = 1,
            .p_wait_semaphore_infos = @ptrCast(&vk.SemaphoreSubmitInfo{
                .value = undefined,
                .device_index = 0,
                .semaphore = swapchain_image_semaphores[frame],
                .stage_mask = .{ .color_attachment_output_bit = true },
            }),
            .signal_semaphore_info_count = 1,
            .p_signal_semaphore_infos = @ptrCast(&vk.SemaphoreSubmitInfo{
                .value = undefined,
                .device_index = 0,
                .semaphore = render_end_semaphores[swapchain_image_index],
                .stage_mask = .{ .all_graphics_bit = true },
            }),
            .command_buffer_info_count = 1,
            .p_command_buffer_infos = @ptrCast(&vk.CommandBufferSubmitInfo{
                .command_buffer = command_buffers[frame],
                .device_mask = 0,
            }),
        }), command_buffer_fences[frame]);

        _ = try vkd.queuePresentKHR(queue, @ptrCast(&vk.PresentInfoKHR{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&render_end_semaphores[swapchain_image_index]),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&swapchain),
            .p_image_indices = @ptrCast(&swapchain_image_index),
        }));

        frame = ~frame;
    }
}

fn registryListener(registry: *wl.Registry, event: wl.Registry.Event, data: *RegistryListenerData) void {
    switch (event) {
        .global => |global| {
            if (std.mem.orderZ(u8, global.interface, "wl_compositor") == .eq) {
                data.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (std.mem.orderZ(u8, global.interface, "xdg_wm_base") == .eq) {
                data.wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
            }
        },
        .global_remove => {},
    }
}

fn xdgSurfaceListener(xdg_surface: *xdg.Surface, event: xdg.Surface.Event, _: *const void) void {
    switch (event) {
        .configure => |configure| {
            xdg_surface.ackConfigure(configure.serial);
        },
    }
}

fn xdgToplevelListener(_: *xdg.Toplevel, event: xdg.Toplevel.Event, data: *?xdg.Toplevel.Event) void {
    data.* = event;
}

fn debugCallback(
    message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
    _: vk.DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
    _: ?*anyopaque,
) callconv(.C) vk.Bool32 {
    const data = p_callback_data.?;

    var color_code: u8 = undefined;
    if (message_severity.verbose_bit_ext) {
        color_code = '0';
    } else if (message_severity.info_bit_ext) {
        color_code = '2';
    } else if (message_severity.warning_bit_ext) {
        color_code = '3';
    } else {
        color_code = '1';
    }

    std.debug.print(
        "\x1b[9{c};1;4m{s}\x1b[m\x1b[9{c}m\n{s}\n\x1b[m\n",
        .{
            color_code,
            data.p_message_id_name.?,
            color_code,
            data.p_message.?,
        },
    );

    return vk.FALSE;
}
