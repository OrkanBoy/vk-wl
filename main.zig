const std = @import("std");
const wayland = @import("wayland");

const wl = wayland.client.wl;
const xdg = wayland.client.xdg;
const zwp = wayland.client.zwp;

const vk = @import("vulkan");

const tau = std.math.tau;
const print = std.debug.print;
const nanoTimestamp = std.time.nanoTimestamp;
const log2_int_ceil = std.math.log2_int_ceil;

const RegistryListenerData = struct {
    wl_compositor: *wl.Compositor,
    xdg_wm_base: *xdg.WmBase,
    wl_seat: *wl.Seat,
    wl_output: *wl.Output,
    zwp_relative_pointer_manager_v1: *zwp.RelativePointerManagerV1,
    zwp_pointer_constraints_v1: *zwp.PointerConstraintsV1,
};
const Surface = packed struct {
    combined_indices: CombinedIndices,
    direction_unsigned: Direction,
    direction_sign: u1,
    ambients: Ambients,
};
const CombinedIndices = UnsignedInt(directions * log_side_len);
const Direction = UnsignedInt(log2_int_ceil(usize, directions));
// array of ambient values for entire surface
// each ambient value
const Ambients = UnsignedInt((directions - 1) * (1 << (directions - 1)));

const LogUsize = std.math.Log2Int(usize);

const clamp = std.math.clamp;

const rotate_factor = 1 << 0x10;
const range_rotate_factor = 1 << 0x12;

const rotate_unit: i32 = @intFromFloat(tau * rotate_factor);
const range_rotate_unit: i32 = @intFromFloat(tau * range_rotate_factor);

const directions = 3;

const log_side_len = 0x4;
const side_len = 1 << log_side_len;

const log_volume_len = log_side_len * directions;
const volume_len = 1 << log_volume_len;

const morton_mask: CombinedIndices = _: {
    var value: usize = 0;
    for (0..log_side_len) |_| {
        value <<= directions;
        value |= 1;
    }
    break :_ value;
};

const Voxel = enum(u8) {
    Air,
    Stone,
};

const Uniform = extern struct {
    light: extern struct {
        affine: [directions][directions + 1]f32,
        direction: [directions]f32,
    },
    padding: u32,
    camera: extern struct {
        position: [directions]f32,

        near: f32,

        cos: [directions - 1]f32,
        sin: [directions - 1]f32,
        scale: [directions - 1]f32,

        far: f32,
    },
};
const camera = struct {
    const Keyboard = struct {
        positive: [directions]bool,
        negative: [directions]bool,

        locked_pointer_toggle: bool,
    };
    const RelativePointerV1 = struct {
        // fixed point 24.8
        dzx: i32,
        dzy: i32,
    };
    const Pointer = struct {
        dzx_half_range: i32,
        speedup: bool,
        speeddown: bool,
        destroy: bool,
        create: bool,
    };
};

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    const wl_display = try wl.Display.connect(null);
    const wl_registry = try wl_display.getRegistry();
    var registry_listener_data: RegistryListenerData = undefined;
    wl_registry.setListener(*RegistryListenerData, registryListener, &registry_listener_data);
    if (wl_display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const wl_compositor = registry_listener_data.wl_compositor;
    const xdg_wm_base = registry_listener_data.xdg_wm_base;
    const wl_seat = registry_listener_data.wl_seat;
    const wl_output = registry_listener_data.wl_output;
    const zwp_relative_pointer_manager_v1 = registry_listener_data.zwp_relative_pointer_manager_v1;
    const zwp_pointer_constraints_v1 = registry_listener_data.zwp_pointer_constraints_v1;

    const keyboard = try wl_seat.getKeyboard();
    defer keyboard.destroy();

    var camera_keyboard: camera.Keyboard = .{
        .positive = [_]bool{false} ** directions,
        .negative = [_]bool{false} ** directions,
        .locked_pointer_toggle = false,
    };
    keyboard.setListener(
        *camera.Keyboard,
        keyboardListener,
        &camera_keyboard,
    );

    const wl_pointer = try wl_seat.getPointer();
    defer wl_pointer.destroy();

    var camera_pointer: camera.Pointer = .{
        .dzx_half_range = 0,
        .speedup = false,
        .speeddown = false,
        .create = false,
        .destroy = false,
    };
    var camera_zx_half_range: i32 = range_rotate_unit / 6;
    wl_pointer.setListener(
        *camera.Pointer,
        pointerListener,
        &camera_pointer,
    );

    var camera_relative_pointer_v1: camera.RelativePointerV1 = .{
        .dzx = 0,
        .dzy = 0,
    };
    var camera_angles: [directions - 1]i32 = [_]i32{0} ** (directions - 1);
    const relative_pointer_v1 = try zwp_relative_pointer_manager_v1.getRelativePointer(wl_pointer);
    relative_pointer_v1.setListener(
        *camera.RelativePointerV1,
        relativePointerV1Listener,
        &camera_relative_pointer_v1,
    );

    const wl_surface = try wl_compositor.createSurface();
    defer wl_surface.destroy();

    var scale_factor: i32 = 0;
    wl_output.setListener(*i32, outputListener, &scale_factor);

    const xdg_surface = try xdg_wm_base.getXdgSurface(wl_surface);
    defer xdg_surface.destroy();

    const toplevel = try xdg_surface.getToplevel();
    defer toplevel.destroy();

    var xdg_toplevel_event: ?xdg.Toplevel.Event = null;
    toplevel.setListener(
        *?xdg.Toplevel.Event,
        xdgToplevelListener,
        &xdg_toplevel_event,
    );

    var zwp_locked_pointer_v1: ?*zwp.LockedPointerV1 = null;

    wl_surface.commit();

    var lib = try std.DynLib.open("libvulkan.so");
    defer lib.close();
    const vkGetInstanceProcAddr = lib.lookup(
        vk.PfnGetInstanceProcAddr,
        "vkGetInstanceProcAddr",
    ).?;

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
        // vk.LayerSettingEXT{
        //     .p_layer_name = "VK_LAYER_KHRONOS_validation",
        //     .p_setting_name = "debug_action",
        //     .type = .string_ext,
        //     .value_count = 1,
        //     .p_values = @ptrCast(&"VK_DBG_LAYER_ACTION_LOG_MSG"),
        // },
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

    var physical_device_descriptor_buffer_properties: vk.PhysicalDeviceDescriptorBufferPropertiesEXT = undefined;
    physical_device_descriptor_buffer_properties.s_type = .physical_device_descriptor_buffer_properties_ext;
    physical_device_descriptor_buffer_properties.p_next = null;

    var physical_device_properties: vk.PhysicalDeviceProperties2 = .{
        .p_next = @ptrCast(&physical_device_descriptor_buffer_properties),
        .properties = undefined,
    };
    vki.getPhysicalDeviceProperties2(physical_device, &physical_device_properties);
    const physical_device_memory_properties = vki.getPhysicalDeviceMemoryProperties(physical_device);

    var features_descriptor_buffer: vk.PhysicalDeviceDescriptorBufferFeaturesEXT = .{
        .descriptor_buffer = vk.TRUE,
    };
    var featuers_1_4: vk.PhysicalDeviceVulkan14Features = .{
        .dynamic_rendering_local_read = vk.TRUE,
        .maintenance_6 = vk.TRUE,
        .p_next = @ptrCast(&features_descriptor_buffer),
    };
    var features_1_3: vk.PhysicalDeviceVulkan13Features = .{
        .dynamic_rendering = vk.TRUE,
        .synchronization_2 = vk.TRUE,
        .p_next = @ptrCast(&featuers_1_4),
    };
    var features_1_2: vk.PhysicalDeviceVulkan12Features = .{
        .buffer_device_address = vk.TRUE,
        .descriptor_indexing = vk.TRUE,
        .vulkan_memory_model = vk.TRUE,
        .vulkan_memory_model_availability_visibility_chains = vk.TRUE,
        .shader_int_8 = vk.TRUE,
        .uniform_and_storage_buffer_8_bit_access = vk.TRUE,
        .p_next = @ptrCast(&features_1_3),
    };
    var features_1_1: vk.PhysicalDeviceVulkan11Features = .{
        .shader_draw_parameters = vk.TRUE,
        .p_next = @ptrCast(&features_1_2),
    };
    var features: vk.PhysicalDeviceFeatures2 = .{
        .features = .{
            .shader_int_64 = vk.TRUE,
            .shader_int_16 = vk.TRUE,
            .fill_mode_non_solid = vk.TRUE,
            .wide_lines = vk.TRUE,
        },
        .p_next = @ptrCast(&features_1_1),
    };

    const device_extensions = [_][*:0]const u8{
        "VK_KHR_swapchain",
        "VK_EXT_descriptor_buffer",
        "VK_KHR_maintenance6",
    };

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
            .p_enabled_features = null,
            .p_next = @ptrCast(&features),
        },
        null,
    );
    const vkd: vk.DeviceWrapper = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
    defer vkd.destroyDevice(device, null);

    const queue = vkd.getDeviceQueue2(device, &vk.DeviceQueueInfo2{
        .queue_family_index = 0,
        .queue_index = 0,
    });

    const vk_surface = try vki.createWaylandSurfaceKHR(
        instance,
        &.{
            .display = @ptrCast(wl_display),
            .surface = @ptrCast(wl_surface),
        },
        null,
    );
    defer vki.destroySurfaceKHR(instance, vk_surface, null);

    const surface_format: vk.SurfaceFormatKHR = .{
        .format = .r8g8b8a8_srgb,
        .color_space = .srgb_nonlinear_khr,
    };
    {
        var surface_formats_count: u32 = undefined;
        _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(physical_device, vk_surface, &surface_formats_count, null);
        const surface_formats = try allocator.alloc(vk.SurfaceFormatKHR, surface_formats_count);
        allocator.free(surface_formats);
        _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(physical_device, vk_surface, &surface_formats_count, surface_formats.ptr);
    }

    var swapchain: vk.SwapchainKHR = .null_handle;
    var swapchain_extent: vk.Extent2D = .{
        .width = ~@as(u32, 0),
        .height = ~@as(u32, 0),
    };
    var swapchain_images: [4]vk.Image = undefined;
    var swapchain_images_count: u32 = undefined;
    var swapchain_image_views: [4]vk.ImageView = undefined;

    var depth: RenderTarget = .{
        .format = .d32_sfloat,
    };
    defer depth.destroy(vkd, device);

    var geometry: RenderTarget = .{
        .format = .r32g32b32a32_uint,
    };
    defer geometry.destroy(vkd, device);

    var color: RenderTarget = .{
        .format = .r16g16b16_sfloat,
    };
    defer color.destroy(vkd, device);

    const _voxels: []Voxel = try allocator.alloc(Voxel, volume_len);
    defer allocator.free(_voxels);
    const voxels: [*]Voxel = _voxels.ptr;

    const log_shadow_side_len = 0xa;
    const shadow_side_len = 1 << log_shadow_side_len;
    const shadow_extent: vk.Extent2D = .{
        .width = shadow_side_len,
        .height = shadow_side_len,
    };

    var shadow: RenderTarget = .{
        .format = .d32_sfloat,
    };
    try shadow.init(
        vkd,
        device,
        shadow_extent,
        physical_device_memory_properties,
    );
    defer shadow.destroy(vkd, device);

    const shadow_sampler = try vkd.createSampler(
        device,
        &vk.SamplerCreateInfo{
            .address_mode_u = .clamp_to_border,
            .address_mode_v = .clamp_to_border,
            .address_mode_w = undefined,
            .anisotropy_enable = vk.FALSE,
            .border_color = .float_opaque_black,
            .compare_enable = vk.TRUE,
            .compare_op = .less,
            .flags = .{},
            .mag_filter = .linear,
            .min_filter = .linear,
            .max_anisotropy = 0,
            .max_lod = 0,
            .min_lod = 0,
            .mip_lod_bias = 0,
            .mipmap_mode = .nearest,
            .unnormalized_coordinates = vk.FALSE,
        },
        null,
    );
    defer vkd.destroySampler(
        device,
        shadow_sampler,
        null,
    );

    const full_screen_sampler = try vkd.createSampler(
        device,
        &vk.SamplerCreateInfo{
            .address_mode_u = .clamp_to_border,
            .address_mode_v = .clamp_to_border,
            .address_mode_w = undefined,
            .anisotropy_enable = vk.FALSE,
            .border_color = .float_opaque_black,
            .compare_enable = vk.FALSE,
            .compare_op = .less,
            .flags = .{},
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .max_anisotropy = 0,
            .max_lod = 0,
            .min_lod = 0,
            .mip_lod_bias = 0,
            .mipmap_mode = .nearest,
            .unnormalized_coordinates = vk.FALSE,
        },
        null,
    );
    defer vkd.destroySampler(
        device,
        full_screen_sampler,
        null,
    );

    var uniform: Buffer = try Buffer.init(
        vkd,
        device,
        physical_device_memory_properties,
        vk.BufferUsageFlags{
            .uniform_buffer_bit = true,
            .shader_device_address_bit = true,
        },
        @sizeOf(Uniform),
        vk.MemoryPropertyFlags{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        },
    );
    defer uniform.destroy(vkd, device);

    const uniform_mapped: *Uniform = @ptrCast(@alignCast(try uniform.mapMemory(vkd, device)));

    const descriptor_types_list = .{
        .shadow = .{
            .uniform = .uniform_buffer,
        },
        .geometry = .{
            .uniform = .uniform_buffer,
        },
        .lighting = .{
            .uniform = .uniform_buffer,
            .shadow = .combined_image_sampler,
            .depth = .combined_image_sampler,
            .geometry = .combined_image_sampler,
        },
    };
    const descriptor_memory_offsets = initDescriptorBufferOffsets(
        DescriptorBufferOffsets(descriptor_types_list),
        physical_device_descriptor_buffer_properties,
    );
    print("{x}\n\n\n", .{descriptor_memory_offsets.geometry.uniform.value});

    const shaders_spv align(@alignOf(u32)) = @embedFile("shaders").*;

    const shader = try vkd.createShaderModule(
        device,
        &vk.ShaderModuleCreateInfo{
            .p_code = @ptrCast(&shaders_spv),
            .code_size = shaders_spv.len,
        },
        null,
    );
    defer vkd.destroyShaderModule(device, shader, null);

    const pipelines = try createPipelines(
        Pipelines(.{
            .shadow = vk.ShaderStageFlags{
                .vertex_bit = true,
            },
            .geometry = vk.ShaderStageFlags{
                .vertex_bit = true,
            },
            .lighting = vk.ShaderStageFlags{
                .fragment_bit = true,
            },
        }),
        descriptor_types_list,
        .{
            .shadow = vk.GraphicsPipelineCreateInfo{
                .flags = .{
                    .descriptor_buffer_bit_ext = true,
                },
                .base_pipeline_handle = .null_handle,
                .base_pipeline_index = undefined,
                .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                    .topology = .triangle_strip,
                    .primitive_restart_enable = vk.FALSE,
                },
                .p_color_blend_state = null,
                .p_depth_stencil_state = &vk.PipelineDepthStencilStateCreateInfo{
                    .depth_bounds_test_enable = vk.TRUE,
                    .flags = .{},
                    .depth_compare_op = .less,
                    .depth_test_enable = vk.TRUE,
                    .depth_write_enable = vk.TRUE,
                    .stencil_test_enable = vk.FALSE,
                    .back = undefined,
                    .front = undefined,
                    .min_depth_bounds = 0.0,
                    .max_depth_bounds = 1.0,
                },
                .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
                    .vertex_attribute_description_count = 1,
                    .vertex_binding_description_count = 1,
                    .p_vertex_attribute_descriptions = @ptrCast(&vk.VertexInputAttributeDescription{
                        .binding = 0,
                        .format = getVkFormat(Surface),
                        .location = 0,
                        .offset = 0,
                    }),
                    .p_vertex_binding_descriptions = @ptrCast(&vk.VertexInputBindingDescription{
                        .binding = 0,
                        .stride = @sizeOf(Surface),
                        .input_rate = .instance,
                    }),
                },
                .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
                    .cull_mode = .{ .front_bit = true },
                    .front_face = .clockwise,
                    .polygon_mode = .fill,
                    .depth_bias_enable = vk.TRUE,
                    .depth_clamp_enable = vk.FALSE,
                    .depth_bias_clamp = 0.0,
                    .depth_bias_constant_factor = 0.0,
                    .depth_bias_slope_factor = -2.0,
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
                    .p_scissors = @ptrCast(&vk.Rect2D{
                        .extent = shadow_extent,
                        .offset = offset_zero,
                    }),
                    .viewport_count = 1,
                    .p_viewports = @ptrCast(&vk.Viewport{
                        .x = 0.0,
                        .y = 0.0,
                        .width = @floatFromInt(shadow_side_len),
                        .height = @floatFromInt(shadow_side_len),
                        .min_depth = 0.0,
                        .max_depth = 1.0,
                    }),
                },
                .p_dynamic_state = null,
                .stage_count = 1,
                .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
                    vk.PipelineShaderStageCreateInfo{
                        .p_name = "vertexShadow",
                        .stage = .{
                            .vertex_bit = true,
                        },
                        .module = shader,
                    },
                },
                .render_pass = .null_handle,
                .subpass = undefined,
                .p_next = &vk.PipelineRenderingCreateInfo{
                    .depth_attachment_format = shadow.format,
                    .stencil_attachment_format = .undefined,
                    .view_mask = 0,
                },
            },
            .geometry = vk.GraphicsPipelineCreateInfo{
                .flags = .{
                    .descriptor_buffer_bit_ext = true,
                },
                .base_pipeline_handle = .null_handle,
                .base_pipeline_index = undefined,
                .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                    .topology = .triangle_strip,
                    .primitive_restart_enable = vk.FALSE,
                },
                .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
                    .attachment_count = 1,
                    .logic_op = undefined,
                    .blend_constants = undefined,
                    .flags = .{},
                    .logic_op_enable = vk.FALSE,
                    .p_attachments = @ptrCast(&vk.PipelineColorBlendAttachmentState{
                        .color_write_mask = .{
                            .r_bit = true,
                            .g_bit = true,
                            .b_bit = true,
                            .a_bit = true,
                        },
                        .blend_enable = vk.FALSE,
                        .alpha_blend_op = undefined,
                        .color_blend_op = undefined,
                        .dst_alpha_blend_factor = undefined,
                        .dst_color_blend_factor = undefined,
                        .src_alpha_blend_factor = undefined,
                        .src_color_blend_factor = undefined,
                    }),
                },
                .p_depth_stencil_state = &vk.PipelineDepthStencilStateCreateInfo{
                    .depth_bounds_test_enable = vk.TRUE,
                    .flags = .{},
                    .depth_compare_op = .greater,
                    .depth_test_enable = vk.TRUE,
                    .depth_write_enable = vk.TRUE,
                    .stencil_test_enable = vk.FALSE,
                    .back = undefined,
                    .front = undefined,
                    .min_depth_bounds = 0.0,
                    .max_depth_bounds = 1.0,
                },
                .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
                    .vertex_attribute_description_count = 1,
                    .vertex_binding_description_count = 1,
                    .p_vertex_attribute_descriptions = @ptrCast(&vk.VertexInputAttributeDescription{
                        .binding = 0,
                        .format = getVkFormat(Surface),
                        .location = 0,
                        .offset = 0,
                    }),
                    .p_vertex_binding_descriptions = @ptrCast(&vk.VertexInputBindingDescription{
                        .binding = 0,
                        .stride = @sizeOf(Surface),
                        .input_rate = .instance,
                    }),
                },
                .p_rasterization_state = &vk.PipelineRasterizationStateCreateInfo{
                    .cull_mode = .{ .back_bit = true },
                    .front_face = .clockwise,
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
                    .p_dynamic_states = &[_]vk.DynamicState{
                        .scissor,
                        .viewport,
                    },
                },
                .stage_count = 2,
                .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
                    vk.PipelineShaderStageCreateInfo{
                        .p_name = "vertexGeometry",
                        .stage = .{
                            .vertex_bit = true,
                        },
                        .module = shader,
                    },
                    vk.PipelineShaderStageCreateInfo{
                        .p_name = "fragmentGeometry",
                        .stage = .{
                            .fragment_bit = true,
                        },
                        .module = shader,
                    },
                },
                .render_pass = .null_handle,
                .subpass = undefined,
                .p_next = &vk.PipelineRenderingCreateInfo{
                    .color_attachment_count = 1,
                    .p_color_attachment_formats = &[_]vk.Format{geometry.format},
                    .depth_attachment_format = depth.format,
                    .stencil_attachment_format = .undefined,
                    .view_mask = 0,
                },
            },
            .lighting = vk.GraphicsPipelineCreateInfo{
                .flags = .{
                    .descriptor_buffer_bit_ext = true,
                },
                .base_pipeline_handle = .null_handle,
                .base_pipeline_index = undefined,
                .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                    .topology = .triangle_list,
                    .primitive_restart_enable = vk.FALSE,
                },
                .p_color_blend_state = &vk.PipelineColorBlendStateCreateInfo{
                    .attachment_count = 1,
                    .blend_constants = undefined,
                    .logic_op = undefined,
                    .logic_op_enable = undefined,
                    .p_attachments = @ptrCast(&vk.PipelineColorBlendAttachmentState{
                        .color_write_mask = .{
                            .r_bit = true,
                            .b_bit = true,
                            .g_bit = true,
                            .a_bit = false,
                        },
                        .blend_enable = undefined,
                        .src_color_blend_factor = undefined,
                        .dst_color_blend_factor = undefined,
                        .color_blend_op = undefined,
                        .src_alpha_blend_factor = undefined,
                        .dst_alpha_blend_factor = undefined,
                        .alpha_blend_op = undefined,
                    }),
                },
                .p_depth_stencil_state = null,
                .p_vertex_input_state = &vk.PipelineVertexInputStateCreateInfo{
                    .vertex_attribute_description_count = 0,
                    .vertex_binding_description_count = 0,
                },
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
                    .p_dynamic_states = @ptrCast(&[_]vk.DynamicState{
                        .viewport,
                        .scissor,
                    }),
                },
                .stage_count = 2,
                .p_stages = &[_]vk.PipelineShaderStageCreateInfo{
                    vk.PipelineShaderStageCreateInfo{
                        .p_name = "vertexFullScreen",
                        .stage = .{
                            .vertex_bit = true,
                        },
                        .module = shader,
                    },
                    vk.PipelineShaderStageCreateInfo{
                        .p_name = "fragmentLighting",
                        .stage = .{
                            .fragment_bit = true,
                        },
                        .module = shader,
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
            },
        },
        vkd,
        device,
    );
    defer destroyPipelines(pipelines, vkd, device);

    const shadow_descriptor_set_layout_size = vkd.getDescriptorSetLayoutSizeEXT(
        device,
        pipelines.shadow.descriptor_set_layout,
    );

    const geometry_descriptor_set_layout_size = vkd.getDescriptorSetLayoutSizeEXT(
        device,
        pipelines.geometry.descriptor_set_layout,
    );

    const lighting_descriptor_set_layout_size = vkd.getDescriptorSetLayoutSizeEXT(
        device,
        pipelines.lighting.descriptor_set_layout,
    );

    var descriptors: Buffer = try Buffer.init(
        vkd,
        device,
        physical_device_memory_properties,
        vk.BufferUsageFlags{
            .resource_descriptor_buffer_bit_ext = true,
            .shader_device_address_bit = true,
        },
        0 +
            shadow_descriptor_set_layout_size +
            geometry_descriptor_set_layout_size +
            lighting_descriptor_set_layout_size,
        vk.MemoryPropertyFlags{
            .host_visible_bit = true,
            .host_coherent_bit = true,
        },
    );
    defer descriptors.destroy(vkd, device);

    const descriptors_mapped: [*]u8 = @ptrCast(try descriptors.mapMemory(vkd, device));

    const descriptor_buffer_device_address = vkd.getBufferDeviceAddress(
        device,
        &vk.BufferDeviceAddressInfo{
            .buffer = descriptors.buffer,
        },
    );

    const uniform_buffer_address_info = vk.DescriptorAddressInfoEXT{
        .format = .undefined,
        .range = uniform.size,
        .address = vkd.getBufferDeviceAddress(
            device,
            &vk.BufferDeviceAddressInfo{
                .buffer = uniform.buffer,
            },
        ),
    };

    getDescriptor(
        descriptor_memory_offsets.shadow.uniform,
        vkd,
        device,
        physical_device_descriptor_buffer_properties,
        uniform_buffer_address_info,
        descriptors_mapped,
    );

    getDescriptor(
        descriptor_memory_offsets.geometry.uniform,
        vkd,
        device,
        physical_device_descriptor_buffer_properties,
        uniform_buffer_address_info,
        descriptors_mapped,
    );

    getDescriptor(
        descriptor_memory_offsets.lighting.uniform,
        vkd,
        device,
        physical_device_descriptor_buffer_properties,
        uniform_buffer_address_info,
        descriptors_mapped,
    );

    getDescriptor(
        descriptor_memory_offsets.lighting.shadow,
        vkd,
        device,
        physical_device_descriptor_buffer_properties,
        vk.DescriptorImageInfo{
            .image_layout = .depth_read_only_optimal,
            .image_view = shadow.image_view,
            .sampler = shadow_sampler,
        },
        descriptors_mapped,
    );

    for (0..volume_len) |voxel_i| {
        var sum: usize = 0;
        for (0..directions) |_direction_i| {
            const direction_i: Direction = @truncate(_direction_i);
            const coord = pext(voxel_i, morton_mask << direction_i);
            sum += coord * coord;
        }
        const max_sum = 1 << (2 * log_side_len);
        voxels[voxel_i] = if (sum > max_sum) Voxel.Stone else Voxel.Air;
    }

    // const PartialVoxelId = UnsignedInt(log_side_len);
    // factor of 2 is completely arbitrary
    // it's to allow modification of mesh once memory been allocated
    // TODO: fix, more dynamic voxel set modification, and mesh generation
    var surfaces_len = computeSurfacesLen(voxels);
    const surfaces_len_saturated = surfaces_len * 2;

    var surfaces_transfer: Buffer = try Buffer.init(
        vkd,
        device,
        physical_device_memory_properties,
        vk.BufferUsageFlags{
            .transfer_src_bit = true,
        },
        @sizeOf(Surface) * surfaces_len_saturated,
        vk.MemoryPropertyFlags{
            .host_visible_bit = true,
        },
    );
    defer surfaces_transfer.destroy(vkd, device);

    const surfaces_mapped: [*]Surface = @ptrCast(@alignCast(try surfaces_transfer.mapMemory(vkd, device)));

    var surfaces: Buffer = try Buffer.init(
        vkd,
        device,
        physical_device_memory_properties,
        vk.BufferUsageFlags{
            .vertex_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        surfaces_transfer.size,
        vk.MemoryPropertyFlags{
            .device_local_bit = true,
        },
    );
    defer surfaces.destroy(vkd, device);

    defer vkd.destroySwapchainKHR(device, swapchain, null);
    defer for (0..swapchain_images_count) |i| {
        vkd.destroyImageView(device, swapchain_image_views[i], null);
    };

    const surface_capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(
        physical_device,
        vk_surface,
    );

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
    var time = nanoTimestamp();

    var camera_speed: f32 = 4.0;

    // var rand = std.Random.DefaultPrng.init(@bitCast(std.time.microTimestamp()));
    // const random = rand.random();
    var light_angle: f32 = 0.0;

    // const side: f32 = (side_len / @sqrt(@as(f32, @floatFromInt(directions))));
    uniform_mapped.camera.position = [_]f32{-0.5} ** directions;

    voxels[0] = Voxel.Stone;
    voxels[0b000_010_010_011] = Voxel.Stone;

    var voxel_regenerate = true;

    while (true) {
        if (wl_display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

        const new_time = nanoTimestamp();
        const dt = @as(f32, @floatFromInt(new_time - time)) / 1_000_000_000.0;
        time = new_time;

        light_angle += dt * 0.05;
        light_angle = @max(tau, light_angle);
        const light_direction: [directions]f32 = .{
            @sin(0.3) / @sqrt(2.0),
            @sin(0.3) / @sqrt(2.0),
            @cos(0.3),
        };
        {
            if (camera_keyboard.locked_pointer_toggle) {
                if (zwp_locked_pointer_v1 == null) {
                    zwp_locked_pointer_v1 = try zwp_pointer_constraints_v1.lockPointer(
                        wl_surface,
                        wl_pointer,
                        null,
                        .persistent,
                    );
                } else {
                    zwp_locked_pointer_v1.?.destroy();
                    zwp_locked_pointer_v1 = null;
                }
                camera_keyboard.locked_pointer_toggle = false;
            }

            if (zwp_locked_pointer_v1 != null) {
                camera_angles[0] += camera_relative_pointer_v1.dzx;

                camera_angles[1] += camera_relative_pointer_v1.dzy;
                camera_angles[1] = clamp(
                    camera_angles[1],
                    -(rotate_unit / 4) * 0xf / 0x10,
                    (rotate_unit / 4) * 0xf / 0x10,
                );

                camera_zx_half_range += camera_pointer.dzx_half_range;

                camera_zx_half_range = clamp(
                    camera_zx_half_range,
                    range_rotate_unit / 0xd,
                    range_rotate_unit / 0x4,
                );

                camera_relative_pointer_v1.dzx = 0;
                camera_relative_pointer_v1.dzy = 0;
                camera_pointer.dzx_half_range = 0;
            }

            var angles: [directions - 1]f32 = undefined;
            for (0..directions - 1) |direction| {
                angles[direction] = @floatFromInt(camera_angles[direction]);
                angles[direction] /= rotate_factor;
            }

            var zx_half_range: f32 = @floatFromInt(camera_zx_half_range);
            zx_half_range /= range_rotate_factor;

            if (zwp_locked_pointer_v1 != null) {
                if (camera_pointer.speedup and !camera_pointer.speeddown) {
                    camera_speed += dt;
                } else if (!camera_pointer.speedup and camera_pointer.speeddown) {
                    camera_speed -= dt;
                    camera_speed = @max(0.0, camera_speed);
                }

                const ds = camera_speed * dt;

                var d: [directions]f32 = [_]f32{
                    @sin(angles[0]) * @cos(angles[1]),
                    1.0,
                    @cos(angles[0]) * @cos(angles[1]),
                };
                for (&d) |*v| {
                    v.* *= ds;
                }

                if (camera_keyboard.positive[0] and !camera_keyboard.negative[0]) {
                    uniform_mapped.camera.position[2] -= d[0];
                    uniform_mapped.camera.position[0] += d[2];
                } else if (!camera_keyboard.positive[0] and camera_keyboard.negative[0]) {
                    uniform_mapped.camera.position[2] += d[0];
                    uniform_mapped.camera.position[0] -= d[2];
                }

                if (camera_keyboard.positive[1] and !camera_keyboard.negative[1]) {
                    uniform_mapped.camera.position[1] += d[1];
                } else if (!camera_keyboard.positive[1] and camera_keyboard.negative[1]) {
                    uniform_mapped.camera.position[1] -= d[1];
                }

                if (camera_keyboard.positive[2] and !camera_keyboard.negative[2]) {
                    uniform_mapped.camera.position[2] += d[2];
                    uniform_mapped.camera.position[0] += d[0];
                } else if (!camera_keyboard.positive[2] and camera_keyboard.negative[2]) {
                    uniform_mapped.camera.position[2] -= d[2];
                    uniform_mapped.camera.position[0] -= d[0];
                }
            }

            if (camera_pointer.create != camera_pointer.destroy) {
                const origin = uniform_mapped.camera.position;
                var indices: [directions]usize = undefined;
                var combined_indices: usize = 0;

                // const direction: [directions]f32 = [_]f32{1.0} ** directions;
                // for (0 .. directions

                var ray: [directions]f32 = [_]f32{
                    0.0,
                    0.0,
                    1.0,
                };
                ray[1] = ray[2] * @sin(angles[1]);
                ray[2] *= @cos(angles[1]);

                ray[0] = ray[2] * @sin(angles[0]);
                ray[2] *= @cos(angles[0]);

                var planes: [directions]f32 = undefined;
                for (0..directions) |direction_i| {
                    planes[direction_i] =
                        if (ray[direction_i] > 0.0)
                            @ceil(origin[direction_i])
                        else
                            @floor(origin[direction_i]);

                    indices[direction_i] = @intFromFloat(origin[direction_i]);
                    combined_indices |= pdep(indices[direction_i], morton_mask << @truncate(direction_i));
                }

                var old_combined_indices: ?usize = null;
                while (true) {
                    if (voxels[combined_indices] != Voxel.Air) {
                        if (camera_pointer.destroy) {
                            voxels[combined_indices] = Voxel.Air;
                        } else if (old_combined_indices != null) {
                            voxels[old_combined_indices.?] = Voxel.Stone;
                        }
                        voxel_regenerate = true;
                        break;
                    }
                    var t_min: f32 = std.math.inf(f32);
                    var direction_i_min: usize = undefined;
                    for (0..directions) |direction_i| {
                        const t = (planes[direction_i] - origin[direction_i]) / ray[direction_i];
                        if (t < t_min) {
                            t_min = t;
                            direction_i_min = direction_i;
                        }
                    }
                    if (ray[direction_i_min] > 0.0) {
                        if (indices[direction_i_min] == side_len - 1) break;
                        planes[direction_i_min] += 1.0;
                        indices[direction_i_min] += 1;
                    } else {
                        if (indices[direction_i_min] == 0) break;
                        planes[direction_i_min] -= 1.0;
                        indices[direction_i_min] -= 1;
                    }
                    const mask = morton_mask << @truncate(direction_i_min);

                    old_combined_indices = combined_indices;
                    combined_indices &= ~mask;
                    combined_indices |= pdep(indices[direction_i_min], mask);
                }
            }

            camera_pointer.create = false;
            camera_pointer.destroy = false;

            if (voxel_regenerate) {
                surfaces_len = computeSurfacesLen(voxels);
                generateSurfaces(voxels, surfaces_mapped);
            }

            var camera_scale: [directions - 1]f32 = undefined;
            camera_scale[0] = 0.2;
            uniform_mapped.camera.near = camera_scale[0] / @tan(zx_half_range);
            uniform_mapped.camera.scale[0] = uniform_mapped.camera.near / camera_scale[0];
            uniform_mapped.camera.scale[1] = uniform_mapped.camera.scale[0] *
                @as(f32, @floatFromInt(swapchain_extent.width)) /
                @as(f32, @floatFromInt(swapchain_extent.height));

            for (0..directions - 1) |direction| {
                uniform_mapped.camera.cos[direction] = @cos(angles[direction]);
                uniform_mapped.camera.sin[direction] = @sin(angles[direction]);
            }
            uniform_mapped.light.direction = light_direction;

            camera_scale[1] = camera_scale[0] * @as(f32, @floatFromInt(swapchain_extent.height)) /
                @as(f32, @floatFromInt(swapchain_extent.width));

            uniform_mapped.light.affine = computeLightAffine(
                uniform_mapped.camera.position,
                light_direction,
                angles,
                uniform_mapped.camera.near,
                uniform_mapped.camera.near + 10.0,
                camera_scale,
            );
        }

        var init_swapchain: bool = false;

        if (xdg_toplevel_event) |event| event_handle: {
            switch (event) {
                .close => {
                    try vkd.deviceWaitIdle(device);
                    return;
                },
                .configure => |configure| {
                    const width: u32 = @intCast(scale_factor * configure.width);
                    const height: u32 = @intCast(scale_factor * configure.height);

                    if (width == swapchain_extent.width and height == swapchain_extent.height) {
                        break :event_handle;
                    }

                    swapchain_extent = .{
                        .width = width,
                        .height = height,
                    };

                    if (swapchain != .null_handle) {
                        try vkd.deviceWaitIdle(device);
                        for (0..swapchain_images_count) |i| {
                            vkd.destroyImageView(
                                device,
                                swapchain_image_views[i],
                                null,
                            );
                        }
                        vkd.destroySwapchainKHR(
                            device,
                            swapchain,
                            null,
                        );

                        depth.destroy(vkd, device);

                        geometry.destroy(vkd, device);

                        color.destroy(vkd, device);
                    }

                    {
                        try depth.init(
                            vkd,
                            device,
                            swapchain_extent,
                            physical_device_memory_properties,
                        );

                        getDescriptor(
                            descriptor_memory_offsets.lighting.depth,
                            vkd,
                            device,
                            physical_device_descriptor_buffer_properties,
                            vk.DescriptorImageInfo{
                                .image_layout = .depth_read_only_optimal,
                                .image_view = depth.image_view,
                                .sampler = full_screen_sampler,
                            },
                            descriptors_mapped,
                        );
                    }

                    {
                        try geometry.init(
                            vkd,
                            device,
                            swapchain_extent,
                            physical_device_memory_properties,
                        );

                        getDescriptor(
                            descriptor_memory_offsets.lighting.geometry,
                            vkd,
                            device,
                            physical_device_descriptor_buffer_properties,
                            vk.DescriptorImageInfo{
                                .image_layout = .depth_read_only_optimal,
                                .image_view = geometry.image_view,
                                .sampler = full_screen_sampler,
                            },
                            descriptors_mapped,
                        );
                    }

                    {
                        try color.init(
                            vkd,
                            device,
                            swapchain_extent,
                            physical_device_memory_properties,
                        );
                    }

                    init_swapchain = true;

                    swapchain = try vkd.createSwapchainKHR(
                        device,
                        &.{
                            .surface = vk_surface,
                            .min_image_count = surface_capabilities.min_image_count,
                            .image_format = surface_format.format,
                            .image_color_space = surface_format.color_space,
                            .image_extent = swapchain_extent,
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
                            &imageViewCreateInfo(
                                swapchain_images[i],
                                surface_format.format,
                            ),
                            null,
                        );
                        if (render_end_semaphores[i] == .null_handle) {
                            render_end_semaphores[i] = try vkd.createSemaphore(device, &.{}, null);
                        }
                    }
                },
                else => {},
            }
            xdg_toplevel_event = null;
        }

        var descriptor_buffer_binding_info = vk.DescriptorBufferBindingInfoEXT{
            .address = descriptor_buffer_device_address,
            .usage = descriptors.usage,
        };
        var descriptor_buffer_offsets_info = vk.SetDescriptorBufferOffsetsInfoEXT{
            .first_set = 0,
            .set_count = 1,
            .layout = .null_handle,
            .p_buffer_indices = &[_]u32{0},
            .stage_flags = undefined,
            .p_offsets = &[_]vk.DeviceSize{0},
        };
        const image_memory_barrier: vk.ImageMemoryBarrier2 = .{
            .src_stage_mask = undefined,
            .src_access_mask = undefined,
            .old_layout = undefined,
            .new_layout = undefined,
            .dst_stage_mask = undefined,
            .dst_access_mask = undefined,
            .image = undefined,
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
            .subresource_range = .{
                .aspect_mask = .{},
                .base_array_layer = 0,
                .base_mip_level = 0,
                .layer_count = 1,
                .level_count = 1,
            },
        };
        // _ = image_memory_barrier;

        _ = try vkd.waitForFences(
            device,
            1,
            @ptrCast(&command_buffer_fences[frame]),
            vk.TRUE,
            std.math.maxInt(u64),
        );
        _ = try vkd.resetFences(
            device,
            1,
            @ptrCast(&command_buffer_fences[frame]),
        );

        const swapchain_image_index = (try vkd.acquireNextImageKHR(
            device,
            swapchain,
            std.math.maxInt(u64),
            swapchain_image_semaphores[frame],
            .null_handle,
        )).image_index;

        try vkd.resetCommandPool(device, command_pools[frame], .{});

        const command_buffer = command_buffers[frame];
        {
            try vkd.beginCommandBuffer(command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });

            if (voxel_regenerate) {
                vkd.cmdCopyBuffer2(
                    command_buffer,
                    &vk.CopyBufferInfo2{
                        .src_buffer = surfaces_transfer.buffer,
                        .dst_buffer = surfaces.buffer,
                        .region_count = 1,
                        .p_regions = @ptrCast(&vk.BufferCopy2{
                            .src_offset = 0,
                            .dst_offset = 0,
                            .size = @sizeOf(Surface) * surfaces_len,
                        }),
                    },
                );

                voxel_regenerate = false;
            }

            cmdPipelineImageMemoryBarrier(
                vkd,
                command_buffer,
                [_]*RenderTarget{
                    &shadow,
                },
            );

            vkd.cmdPipelineBarrier2(command_buffer, &vk.DependencyInfo{
                .buffer_memory_barrier_count = 1,
                .p_buffer_memory_barriers = @ptrCast(&vk.BufferMemoryBarrier2{
                    .buffer = surfaces.buffer,
                    .offset = 0,
                    .size = @sizeOf(Surface) * surfaces_len,
                    .src_stage_mask = .{
                        .copy_bit = true,
                    },
                    .dst_stage_mask = .{
                        .copy_bit = true,
                    },
                    .src_access_mask = .{
                        .transfer_write_bit = true,
                    },
                    .dst_access_mask = .{
                        .transfer_write_bit = true,
                    },
                    .src_queue_family_index = 0,
                    .dst_queue_family_index = 0,
                }),
            });

            // shadow pass
            {
                vkd.cmdBeginRendering(command_buffer, &vk.RenderingInfo{
                    .color_attachment_count = 0,
                    .view_mask = 0,
                    .render_area = .{
                        .offset = offset_zero,
                        .extent = shadow_extent,
                    },
                    .flags = .{},
                    .layer_count = 1,
                    .p_depth_attachment = &vk.RenderingAttachmentInfo{
                        .clear_value = .{
                            .depth_stencil = .{
                                .depth = 1.0,
                                .stencil = undefined,
                            },
                        },
                        .image_layout = shadow.image_memory_barrier.old_layout,
                        .image_view = shadow.image_view,
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_mode = .{},
                        .resolve_image_layout = .undefined,
                    },
                });
                // cmdBindPipeline(pipelines.shadow);
                // pipeline: vk.Pipeline,
                // pipeline_layout: vk.PipelineLayout,
                // descriptor_set_layout_size: vk.DeviceSize
                // const stage_flags: vk.DeviceSize

                vkd.cmdBindPipeline(
                    command_buffer,
                    .graphics,
                    pipelines.shadow.pipeline,
                );

                vkd.cmdBindDescriptorBuffersEXT(
                    command_buffer,
                    1,
                    @ptrCast(&descriptor_buffer_binding_info),
                );
                descriptor_buffer_binding_info.address += shadow_descriptor_set_layout_size;

                descriptor_buffer_offsets_info.layout = pipelines.shadow.pipeline_layout;
                descriptor_buffer_offsets_info.stage_flags = .{
                    .vertex_bit = true,
                };
                vkd.cmdSetDescriptorBufferOffsets2EXT(
                    command_buffer,
                    &descriptor_buffer_offsets_info,
                );

                vkd.cmdBindVertexBuffers(
                    command_buffer,
                    0,
                    1,
                    @ptrCast(&surfaces.buffer),
                    @ptrCast(&@as(vk.DeviceSize, 0)),
                );

                vkd.cmdDraw(
                    command_buffer,
                    4,
                    surfaces_len,
                    0,
                    0,
                );

                vkd.cmdEndRendering(command_buffer);
            }

            {
                cmdPipelineImageMemoryBarrier(
                    vkd,
                    command_buffer,
                    [_]*RenderTarget{
                        &depth,
                        &geometry,
                    },
                );
            }

            // geometry pass
            {
                vkd.cmdBeginRendering(command_buffer, &vk.RenderingInfo{
                    .color_attachment_count = 1,
                    .view_mask = 0,
                    .render_area = .{
                        .offset = offset_zero,
                        .extent = swapchain_extent,
                    },
                    .flags = .{},
                    .layer_count = 1,
                    .p_depth_attachment = &vk.RenderingAttachmentInfo{
                        .clear_value = .{
                            .depth_stencil = .{
                                .depth = 0.0,
                                .stencil = undefined,
                            },
                        },
                        .image_layout = .depth_attachment_optimal,
                        .image_view = depth.image_view,
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_mode = .{},
                        .resolve_image_layout = .undefined,
                    },
                    .p_color_attachments = @ptrCast(&vk.RenderingAttachmentInfo{
                        .clear_value = .{
                            .color = .{
                                .uint_32 = [_]u32{
                                    0,
                                    0,
                                    0,
                                    0,
                                },
                            },
                        },
                        .image_layout = .color_attachment_optimal,
                        .image_view = geometry.image_view,
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_mode = .{},
                        .resolve_image_layout = .undefined,
                    }),
                });

                vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&vk.Viewport{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @floatFromInt(swapchain_extent.width),
                    .height = @floatFromInt(swapchain_extent.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                }));

                vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&vk.Rect2D{
                    .extent = swapchain_extent,
                    .offset = offset_zero,
                }));

                vkd.cmdBindPipeline(
                    command_buffer,
                    .graphics,
                    pipelines.geometry.pipeline,
                );

                vkd.cmdBindDescriptorBuffersEXT(
                    command_buffer,
                    1,
                    @ptrCast(&descriptor_buffer_binding_info),
                );
                descriptor_buffer_binding_info.address += geometry_descriptor_set_layout_size;

                descriptor_buffer_offsets_info.layout = pipelines.geometry.pipeline_layout;
                descriptor_buffer_offsets_info.stage_flags = .{
                    .vertex_bit = true,
                };
                vkd.cmdSetDescriptorBufferOffsets2EXT(
                    command_buffer,
                    &descriptor_buffer_offsets_info,
                );

                vkd.cmdBindVertexBuffers(
                    command_buffer,
                    0,
                    1,
                    @ptrCast(&surfaces.buffer),
                    @ptrCast(&@as(vk.DeviceSize, 0)),
                );

                vkd.cmdDraw(command_buffer, 4, surfaces_len, 0, 0);

                vkd.cmdEndRendering(command_buffer);
            }

            var swapchain_image_memory_barrier = image_memory_barrier;
            swapchain_image_memory_barrier.image = swapchain_images[swapchain_image_index];
            swapchain_image_memory_barrier.subresource_range.aspect_mask.color_bit = true;

            swapchain_image_memory_barrier.src_stage_mask = .{
                .color_attachment_output_bit = true,
            };
            swapchain_image_memory_barrier.src_access_mask = .{};
            swapchain_image_memory_barrier.old_layout = .undefined;

            swapchain_image_memory_barrier.new_layout = .color_attachment_optimal;
            swapchain_image_memory_barrier.dst_stage_mask = .{
                .color_attachment_output_bit = true,
            };
            swapchain_image_memory_barrier.dst_access_mask = .{
                .color_attachment_write_bit = true,
            };

            {
                const image_memory_barriers = [_]vk.ImageMemoryBarrier2{
                    shadow.image_memory_barrier,
                    depth.image_memory_barrier,
                    geometry.image_memory_barrier,
                    swapchain_image_memory_barrier,
                };

                vkd.cmdPipelineBarrier2(command_buffer, &vk.DependencyInfo{
                    .image_memory_barrier_count = image_memory_barriers.len,
                    .p_image_memory_barriers = &image_memory_barriers,
                });

                shadow.advanceImageMemoryBarrier();
                depth.advanceImageMemoryBarrier();
                geometry.advanceImageMemoryBarrier();
            }

            {
                vkd.cmdBeginRendering(command_buffer, &vk.RenderingInfo{
                    .view_mask = 0,
                    .render_area = .{
                        .offset = offset_zero,
                        .extent = swapchain_extent,
                    },
                    .layer_count = 1,
                    .color_attachment_count = 1,
                    .p_color_attachments = @ptrCast(&vk.RenderingAttachmentInfo{
                        .image_view = swapchain_image_views[swapchain_image_index],
                        .clear_value = vk.ClearValue{
                            .color = .{
                                .float_32 = [4]f32{
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                },
                            },
                        },
                        .image_layout = .color_attachment_optimal,
                        .resolve_mode = .{},
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_image_layout = .undefined,
                    }),
                });

                vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&vk.Viewport{
                    .x = 0.0,
                    .y = 0.0,
                    .width = @floatFromInt(swapchain_extent.width),
                    .height = @floatFromInt(swapchain_extent.height),
                    .min_depth = 0.0,
                    .max_depth = 1.0,
                }));

                vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&vk.Rect2D{
                    .extent = swapchain_extent,
                    .offset = offset_zero,
                }));

                vkd.cmdBindPipeline(
                    command_buffer,
                    .graphics,
                    pipelines.lighting.pipeline,
                );

                vkd.cmdBindDescriptorBuffersEXT(
                    command_buffer,
                    1,
                    @ptrCast(&descriptor_buffer_binding_info),
                );
                // descriptor_buffer_binding_info.address += lighting_pipeline_layout_size;

                descriptor_buffer_offsets_info.layout = pipelines.lighting.pipeline_layout;
                descriptor_buffer_offsets_info.stage_flags = .{
                    .vertex_bit = true,
                    .fragment_bit = true,
                };
                vkd.cmdSetDescriptorBufferOffsets2EXT(
                    command_buffer,
                    &descriptor_buffer_offsets_info,
                );

                vkd.cmdDraw(command_buffer, 3, 1, 0, 0);

                vkd.cmdEndRendering(command_buffer);
            }

            advanceImageBarrier(&swapchain_image_memory_barrier);
            swapchain_image_memory_barrier.new_layout = .present_src_khr;
            swapchain_image_memory_barrier.dst_stage_mask = .{
                .color_attachment_output_bit = true,
            };
            swapchain_image_memory_barrier.dst_access_mask = .{};

            vkd.cmdPipelineBarrier2(
                command_buffer,
                &vk.DependencyInfo{
                    .image_memory_barrier_count = 1,
                    .p_image_memory_barriers = @ptrCast(&swapchain_image_memory_barrier),
                },
            );

            try vkd.endCommandBuffer(command_buffer);
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
                .command_buffer = command_buffer,
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
            const eql = std.mem.eql;
            const global_interface = std.mem.span(global.interface);

            inline for (@typeInfo(RegistryListenerData).@"struct".fields) |data_field| {
                if (eql(u8, global_interface, data_field.name)) {
                    @field(data.*, data_field.name) = registry.bind(
                        global.name,
                        @typeInfo(data_field.type).pointer.child,
                        global.version,
                    ) catch return;
                }
            }
        },
        .global_remove => {},
    }
}

fn relativePointerV1Listener(
    _: *zwp.RelativePointerV1,
    event: zwp.RelativePointerV1.Event,
    camera_relative_pointer_v1: *camera.RelativePointerV1,
) void {
    switch (event) {
        .relative_motion => |relative_motion| {
            camera_relative_pointer_v1.dzx = @intFromEnum(relative_motion.dx);
            camera_relative_pointer_v1.dzy = @intFromEnum(relative_motion.dy);
        },
    }
}

fn keyboardListener(
    _: *wl.Keyboard,
    event: wl.Keyboard.Event,
    camera_keyboard: *camera.Keyboard,
) void {
    // _ = event;
    switch (event) {
        .key => |key| {
            // TODO: provide runtime binding of key values to fields
            const pressed = key.state == .pressed;
            switch (key.key) {
                0x20 => camera_keyboard.positive[0] = pressed,
                0x1e => camera_keyboard.negative[0] = pressed,
                0x2a => camera_keyboard.positive[1] = pressed,
                0x39 => camera_keyboard.negative[1] = pressed,
                0x11 => camera_keyboard.positive[2] = pressed,
                0x1f => camera_keyboard.negative[2] = pressed,
                0x01 => camera_keyboard.locked_pointer_toggle = pressed,
                else => {},
            }
        },
        else => {},
    }
}

fn pointerListener(
    _: *wl.Pointer,
    event: wl.Pointer.Event,
    camera_pointer: *camera.Pointer,
) void {
    switch (event) {
        .axis => |axis| if (axis.axis == .vertical_scroll) {
            camera_pointer.dzx_half_range = @intFromEnum(axis.value);
        },
        .button => |button| {
            const pressed = button.state == .pressed;
            switch (button.button) {
                0x110 => {
                    camera_pointer.destroy = pressed;
                },
                0x111 => {
                    camera_pointer.create = pressed;
                },
                0x114 => {
                    camera_pointer.speedup = pressed;
                },
                0x113 => {
                    camera_pointer.speeddown = pressed;
                },
                else => {},
            }
        },
        else => {},
    }
}

fn xdgToplevelListener(
    _: *xdg.Toplevel,
    event: xdg.Toplevel.Event,
    toplevel_event: *?xdg.Toplevel.Event,
) void {
    toplevel_event.* = event;
}

fn outputListener(
    _: *wl.Output,
    event: wl.Output.Event,
    scale_factor: *i32,
) void {
    switch (event) {
        .scale => |scale| {
            scale_factor.* = scale.factor;
        },
        else => {},
    }
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
            data.p_message_id_name orelse "null",
            color_code,
            data.p_message.?,
        },
    );
    if (message_severity.error_bit_ext) {
        @panic("shit");
    }
    return vk.FALSE;
}

fn findMemoryTypeIndex(
    type_mask: u32,
    required_props: vk.MemoryPropertyFlags,
    props: vk.PhysicalDeviceMemoryProperties,
) !u32 {
    for (props.memory_types[0..props.memory_type_count], 0..) |mem_type, i| {
        if ((type_mask >> @truncate(i)) & 1 == 1 and mem_type.property_flags.contains(required_props)) {
            return @truncate(i);
        }
    }
    return error.MemoryTypeIndex;
}

inline fn pdep(src: anytype, mask: @TypeOf(src)) @TypeOf(src) {
    const T = @TypeOf(src);
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

inline fn pext(src: anytype, mask: @TypeOf(src)) @TypeOf(src) {
    const T = @TypeOf(src);
    if (T != u64 and T != u32 and T != usize) {
        @compileError("pext only implemented for u32 and u64");
    }
    return asm volatile ("pext %[mask], %[src], %[dst]"
        // Map to actual registers:
        : [dst] "=r" (-> T),
        : [src] "r" (src),
          [mask] "r" (mask),
    );
}

fn UnsignedInt(bits: comptime_int) type {
    return @Type(.{ .int = .{
        .bits = bits,
        .signedness = .unsigned,
    } });
}

fn computeSurfacesLen(voxels: [*]const Voxel) u32 {
    var surfaces_len: u32 = 0;
    for (0..volume_len) |voxel_i| {
        if (voxels[voxel_i] == Voxel.Air) {
            continue;
        }
        for (0..directions) |_direction_i| {
            const direction_i: Direction = @truncate(_direction_i);
            const mask = morton_mask << direction_i;
            const coord = pext(voxel_i, mask);
            const cleaned_voxel_i = voxel_i & ~mask;

            if (coord == 0 or
                voxels[cleaned_voxel_i | pdep(coord - 1, mask)] == Voxel.Air)
            {
                surfaces_len += 1;
            }
            if (coord == side_len - 1 or
                voxels[cleaned_voxel_i | pdep(coord + 1, mask)] == Voxel.Air)
            {
                surfaces_len += 1;
            }
        }
    }
    return surfaces_len;
}

fn generateSurfaces(voxels: [*]const Voxel, surfaces: [*]Surface) void {
    var surface_i: usize = 0;
    for (0..volume_len) |voxel_i| {
        if (voxels[voxel_i] == Voxel.Air) {
            continue;
        }

        var combined_indices: CombinedIndices = 0;
        for (0..directions) |_direction_i| {
            const direction_i: Direction = @truncate(_direction_i);
            combined_indices <<= log_side_len;
            combined_indices |= @truncate(pext(voxel_i, morton_mask << direction_i));
        }
        for (0..directions) |_direction| {
            const direction: Direction = @truncate(_direction);
            const mask = morton_mask << direction;
            const coord = pext(voxel_i, mask);

            const cleaned_voxel_i = voxel_i & ~mask;

            {
                var surface = Surface{
                    .direction_sign = 0,
                    .direction_unsigned = direction,
                    .combined_indices = combined_indices,
                    .ambients = undefined,
                };
                if (coord == 0) {
                    surface.ambients = std.math.maxInt(Ambients);
                    surfaces[surface_i] = surface;
                    surface_i += 1;
                } else {
                    const air_voxel_i = cleaned_voxel_i | pdep(coord - 1, mask);
                    if (voxels[air_voxel_i] == Voxel.Air) {
                        surface.ambients = computeAmbients(
                            voxels,
                            air_voxel_i,
                            direction,
                        );
                        surfaces[surface_i] = surface;
                        surface_i += 1;
                    }
                }
            }
            {
                var surface = Surface{
                    .direction_sign = 1,
                    .direction_unsigned = direction,
                    .combined_indices = combined_indices,
                    .ambients = 0,
                };
                if (coord == side_len - 1) {
                    surface.ambients = std.math.maxInt(Ambients);
                    surfaces[surface_i] = surface;
                    surface_i += 1;
                } else {
                    const air_voxel_i = cleaned_voxel_i | pdep(coord + 1, mask);
                    if (voxels[air_voxel_i] == Voxel.Air) {
                        surface.ambients = computeAmbients(
                            voxels,
                            air_voxel_i,
                            direction,
                        );
                        surfaces[surface_i] = surface;
                        surface_i += 1;
                    }
                }
            }
        }
    }
}

fn computeAmbients(
    voxels: [*]const Voxel,
    voxel_i: usize,
    direction_unsigned: Direction,
) Ambients {
    var ambients: Ambients = 0;
    for (0..1 << (directions - 1)) |corner| {
        var ambient: Ambients = 0;
        for (0..directions - 1) |direction_offset| {
            const direction: Direction = @truncate((direction_unsigned + direction_offset + 1) % directions);
            const mask = morton_mask << direction;
            const cleaned_voxel_i = voxel_i & ~mask;
            const coord = pext(voxel_i, mask);
            const bit = (corner >> @truncate(direction_offset)) & 1;

            if ((bit == 0 and (coord == 0 or voxels[cleaned_voxel_i | pdep(coord - 1, mask)] == Voxel.Air)) or
                (bit == 1 and (coord == side_len - 1 or voxels[cleaned_voxel_i | pdep(coord + 1, mask)] == Voxel.Air)))
            {
                ambient += 1;
            }
        }
        if (ambient != 0) diagonal: {
            var sample_voxel_i = voxel_i & (morton_mask << direction_unsigned);
            for (0..directions - 1) |direction_offset| {
                const direction: Direction = @truncate((direction_unsigned + direction_offset + 1) % directions);
                const bit = (corner >> @truncate(direction_offset)) & 1;
                const mask = morton_mask << direction;
                const coord = pext(voxel_i, mask);

                if (bit == 0) {
                    if (coord == 0) {
                        ambient += 1;
                        break :diagonal;
                    }
                    sample_voxel_i |= pdep(coord - 1, mask);
                } else if (bit == 1) {
                    if (coord == side_len - 1) {
                        ambient += 1;
                        break :diagonal;
                    }
                    sample_voxel_i |= pdep(coord + 1, mask);
                }
            }
            if (voxels[sample_voxel_i] == Voxel.Air) {
                ambient += 1;
            }
        }
        ambients |= ambient << @truncate(corner * (directions - 1));
    }
    return ambients;
}

fn computeLightAffine(
    position: [directions]f32,
    light_direction: [directions]f32,
    angles: [directions - 1]f32,
    near: f32,
    far: f32,
    near_scale: [directions - 1]f32,
) [directions][directions + 1]f32 {
    if (directions != 3) {
        unreachable;
    }
    var frustum: [1 << directions][directions]f32 = undefined;

    for (0..1 << (directions - 1)) |corner| {
        for (0..directions - 1) |_direction| {
            const direction: Direction = @truncate(_direction);
            frustum[corner][direction] =
                if ((corner >> direction) & 1 == 1)
                    near_scale[direction]
                else
                    -near_scale[direction];
        }
        frustum[corner][directions - 1] = 0.0;
    }
    for (1 << (directions - 1)..1 << directions) |corner| {
        for (0..directions - 1) |_direction| {
            const direction: Direction = @truncate(_direction);
            frustum[corner][direction] = (far / near) *
                if ((corner >> direction) & 1 == 1)
                    near_scale[direction]
                else
                    -near_scale[direction];
        }
        frustum[corner][directions - 1] = far - near;
    }

    var light_frame: [directions][directions + 1]f32 = undefined;
    for (0..directions) |direction| {
        light_frame[directions - 1][direction] = light_direction[direction];
    }
    // var rand = std.Random.DefaultPrng.init(@bitCast(std.time.microTimestamp()));
    // const random = rand.random();

    for (1..directions) |_new_frame_direction| {
        const new_frame_direction = directions - 1 - _new_frame_direction;
        for (0..directions) |direction| {
            light_frame[new_frame_direction][direction] =
                if (direction == new_frame_direction) 1.0 else 0.0;
        }
        for (0.._new_frame_direction) |_frame_direction| {
            const frame_direction = directions - 1 - _frame_direction;
            const dot: f32 = dotProduct(
                directions,
                &light_frame[new_frame_direction],
                &light_frame[frame_direction],
            );
            for (0..directions) |direction| {
                light_frame[new_frame_direction][direction] -= dot * light_frame[frame_direction][direction];
            }
        }
        normalize(directions, &light_frame[new_frame_direction]);
    }

    var min: [directions]f32 = undefined;
    var max: [directions]f32 = undefined;
    for (0..directions) |direction| {
        min[direction] = std.math.inf(f32);
        max[direction] = -std.math.inf(f32);
    }
    // camera frustum
    for (0..1 << directions) |corner| {
        {
            const cos = frustum[corner][2];
            const sin = frustum[corner][1];

            frustum[corner][2] = @cos(angles[1]) * cos - @sin(angles[1]) * sin;
            frustum[corner][1] = @sin(angles[1]) * cos + @cos(angles[1]) * sin;
        }
        {
            const cos = frustum[corner][2];
            const sin = frustum[corner][0];

            frustum[corner][2] = @cos(angles[0]) * cos - @sin(angles[0]) * sin;
            frustum[corner][0] = @sin(angles[0]) * cos + @cos(angles[0]) * sin;
        }
        for (0..directions) |direction| {
            frustum[corner][direction] += position[direction];
        }

        // apply affine
        for (0..directions) |frame_direction| {
            const dot: f32 = dotProduct(
                directions,
                &frustum[corner],
                &light_frame[frame_direction],
            );
            min[frame_direction] = @min(dot, min[frame_direction]);
            max[frame_direction] = @max(dot, max[frame_direction]);
        }
    }
    // geometry box
    for (0..1 << directions) |corner| {
        var point: [directions]f32 = undefined;
        for (0..directions) |_direction| {
            const direction: Direction = @truncate(_direction);
            point[direction] =
                if ((corner >> direction) & 1 == 1)
                    0.0
                else
                    @floatFromInt(side_len);
        }

        const dot: f32 = dotProduct(
            directions,
            &point,
            &light_frame[directions - 1],
        );
        min[directions - 1] = @min(dot, min[directions - 1]);
        max[directions - 1] = @max(dot, max[directions - 1]);
    }

    light_frame[directions - 1][directions] = -min[directions - 1];
    for (0..directions - 1) |frame_direction| {
        light_frame[frame_direction][directions] = -(max[frame_direction] + min[frame_direction]) / 2.0;
    }
    for (0..directions + 1) |direction| {
        light_frame[directions - 1][direction] /= max[directions - 1] - min[directions - 1];
    }
    for (0..directions - 1) |frame_direction| {
        for (0..directions + 1) |direction| {
            light_frame[frame_direction][direction] /= (max[frame_direction] - min[frame_direction]) / 2.0;
        }
    }

    return light_frame;
}

fn dotProduct(n: usize, lhs: [*]f32, rhs: [*]f32) f32 {
    var dot: f32 = 0.0;
    for (0..n) |i| {
        dot += lhs[i] * rhs[i];
    }
    return dot;
}

fn normalize(n: usize, v: [*]f32) void {
    const dot = dotProduct(n, v, v);
    for (0..n) |i| {
        v[i] /= @sqrt(dot);
    }
}

fn advanceImageBarrier(barrier: *vk.ImageMemoryBarrier2) void {
    barrier.src_stage_mask = barrier.dst_stage_mask;
    barrier.src_access_mask = barrier.dst_access_mask;
    barrier.old_layout = barrier.new_layout;
}

const offset_zero: vk.Offset2D = .{
    .x = 0,
    .y = 0,
};

fn getVkFormat(T: type) vk.Format {
    return switch (@sizeOf(T)) {
        1 => vk.Format.r8_uint,
        2 => vk.Format.r16_uint,
        4 => vk.Format.r32_uint,
        8 => vk.Format.r64_uint,
        else => @compileError("unsupported format"),
    };
}

fn imageViewCreateInfo(
    image: vk.Image,
    format: vk.Format,
) vk.ImageViewCreateInfo {
    const is_depth = format == .d32_sfloat;
    const aspect = vk.ImageAspectFlags{
        .depth_bit = is_depth,
        .color_bit = !is_depth,
    };

    return vk.ImageViewCreateInfo{
        .format = format,
        .image = image,
        .components = .{
            .a = .identity,
            .b = .identity,
            .g = .identity,
            .r = .identity,
        },
        .flags = .{},
        .view_type = .@"2d",
        .subresource_range = .{
            .aspect_mask = aspect,
            .base_array_layer = 0,
            .base_mip_level = 0,
            .layer_count = 1,
            .level_count = 1,
        },
    };
}

const RenderTarget = struct {
    format: vk.Format,
    memory: vk.DeviceMemory = .null_handle,
    image: vk.Image = .null_handle,
    image_view: vk.ImageView = .null_handle,
    image_memory_barrier: vk.ImageMemoryBarrier2 = undefined,

    const Self = @This();

    fn init(
        self: *Self,
        vkd: vk.DeviceWrapper,
        device: vk.Device,
        extent: vk.Extent2D,
        physical_device_memory_properties: vk.PhysicalDeviceMemoryProperties,
    ) !void {
        const is_depth = self.format == .d32_sfloat;
        const aspect = vk.ImageAspectFlags{
            .depth_bit = is_depth,
            .color_bit = !is_depth,
        };

        var memory_requirements: vk.MemoryRequirements2 = .{
            .memory_requirements = undefined,
        };
        const create_info = vk.ImageCreateInfo{
            .format = self.format,
            .usage = .{
                .sampled_bit = true,
                .depth_stencil_attachment_bit = is_depth,
                .color_attachment_bit = !is_depth,
            },
            .array_layers = 1,
            .extent = .{
                .width = extent.width,
                .height = extent.height,
                .depth = 1,
            },
            .flags = .{},
            .image_type = .@"2d",
            .initial_layout = .undefined,
            .mip_levels = 1,
            .queue_family_index_count = 1,
            .p_queue_family_indices = &[_]u32{0},
            .samples = .{
                .@"1_bit" = true,
            },
            .tiling = .optimal,
            .sharing_mode = .exclusive,
        };
        _ = vkd.getDeviceImageMemoryRequirements(
            device,
            &vk.DeviceImageMemoryRequirements{
                .p_create_info = &create_info,
                .plane_aspect = aspect,
            },
            &memory_requirements,
        );
        self.memory = try vkd.allocateMemory(
            device,
            &vk.MemoryAllocateInfo{
                .allocation_size = memory_requirements.memory_requirements.size,
                .memory_type_index = try findMemoryTypeIndex(
                    memory_requirements.memory_requirements.memory_type_bits,
                    vk.MemoryPropertyFlags{
                        .device_local_bit = true,
                    },
                    physical_device_memory_properties,
                ),
            },
            null,
        );
        self.image = try vkd.createImage(
            device,
            &create_info,
            null,
        );
        _ = try vkd.bindImageMemory2(
            device,
            1,
            @ptrCast(&vk.BindImageMemoryInfo{
                .image = self.image,
                .memory = self.memory,
                .memory_offset = 0,
            }),
        );
        self.image_view = try vkd.createImageView(
            device,
            &imageViewCreateInfo(self.image, self.format),
            null,
        );

        self.image_memory_barrier = vk.ImageMemoryBarrier2{
            .src_stage_mask = .{
                .fragment_shader_bit = true,
            },
            .src_access_mask = .{
                .depth_stencil_attachment_read_bit = is_depth,
                .color_attachment_read_bit = !is_depth,
            },
            .old_layout = .undefined,
            .new_layout = if (is_depth)
                vk.ImageLayout.depth_attachment_optimal
            else
                vk.ImageLayout.color_attachment_optimal,
            .dst_stage_mask = .{
                .early_fragment_tests_bit = is_depth,
                .late_fragment_tests_bit = is_depth,
                .color_attachment_output_bit = !is_depth,
            },
            .dst_access_mask = .{
                .depth_stencil_attachment_write_bit = is_depth,
                .color_attachment_write_bit = !is_depth,
            },
            .image = self.image,
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
            .subresource_range = .{
                .aspect_mask = .{
                    .depth_bit = is_depth,
                    .color_bit = !is_depth,
                },
                .base_array_layer = 0,
                .base_mip_level = 0,
                .layer_count = 1,
                .level_count = 1,
            },
        };
    }

    fn advanceImageMemoryBarrier(
        self: *Self,
    ) void {
        const is_depth = self.format == .d32_sfloat;

        const new_layout = self.image_memory_barrier.new_layout;
        self.image_memory_barrier.old_layout = self.image_memory_barrier.new_layout;
        self.image_memory_barrier.src_stage_mask = self.image_memory_barrier.dst_stage_mask;
        self.image_memory_barrier.src_access_mask = self.image_memory_barrier.dst_access_mask;

        if (new_layout == .color_attachment_optimal or
            new_layout == .depth_attachment_optimal)
        {
            self.image_memory_barrier.new_layout =
                if (is_depth)
                    vk.ImageLayout.depth_read_only_optimal
                else
                    vk.ImageLayout.read_only_optimal;
            self.image_memory_barrier.dst_stage_mask = .{
                .fragment_shader_bit = true,
            };
            self.image_memory_barrier.dst_access_mask = .{
                .depth_stencil_attachment_read_bit = is_depth,
                .color_attachment_read_bit = !is_depth,
            };
        } else {
            self.image_memory_barrier.new_layout =
                if (is_depth)
                    vk.ImageLayout.depth_attachment_optimal
                else
                    vk.ImageLayout.color_attachment_optimal;
            self.image_memory_barrier.dst_stage_mask = .{
                .early_fragment_tests_bit = is_depth,
                .late_fragment_tests_bit = is_depth,
                .color_attachment_output_bit = !is_depth,
            };
            self.image_memory_barrier.dst_access_mask = .{
                .depth_stencil_attachment_write_bit = is_depth,
                .color_attachment_write_bit = !is_depth,
            };
        }
    }

    fn destroy(
        self: *Self,
        vkd: vk.DeviceWrapper,
        device: vk.Device,
    ) void {
        vkd.destroyImageView(
            device,
            self.image_view,
            null,
        );
        self.image_view = .null_handle;
        vkd.destroyImage(
            device,
            self.image,
            null,
        );
        self.image = .null_handle;
        vkd.freeMemory(
            device,
            self.memory,
            null,
        );
        self.memory = .null_handle;
    }
};

const Buffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,

    const Self = @This();

    fn init(
        vkd: vk.DeviceWrapper,
        device: vk.Device,
        physical_device_memory_properties: vk.PhysicalDeviceMemoryProperties,
        usage: vk.BufferUsageFlags,
        size: vk.DeviceSize,
        property_flags: vk.MemoryPropertyFlags,
    ) !Self {
        var self: Self = undefined;

        self.size = size;
        self.usage = usage;

        var memory_requirements: vk.MemoryRequirements2 = .{
            .memory_requirements = undefined,
        };
        const create_info = vk.BufferCreateInfo{
            .usage = usage,
            .size = size,
            .flags = .{},
            .sharing_mode = .exclusive,
            .queue_family_index_count = 1,
            .p_queue_family_indices = &[_]u32{0},
        };
        vkd.getDeviceBufferMemoryRequirements(
            device,
            &vk.DeviceBufferMemoryRequirements{
                .p_create_info = &create_info,
            },
            &memory_requirements,
        );

        self.memory = try vkd.allocateMemory(
            device,
            &vk.MemoryAllocateInfo{
                .allocation_size = memory_requirements.memory_requirements.size,
                .memory_type_index = try findMemoryTypeIndex(
                    memory_requirements.memory_requirements.memory_type_bits,
                    property_flags,
                    physical_device_memory_properties,
                ),
                .p_next = if (usage.shader_device_address_bit) &vk.MemoryAllocateFlagsInfo{
                    .flags = .{
                        .device_address_bit = true,
                    },
                    .device_mask = 0,
                } else null,
            },
            null,
        );

        self.buffer = try vkd.createBuffer(
            device,
            &create_info,
            null,
        );

        _ = try vkd.bindBufferMemory2(
            device,
            1,
            @ptrCast(&vk.BindBufferMemoryInfo{
                .buffer = self.buffer,
                .memory = self.memory,
                .memory_offset = 0,
            }),
        );

        return self;
    }

    fn destroy(self: *Self, vkd: vk.DeviceWrapper, device: vk.Device) void {
        vkd.destroyBuffer(
            device,
            self.buffer,
            null,
        );
        self.buffer = .null_handle;
        vkd.freeMemory(
            device,
            self.memory,
            null,
        );
        self.memory = .null_handle;
    }

    fn mapMemory(self: Self, vkd: vk.DeviceWrapper, device: vk.Device) !*anyopaque {
        return (try vkd.mapMemory2(device, &vk.MemoryMapInfo{
            .flags = .{},
            .memory = self.memory,
            .offset = 0,
            .size = self.size,
        })).?;
    }
};

fn cmdPipelineImageMemoryBarrier(
    vkd: vk.DeviceWrapper,
    command_buffer: vk.CommandBuffer,
    targets: anytype,
) void {
    var barriers: [targets.len]vk.ImageMemoryBarrier2 = undefined;
    for (&barriers, targets) |*barrier, target| {
        barrier.* = target.image_memory_barrier;
    }

    vkd.cmdPipelineBarrier2(command_buffer, &vk.DependencyInfo{
        .image_memory_barrier_count = barriers.len,
        .p_image_memory_barriers = &barriers,
    });

    for (targets) |target| {
        target.advanceImageMemoryBarrier();
    }
}

fn DescriptorBufferOffsets(comptime passes: anytype) type {
    const Type = std.builtin.Type;
    const passes_fields = @typeInfo(@TypeOf(passes)).@"struct".fields;
    var out_passes_fields: [passes_fields.len]Type.StructField = undefined;
    for (&out_passes_fields, passes_fields) |*out_passes_field, passes_field| {
        const pass = @field(passes, passes_field.name);
        const pass_fields = @typeInfo(passes_field.type).@"struct".fields;
        var out_pass_fields: [pass_fields.len]std.builtin.Type.StructField = undefined;
        for (&out_pass_fields, pass_fields) |*out_pass_field, pass_field| {
            out_pass_field.* = std.builtin.Type.StructField{
                .alignment = @alignOf(vk.DeviceSize),
                .default_value_ptr = null,
                .is_comptime = false,
                .name = pass_field.name,
                .type = struct {
                    value: vk.DeviceSize,
                    const @"type": vk.DescriptorType = @field(pass, pass_field.name);
                },
            };
        }

        out_passes_field.default_value_ptr = null;
        out_passes_field.is_comptime = false;
        out_passes_field.name = passes_field.name;
        out_passes_field.type = @Type(.{ .@"struct" = std.builtin.Type.Struct{
            .backing_integer = null,
            .decls = &[_]std.builtin.Type.Declaration{},
            .fields = &out_pass_fields,
            .is_tuple = false,
            .layout = .auto,
        } });
        out_passes_field.alignment = @alignOf(out_passes_field.type);
    }

    return @Type(.{ .@"struct" = std.builtin.Type.Struct{
        .backing_integer = null,
        .decls = &[_]std.builtin.Type.Declaration{},
        .fields = &out_passes_fields,
        .is_tuple = false,
        .layout = .auto,
    } });
}

fn initDescriptorBufferOffsets(
    comptime T: type,
    properties: vk.PhysicalDeviceDescriptorBufferPropertiesEXT,
) T {
    var stages: T = undefined;
    const stages_fields = @typeInfo(T).@"struct".fields;

    var offset: vk.DeviceSize = 0;
    inline for (stages_fields) |stages_field| {
        const stage = &@field(stages, stages_field.name);
        const stage_fields = @typeInfo(stages_field.type).@"struct".fields;
        inline for (stage_fields) |stage_field| {
            @field(stage.*, stage_field.name).value = offset;

            offset += descriptorTypeToSize(
                @TypeOf(@field(stage.*, stage_field.name)).type,
                properties,
            );
        }
    }
    return stages;
}

fn Pipelines(comptime passes: anytype) type {
    const Type = std.builtin.Type;
    const passes_fields = @typeInfo(@TypeOf(passes)).@"struct".fields;
    var out_passes_fields: [passes_fields.len]Type.StructField = undefined;
    for (&out_passes_fields, passes_fields) |*out_passes_field, passes_field| {
        out_passes_field.default_value_ptr = null;
        out_passes_field.is_comptime = false;
        out_passes_field.name = passes_field.name;
        out_passes_field.type = struct {
            descriptor_set_layout: vk.DescriptorSetLayout,
            descriptor_memory_offset: vk.DeviceSize,
            pipeline_layout: vk.PipelineLayout,
            pipeline: vk.Pipeline,
            const stage_flags: vk.ShaderStageFlags = @field(passes, passes_field.name);
        };
        out_passes_field.alignment = @alignOf(out_passes_field.type);
    }

    return @Type(.{ .@"struct" = std.builtin.Type.Struct{
        .backing_integer = null,
        .decls = &[_]std.builtin.Type.Declaration{},
        .fields = &out_passes_fields,
        .is_tuple = false,
        .layout = .auto,
    } });
}

fn createPipelines(
    comptime T: type,
    comptime descriptor_types_list: anytype,
    pipeline_create_infos: anytype,
    vkd: vk.DeviceWrapper,
    device: vk.Device,
) !T {
    var pipelines: T = undefined;
    const descriptor_types_list_fields = @typeInfo(@TypeOf(descriptor_types_list)).@"struct".fields;
    const pipelines_fields = @typeInfo(T).@"struct".fields;

    var descriptor_memory_offset: vk.DeviceSize = 0;
    inline for (descriptor_types_list_fields, pipelines_fields) |descriptor_types_list_field, pipelines_field| {
        const descriptor_types = @field(descriptor_types_list, descriptor_types_list_field.name);
        const descriptor_types_fields = @typeInfo(descriptor_types_list_field.type).@"struct".fields;
        var bindings: [descriptor_types_fields.len]vk.DescriptorSetLayoutBinding = undefined;
        inline for (descriptor_types_fields, 0..) |descriptor_types_field, binding| {
            bindings[binding] = vk.DescriptorSetLayoutBinding{
                .binding = binding,
                .descriptor_count = 1,
                .descriptor_type = @field(descriptor_types, descriptor_types_field.name),
                .stage_flags = pipelines_field.type.stage_flags,
            };
        }

        const pipeline = &@field(pipelines, pipelines_field.name);

        pipeline.descriptor_set_layout = try vkd.createDescriptorSetLayout(
            device,
            &vk.DescriptorSetLayoutCreateInfo{
                .binding_count = descriptor_types_fields.len,
                .p_bindings = &bindings,
                .flags = .{
                    .descriptor_buffer_bit_ext = true,
                },
            },
            null,
        );
        pipeline.pipeline_layout = try vkd.createPipelineLayout(
            device,
            &vk.PipelineLayoutCreateInfo{
                .set_layout_count = 1,
                .p_set_layouts = @ptrCast(&pipeline.descriptor_set_layout),
            },
            null,
        );
        var create_info: vk.GraphicsPipelineCreateInfo = @field(
            pipeline_create_infos,
            pipelines_field.name,
        );
        create_info.layout = pipeline.pipeline_layout;

        _ = try vkd.createGraphicsPipelines(
            device,
            .null_handle,
            1,
            @ptrCast(&create_info),
            null,
            @ptrCast(&pipeline.pipeline),
        );

        pipeline.descriptor_memory_offset = descriptor_memory_offset;

        descriptor_memory_offset += vkd.getDescriptorSetLayoutSizeEXT(
            device,
            pipeline.descriptor_set_layout,
        );
    }

    return pipelines;
}

fn destroyPipelines(
    pipelines: anytype,
    vkd: vk.DeviceWrapper,
    device: vk.Device,
) void {
    const pipelines_fields = @typeInfo(@TypeOf(pipelines)).@"struct".fields;
    inline for (pipelines_fields) |pipelines_field| {
        const pipeline = @field(pipelines, pipelines_field.name);
        vkd.destroyDescriptorSetLayout(
            device,
            pipeline.descriptor_set_layout,
            null,
        );
        vkd.destroyPipelineLayout(
            device,
            pipeline.pipeline_layout,
            null,
        );
        vkd.destroyPipeline(
            device,
            pipeline.pipeline,
            null,
        );
    }
}

fn descriptorTypeToSize(
    descriptor_type: vk.DescriptorType,
    properties: vk.PhysicalDeviceDescriptorBufferPropertiesEXT,
) vk.DeviceSize {
    return switch (descriptor_type) {
        vk.DescriptorType.uniform_buffer => properties.uniform_buffer_descriptor_size,
        vk.DescriptorType.combined_image_sampler => properties.combined_image_sampler_descriptor_size,
        else => @panic("unsupported descriptor type"),
    };
}

fn getDescriptor(
    descriptor: anytype,
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    properties: vk.PhysicalDeviceDescriptorBufferPropertiesEXT,
    data: anytype,
    memory_mapped: [*]u8,
) void {
    const @"type": vk.DescriptorType = @TypeOf(descriptor).type;
    if (@TypeOf(data) != switch (@"type") {
        vk.DescriptorType.uniform_buffer => vk.DescriptorAddressInfoEXT,
        vk.DescriptorType.combined_image_sampler => vk.DescriptorImageInfo,
        else => @panic("unsupported type"),
    }) {
        @compileError("mismatched descriptor types");
    }

    vkd.getDescriptorEXT(
        device,
        &vk.DescriptorGetInfoEXT{
            .type = @TypeOf(descriptor).type,
            .data = switch (@"type") {
                vk.DescriptorType.uniform_buffer => vk.DescriptorDataEXT{
                    .p_uniform_buffer = &data,
                },
                vk.DescriptorType.combined_image_sampler => vk.DescriptorDataEXT{
                    .p_combined_image_sampler = &data,
                },
                else => @panic("unsupported type"),
            },
        },
        descriptorTypeToSize(@TypeOf(descriptor).type, properties),
        memory_mapped + descriptor.value,
    );
}
