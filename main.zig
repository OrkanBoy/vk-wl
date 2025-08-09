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

// const vk = @import("vulkan");

const RegistryListenerData = struct {
    compositor: *wl.Compositor,
    wm_base: *xdg.WmBase,
    seat: *wl.Seat,
    output: *wl.Output,
    relative_pointer_manager_v1: *zwp.RelativePointerManagerV1,
    pointer_constraints_v1: *zwp.PointerConstraintsV1,
};

const clamp = std.math.clamp;
const rotate_factor = 1 << 0x10;
const range_rotate_factor = 1 << 0x10;
const rotate_unit: i32 = @intFromFloat(tau * rotate_factor);
const range_rotate_unit: i32 = @intFromFloat(tau * range_rotate_factor);

const camera = struct {
    const Uniform = extern struct {
        position: extern struct {
            x: f32,
            y: f32,
            z: f32,
        },

        z_near: f32,

        czx: f32,
        szx: f32,
        czy: f32,
        szy: f32,

        scale: extern struct {
            x: f32,
            y: f32,
        },
    };

    const Keyboard = struct {
        x_pos: bool,
        x_neg: bool,

        y_pos: bool,
        y_neg: bool,

        z_pos: bool,
        z_neg: bool,

        locked_pointer_toggle: bool,
    };
    const RelativePointerV1 = struct {
        // fixed point 24.8
        dzx: i32,
        dzy: i32,
    };
    const Pointer = struct {
        dzx_half_range: i32,
    };
};

pub fn main() !void {
    const display = try wl.Display.connect(null);
    const registry = try display.getRegistry();
    var registry_listener_data: RegistryListenerData = undefined;
    registry.setListener(*RegistryListenerData, registryListener, &registry_listener_data);
    if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

    const compositor = registry_listener_data.compositor;
    const wm_base = registry_listener_data.wm_base;
    const seat = registry_listener_data.seat;
    const output = registry_listener_data.output;
    const relative_pointer_manager_v1 = registry_listener_data.relative_pointer_manager_v1;
    const pointer_constraints_v1 = registry_listener_data.pointer_constraints_v1;

    const keyboard = try seat.getKeyboard();
    defer keyboard.destroy();

    var camera_keyboard: camera.Keyboard = .{
        .x_pos = false,
        .x_neg = false,

        .y_pos = false,
        .y_neg = false,

        .z_pos = false,
        .z_neg = false,

        .locked_pointer_toggle = false,
    };
    keyboard.setListener(
        *camera.Keyboard,
        keyboardListener,
        &camera_keyboard,
    );

    const pointer = try seat.getPointer();
    defer pointer.destroy();

    var camera_pointer: camera.Pointer = .{
        .dzx_half_range = 0,
    };
    var camera_zx_half_range: i32 = range_rotate_unit / 6;
    pointer.setListener(
        *camera.Pointer,
        pointerListener,
        &camera_pointer,
    );

    var camera_relative_pointer_v1: camera.RelativePointerV1 = .{
        .dzx = 0,
        .dzy = 0,
    };
    var camera_zx: i32 = 0;
    var camera_zy: i32 = 0;
    const relative_pointer_v1 = try relative_pointer_manager_v1.getRelativePointer(pointer);
    relative_pointer_v1.setListener(
        *camera.RelativePointerV1,
        relativePointerV1Listener,
        &camera_relative_pointer_v1,
    );

    const wl_surface = try compositor.createSurface();
    defer wl_surface.destroy();

    var scale_factor: i32 = 0;
    output.setListener(*i32, outputListener, &scale_factor);

    const xdg_surface = try wm_base.getXdgSurface(wl_surface);
    defer xdg_surface.destroy();

    const toplevel = try xdg_surface.getToplevel();
    defer toplevel.destroy();

    var toplevel_event: ?xdg.Toplevel.Event = null;
    toplevel.setListener(
        *?xdg.Toplevel.Event,
        xdgToplevelListener,
        &toplevel_event,
    );

    var locked_pointer: ?*zwp.LockedPointerV1 = null;

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
    var vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
    defer vkd.destroyDevice(device, null);

    // vkd.dispatch.vkCmdSetDescriptorBufferOffsets2EXT = @ptrCast(vki.dispatch.vkGetDeviceProcAddr.?(
    //     device,
    //     "vkCmdSetDescriptorBufferOffsets2",
    // ));
    //
    // vkd.dispatch.vkCmdSetDescriptorBufferOffsetsEXT = @ptrCast(vki.dispatch.vkGetDeviceProcAddr.?(
    //     device,
    //     "vkCmdSetDescriptorBufferOffsets",
    // ));
    //
    // print("{x}\n{x}\n", .{
    //     @intFromPtr(vkd.dispatch.vkCmdSetDescriptorBufferOffsetsEXT),
    //     @intFromPtr(vkd.dispatch.vkCmdSetDescriptorBufferOffsets2EXT),
    // });
    //
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

    var depth_memory: vk.DeviceMemory = .null_handle;
    defer vkd.freeMemory(
        device,
        depth_memory,
        null,
    );
    var depth_image: vk.Image = .null_handle;
    defer vkd.destroyImage(
        device,
        depth_image,
        null,
    );
    var depth_image_view: vk.ImageView = .null_handle;
    defer vkd.destroyImageView(
        device,
        depth_image_view,
        null,
    );

    var camera_memory_requirements: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
    };
    const camera_buffer_create_info = vk.BufferCreateInfo{
        .usage = .{
            .uniform_buffer_bit = true,
            .shader_device_address_bit = true,
        },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 1,
        .p_queue_family_indices = &[_]u32{0},
        .size = @sizeOf(camera.Uniform),
    };
    vkd.getDeviceBufferMemoryRequirements(
        device,
        &vk.DeviceBufferMemoryRequirements{
            .p_create_info = &camera_buffer_create_info,
        },
        &camera_memory_requirements,
    );

    const camera_memory = try vkd.allocateMemory(
        device,
        &vk.MemoryAllocateInfo{
            .allocation_size = camera_memory_requirements.memory_requirements.size,
            .memory_type_index = try findMemoryTypeIndex(
                camera_memory_requirements.memory_requirements.memory_type_bits,
                vk.MemoryPropertyFlags{
                    .host_coherent_bit = true,
                    .host_visible_bit = true,
                },
                physical_device_memory_properties,
            ),
            .p_next = &vk.MemoryAllocateFlagsInfo{
                .flags = .{
                    .device_address_bit = true,
                },
                .device_mask = 0,
            },
        },
        null,
    );
    defer vkd.freeMemory(device, camera_memory, null);

    const camera_buffer = try vkd.createBuffer(
        device,
        &camera_buffer_create_info,
        null,
    );
    defer vkd.destroyBuffer(
        device,
        camera_buffer,
        null,
    );
    _ = try vkd.bindBufferMemory2(
        device,
        1,
        @ptrCast(&vk.BindBufferMemoryInfo{
            .buffer = camera_buffer,
            .memory = camera_memory,
            .memory_offset = 0,
        }),
    );

    const camera_uniform: *camera.Uniform = @ptrCast(@alignCast(try vkd.mapMemory2(device, &vk.MemoryMapInfo{
        .flags = .{},
        .memory = camera_memory,
        .offset = 0,
        .size = @sizeOf(camera.Uniform),
    })));

    const descriptor_set_layout_bindings = [_]vk.DescriptorSetLayoutBinding{
        vk.DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .stage_flags = .{ .vertex_bit = true },
        },
    };

    const descriptor_set_layout = try vkd.createDescriptorSetLayout(
        device,
        &vk.DescriptorSetLayoutCreateInfo{
            .flags = .{
                .descriptor_buffer_bit_ext = true,
            },
            .binding_count = descriptor_set_layout_bindings.len,
            .p_bindings = &descriptor_set_layout_bindings,
        },
        null,
    );
    defer vkd.destroyDescriptorSetLayout(
        device,
        descriptor_set_layout,
        null,
    );

    var descriptor_memory_requirements: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
    };
    const descriptor_buffer_create_info = vk.BufferCreateInfo{
        .usage = .{
            .resource_descriptor_buffer_bit_ext = true,
            .shader_device_address_bit = true,
        },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 1,
        .p_queue_family_indices = &[_]u32{0},
        .size = vkd.getDescriptorSetLayoutSizeEXT(
            device,
            descriptor_set_layout,
        ),
    };
    vkd.getDeviceBufferMemoryRequirements(
        device,
        &vk.DeviceBufferMemoryRequirements{
            .p_create_info = &descriptor_buffer_create_info,
        },
        &descriptor_memory_requirements,
    );

    const descriptor_memory = try vkd.allocateMemory(
        device,
        &vk.MemoryAllocateInfo{
            .allocation_size = descriptor_memory_requirements.memory_requirements.size,
            .memory_type_index = try findMemoryTypeIndex(
                descriptor_memory_requirements.memory_requirements.memory_type_bits,
                vk.MemoryPropertyFlags{
                    .host_visible_bit = true,
                    // TODO:
                },
                physical_device_memory_properties,
            ),
            .p_next = &vk.MemoryAllocateFlagsInfo{
                .flags = .{
                    .device_address_bit = true,
                },
                .device_mask = 0,
            },
        },
        null,
    );
    defer vkd.freeMemory(device, descriptor_memory, null);

    const descriptor_buffer = try vkd.createBuffer(
        device,
        &descriptor_buffer_create_info,
        null,
    );
    defer vkd.destroyBuffer(device, descriptor_buffer, null);

    const descriptor_buffer_mapped = (try vkd.mapMemory2(
        device,
        &vk.MemoryMapInfo{
            .flags = .{},
            .memory = descriptor_memory,
            .offset = 0,
            .size = descriptor_buffer_create_info.size,
        },
    )).?;

    _ = try vkd.bindBufferMemory2(
        device,
        1,
        @ptrCast(&vk.BindBufferMemoryInfo{
            .buffer = descriptor_buffer,
            .memory = descriptor_memory,
            .memory_offset = 0,
        }),
    );
    // print("{x}\n{x}\n", .{ descriptor_set_layout_size, descriptor_set_layout_binding_offset });

    const descriptor_buffer_device_address = vkd.getBufferDeviceAddress(
        device,
        &vk.BufferDeviceAddressInfo{
            .buffer = descriptor_buffer,
        },
    );
    vkd.getDescriptorEXT(
        device,
        &vk.DescriptorGetInfoEXT{
            .type = .uniform_buffer,
            .data = .{
                .p_uniform_buffer = &vk.DescriptorAddressInfoEXT{
                    .format = .undefined,
                    .range = @sizeOf(camera.Uniform),
                    .address = vkd.getBufferDeviceAddress(
                        device,
                        &vk.BufferDeviceAddressInfo{
                            .buffer = camera_buffer,
                        },
                    ),
                },
            },
        },
        physical_device_descriptor_buffer_properties.uniform_buffer_descriptor_size,
        @ptrCast(descriptor_buffer_mapped),
    );

    const Voxel = enum {
        Air,
        Stone,
    };
    const dimensions = 3;

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

    const allocator = std.heap.page_allocator;
    var voxels = try allocator.alloc(Voxel, volume_len);
    defer allocator.free(voxels);

    const LogUsize = std.math.Log2Int(usize);
    for (0..volume_len) |voxel_i| {
        var sum: usize = 0;
        for (0..dimensions) |_dimension_i| {
            const dimension_i: LogUsize = @truncate(_dimension_i);
            const coord = pext(usize, voxel_i, morton_mask << dimension_i);
            sum += coord * coord;
        }
        const max_sum = 1 << (2 * log_side_len);
        voxels[voxel_i] = if (sum > max_sum or sum < max_sum / 3) Voxel.Stone else Voxel.Air;
    }
    voxels[0] = Voxel.Stone;

    var surfaces_len: u32 = 0;
    for (0..volume_len) |voxel_i| {
        if (voxels[voxel_i] == Voxel.Air) {
            continue;
        }
        for (0..dimensions) |_dimension_i| {
            const dimension_i: LogUsize = @truncate(_dimension_i);
            const mask = morton_mask << dimension_i;
            const coord = pext(usize, voxel_i, mask);
            const cleaned_voxel_i = voxel_i & ~mask;

            if (coord == 0 or
                voxels[cleaned_voxel_i | pdep(usize, coord - 1, mask)] == Voxel.Air)
            {
                surfaces_len += 1;
            }
            if (coord == side_len - 1 or
                voxels[cleaned_voxel_i | pdep(usize, coord + 1, mask)] == Voxel.Air)
            {
                surfaces_len += 1;
            }
        }
    }

    // const PartialVoxelId = UnsignedInt(log_side_len);
    const Surface = UnsignedInt((dimensions * log_side_len) + (1 + log2_int_ceil(usize, dimensions)));
    // const Surface = struct {
    //     position: [dimensions]u8,
    //     direction: u8,
    // };

    const surface_transfer_buffer_create_info: vk.BufferCreateInfo = .{
        .flags = .{},
        .usage = .{
            .transfer_src_bit = true,
        },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 1,
        .p_queue_family_indices = &[_]u32{0},
        .size = @sizeOf(Surface) * surfaces_len,
    };
    var surface_transfer_memory_requirements: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
    };
    vkd.getDeviceBufferMemoryRequirements(
        device,
        &vk.DeviceBufferMemoryRequirements{
            .p_create_info = &surface_transfer_buffer_create_info,
        },
        &surface_transfer_memory_requirements,
    );
    const surface_transfer_memory = try vkd.allocateMemory(
        device,
        &vk.MemoryAllocateInfo{
            .allocation_size = surface_transfer_memory_requirements.memory_requirements.size,
            .memory_type_index = try findMemoryTypeIndex(
                surface_transfer_memory_requirements.memory_requirements.memory_type_bits,
                vk.MemoryPropertyFlags{
                    .host_visible_bit = true,
                },
                physical_device_memory_properties,
            ),
        },
        null,
    );
    defer vkd.freeMemory(
        device,
        surface_transfer_memory,
        null,
    );
    const surface_transfer_buffer = try vkd.createBuffer(
        device,
        &surface_transfer_buffer_create_info,
        null,
    );
    defer vkd.destroyBuffer(
        device,
        surface_transfer_buffer,
        null,
    );

    _ = try vkd.bindBufferMemory2(
        device,
        1,
        @ptrCast(&vk.BindBufferMemoryInfo{
            .buffer = surface_transfer_buffer,
            .memory = surface_transfer_memory,
            .memory_offset = 0,
        }),
    );
    const surface_transfer_memory_mapped = @as([*]Surface, @ptrCast(@alignCast(try vkd.mapMemory2(
        device,
        &vk.MemoryMapInfo{
            .memory = surface_transfer_memory,
            .size = surface_transfer_buffer_create_info.size,
            .flags = .{},
            .offset = 0,
        },
    ))))[0..surfaces_len];

    var surface_i: usize = 0;
    for (0..volume_len) |voxel_i| {
        if (voxels[voxel_i] == Voxel.Air) {
            continue;
        }

        var surface: Surface = 0;
        // var surface: Surface = undefined;
        for (0..dimensions) |_dimension_i| {
            const dimension_i: LogUsize = @truncate(_dimension_i);
            // surface[dimension_i] = pext(usize, voxel_i, morton_mask << dimension_i);
            surface <<= log_side_len;
            surface |= @truncate(pext(usize, voxel_i, morton_mask << dimension_i));
        }
        surface <<= 1 + comptime log2_int_ceil(usize, dimensions);
        for (0..dimensions) |_dimension_i| {
            const dimension_i: LogUsize = @truncate(_dimension_i);
            const mask = morton_mask << dimension_i;
            const coord = pext(usize, voxel_i, mask);

            const cleaned_voxel_i = voxel_i & ~mask;

            if (coord == 0 or
                voxels[cleaned_voxel_i | pdep(usize, coord - 1, mask)] == Voxel.Air)
            {
                surface_transfer_memory_mapped[surface_i] = surface | (dimension_i << 1) | 0;
                // surface_transfer_memory_mapped[surface_i] = 1;
                surface_i += 1;
            }
            if (coord == side_len - 1 or
                voxels[cleaned_voxel_i | pdep(usize, coord + 1, mask)] == Voxel.Air)
            {
                surface_transfer_memory_mapped[surface_i] = surface | (dimension_i << 1) | 1;

                // surface_transfer_memory_mapped[surface_i] = 1;
                surface_i += 1;
            }
        }
    }

    var surface_buffer_create_info: vk.BufferCreateInfo = .{
        .flags = .{},
        .queue_family_index_count = 1,
        .p_queue_family_indices = &[_]u32{0},
        .sharing_mode = .exclusive,
        .usage = .{
            .vertex_buffer_bit = true,
            .transfer_dst_bit = true,
        },
        .size = surface_transfer_buffer_create_info.size,
    };
    var surface_memory_requirements: vk.MemoryRequirements2 = .{
        .memory_requirements = undefined,
    };
    vkd.getDeviceBufferMemoryRequirements(
        device,
        &vk.DeviceBufferMemoryRequirements{
            .p_create_info = &surface_buffer_create_info,
        },
        &surface_memory_requirements,
    );
    const surface_memory = try vkd.allocateMemory(
        device,
        &vk.MemoryAllocateInfo{
            .allocation_size = surface_memory_requirements.memory_requirements.size,
            .memory_type_index = try findMemoryTypeIndex(
                surface_memory_requirements.memory_requirements.memory_type_bits,
                vk.MemoryPropertyFlags{
                    .device_local_bit = true,
                },
                physical_device_memory_properties,
            ),
        },
        null,
    );
    defer vkd.freeMemory(
        device,
        surface_memory,
        null,
    );
    const surface_buffer = try vkd.createBuffer(
        device,
        &surface_buffer_create_info,
        null,
    );
    defer vkd.destroyBuffer(
        device,
        surface_buffer,
        null,
    );

    _ = try vkd.bindBufferMemory2(
        device,
        1,
        @ptrCast(&vk.BindBufferMemoryInfo{
            .buffer = surface_buffer,
            .memory = surface_memory,
            .memory_offset = 0,
        }),
    );
    var transfer_surface: bool = true;

    const pipeline_layout = try vkd.createPipelineLayout(
        device,
        &vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{
                descriptor_set_layout,
            },
        },
        null,
    );
    defer vkd.destroyPipelineLayout(device, pipeline_layout, null);

    var pipeline: vk.Pipeline = undefined;
    const shaders_spv align(@alignOf(u32)) = @embedFile("shaders").*;

    const vertex_shader = try vkd.createShaderModule(
        device,
        &vk.ShaderModuleCreateInfo{
            .p_code = @ptrCast(&shaders_spv),
            .code_size = shaders_spv.len,
        },
        null,
    );
    defer vkd.destroyShaderModule(device, vertex_shader, null);

    const fragment_shader = try vkd.createShaderModule(
        device,
        &vk.ShaderModuleCreateInfo{
            .p_code = @ptrCast(&shaders_spv),
            .code_size = shaders_spv.len,
        },
        null,
    );
    defer vkd.destroyShaderModule(device, fragment_shader, null);

    _ = try vkd.createGraphicsPipelines(
        device,
        .null_handle,
        1,
        @ptrCast(&vk.GraphicsPipelineCreateInfo{
            .flags = .{
                .descriptor_buffer_bit_ext = true,
            },
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = undefined,
            .layout = pipeline_layout,
            .p_input_assembly_state = &vk.PipelineInputAssemblyStateCreateInfo{
                .topology = .triangle_strip,
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
                    .format = .r32_uint,
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
                    .p_name = "vertexMain",
                    .stage = .{ .vertex_bit = true },
                    .module = vertex_shader,
                },
                vk.PipelineShaderStageCreateInfo{
                    .p_name = "fragmentMain",
                    .stage = .{ .fragment_bit = true },
                    .module = fragment_shader,
                },
            },
            .render_pass = .null_handle,
            .subpass = undefined,
            .p_next = &vk.PipelineRenderingCreateInfo{
                .color_attachment_count = 1,
                .p_color_attachment_formats = @ptrCast(&surface_format.format),
                .depth_attachment_format = .d32_sfloat,
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

    camera_uniform.position.x = 8.0;
    camera_uniform.position.y = side_len - 2.0;
    camera_uniform.position.z = 8.0;

    while (true) {
        if (display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

        const new_time = nanoTimestamp();
        const delta_time = new_time - time;
        time = new_time;

        {
            if (camera_keyboard.locked_pointer_toggle) {
                if (locked_pointer == null) {
                    locked_pointer = try pointer_constraints_v1.lockPointer(
                        wl_surface,
                        pointer,
                        null,
                        .persistent,
                    );
                } else {
                    locked_pointer.?.destroy();
                    locked_pointer = null;
                }
                camera_keyboard.locked_pointer_toggle = false;
            }

            if (locked_pointer != null) {
                camera_zx += camera_relative_pointer_v1.dzx;

                camera_zy += camera_relative_pointer_v1.dzy;
                camera_zy = clamp(
                    camera_zy,
                    -rotate_unit / 5,
                    rotate_unit / 5,
                );

                camera_zx_half_range += camera_pointer.dzx_half_range;

                camera_zx_half_range = clamp(
                    camera_zx_half_range,
                    range_rotate_unit / 8,
                    range_rotate_unit / 5,
                );

                camera_relative_pointer_v1.dzx = 0;
                camera_relative_pointer_v1.dzy = 0;
                camera_pointer.dzx_half_range = 0;
            }

            var zx: f32 = @floatFromInt(camera_zx);
            zx /= rotate_factor;

            var zy: f32 = @floatFromInt(camera_zy);
            zy /= rotate_factor;

            var zx_half_range: f32 = @floatFromInt(camera_zx_half_range);
            zx_half_range /= range_rotate_factor;

            if (locked_pointer != null) {
                const dt: f32 = @as(f32, @floatFromInt(delta_time)) / 1_000_000_000.0;

                const dz = @cos(zx) * @cos(zy) * dt;
                const dx = @sin(zx) * @cos(zy) * dt;

                if (camera_keyboard.x_pos and !camera_keyboard.x_neg) {
                    camera_uniform.position.z -= dx;
                    camera_uniform.position.x += dz;
                } else if (!camera_keyboard.x_pos and camera_keyboard.x_neg) {
                    camera_uniform.position.z += dx;
                    camera_uniform.position.x -= dz;
                }

                if (camera_keyboard.y_pos and !camera_keyboard.y_neg) {
                    camera_uniform.position.y += dt;
                } else if (!camera_keyboard.y_pos and camera_keyboard.y_neg) {
                    camera_uniform.position.y -= dt;
                }

                if (camera_keyboard.z_pos and !camera_keyboard.z_neg) {
                    camera_uniform.position.z += dz;
                    camera_uniform.position.x += dx;
                } else if (!camera_keyboard.z_pos and camera_keyboard.z_neg) {
                    camera_uniform.position.z -= dz;
                    camera_uniform.position.x -= dx;
                }
            }

            const camera_size_x: f32 = 1.0;
            camera_uniform.z_near = camera_size_x / @tan(zx_half_range);
            camera_uniform.scale.x = camera_uniform.z_near / camera_size_x;
            camera_uniform.scale.y = camera_uniform.scale.x *
                @as(f32, @floatFromInt(swapchain_extent.width)) /
                @as(f32, @floatFromInt(swapchain_extent.height));

            camera_uniform.czx = @cos(zx);
            camera_uniform.szx = @sin(zx);

            camera_uniform.czy = @cos(zy);
            camera_uniform.szy = @sin(zy);
        }

        var image_memory_barriers: [2]vk.ImageMemoryBarrier2 = undefined;
        var image_memory_barriers_len: u8 = 0;
        var init_swapchain: bool = false;

        if (toplevel_event) |event| event_handle: {
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
                        vkd.destroyImageView(
                            device,
                            depth_image_view,
                            null,
                        );
                        vkd.destroyImage(
                            device,
                            depth_image,
                            null,
                        );
                        vkd.freeMemory(
                            device,
                            depth_memory,
                            null,
                        );
                    }
                    var depth_memory_requirements: vk.MemoryRequirements2 = .{
                        .memory_requirements = undefined,
                    };
                    const depth_image_create_info: vk.ImageCreateInfo = .{
                        .array_layers = 1,
                        .extent = .{
                            .width = swapchain_extent.width,
                            .height = swapchain_extent.height,
                            .depth = 1,
                        },
                        .format = .d32_sfloat,
                        .flags = .{},
                        .image_type = .@"2d",
                        .initial_layout = .undefined,
                        .mip_levels = 1,
                        .usage = .{
                            .depth_stencil_attachment_bit = true,
                        },
                        .queue_family_index_count = 1,
                        .p_queue_family_indices = &[_]u32{0},
                        .sharing_mode = .exclusive,
                        .samples = .{
                            .@"1_bit" = true,
                        },
                        .tiling = .optimal,
                    };

                    vkd.getDeviceImageMemoryRequirements(
                        device,
                        &vk.DeviceImageMemoryRequirements{
                            .p_create_info = &depth_image_create_info,
                            .plane_aspect = .{
                                .depth_bit = true,
                            },
                        },
                        &depth_memory_requirements,
                    );

                    depth_memory = try vkd.allocateMemory(
                        device,
                        &vk.MemoryAllocateInfo{
                            .allocation_size = depth_memory_requirements.memory_requirements.size,
                            .memory_type_index = try findMemoryTypeIndex(
                                depth_memory_requirements.memory_requirements.memory_type_bits,
                                vk.MemoryPropertyFlags{
                                    .device_local_bit = true,
                                },
                                physical_device_memory_properties,
                            ),
                        },
                        null,
                    );

                    depth_image = try vkd.createImage(
                        device,
                        &depth_image_create_info,
                        null,
                    );
                    _ = try vkd.bindImageMemory2(
                        device,
                        1,
                        &[_]vk.BindImageMemoryInfo{
                            vk.BindImageMemoryInfo{
                                .image = depth_image,
                                .memory = depth_memory,
                                .memory_offset = 0,
                            },
                        },
                    );

                    depth_image_view = try vkd.createImageView(
                        device,
                        &vk.ImageViewCreateInfo{
                            .components = .{
                                .r = .identity,
                                .b = .identity,
                                .g = .identity,
                                .a = .identity,
                            },
                            .flags = .{},
                            .format = depth_image_create_info.format,
                            .image = depth_image,
                            .view_type = .@"2d",
                            .subresource_range = .{
                                .aspect_mask = .{
                                    .depth_bit = true,
                                },
                                .base_array_layer = 0,
                                .base_mip_level = 0,
                                .layer_count = 1,
                                .level_count = 1,
                            },
                        },
                        null,
                    );

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
                },
                else => {},
            }
            toplevel_event = null;
        }

        image_memory_barriers[image_memory_barriers_len] = if (init_swapchain)
            vk.ImageMemoryBarrier2{
                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
                .src_stage_mask = .{},
                .src_access_mask = .{},
                .dst_stage_mask = .{
                    .early_fragment_tests_bit = true,
                    .late_fragment_tests_bit = true,
                },
                .dst_access_mask = .{
                    .depth_stencil_attachment_write_bit = true,
                },
                .image = depth_image,
                .old_layout = .undefined,
                .new_layout = .depth_attachment_optimal,
                .subresource_range = .{
                    .aspect_mask = .{
                        .depth_bit = true,
                    },
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                    .layer_count = 1,
                    .level_count = 1,
                },
            }
        else
            vk.ImageMemoryBarrier2{
                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
                .src_stage_mask = .{
                    .early_fragment_tests_bit = true,
                    .late_fragment_tests_bit = true,
                },
                .src_access_mask = .{
                    .depth_stencil_attachment_write_bit = true,
                },
                .dst_stage_mask = .{
                    .early_fragment_tests_bit = true,
                    .late_fragment_tests_bit = true,
                },
                .dst_access_mask = .{
                    .depth_stencil_attachment_write_bit = true,
                },
                .image = depth_image,
                .old_layout = .depth_attachment_optimal,
                .new_layout = .depth_attachment_optimal,
                .subresource_range = .{
                    .aspect_mask = .{
                        .depth_bit = true,
                    },
                    .base_array_layer = 0,
                    .base_mip_level = 0,
                    .layer_count = 1,
                    .level_count = 1,
                },
            };
        image_memory_barriers_len += 1;

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
        {
            try vkd.beginCommandBuffer(command_buffers[frame], &.{ .flags = .{ .one_time_submit_bit = true } });

            image_memory_barriers[image_memory_barriers_len] = vk.ImageMemoryBarrier2{
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
                .dst_stage_mask = .{
                    .color_attachment_output_bit = true,
                },
                .dst_access_mask = .{
                    .color_attachment_write_bit = true,
                },
            };
            image_memory_barriers_len += 1;

            if (transfer_surface) {
                vkd.cmdCopyBuffer2(
                    command_buffers[frame],
                    &vk.CopyBufferInfo2{
                        .src_buffer = surface_transfer_buffer,
                        .dst_buffer = surface_buffer,
                        .region_count = 1,
                        .p_regions = @ptrCast(&vk.BufferCopy2{
                            .src_offset = 0,
                            .dst_offset = 0,
                            .size = surface_buffer_create_info.size,
                        }),
                    },
                );
            }

            vkd.cmdPipelineBarrier2(
                command_buffers[frame],
                &vk.DependencyInfo{
                    .image_memory_barrier_count = image_memory_barriers_len,
                    .p_image_memory_barriers = &image_memory_barriers,
                    // .buffer_memory_barrier_count = if (transfer_surface) 1 else 0,
                    // .p_buffer_memory_barriers = @ptrCast(&vk.BufferMemoryBarrier2{
                    //     .buffer = surface_buffer,
                    //     .offset = 0,
                    //     .size = surface_buffer_create_info.size,
                    //     .src_stage_mask = .{
                    //         .copy_bit = true,
                    //     },
                    //     .dst_stage_mask = .{
                    //         .vertex_input_bit = true,
                    //     },
                    //     .src_access_mask = .{
                    //         .transfer_write_bit = true,
                    //     },
                    //     .dst_access_mask = .{
                    //         .vertex_attribute_read_bit = true,
                    //     },
                    //     .src_queue_family_index = 0,
                    //     .dst_queue_family_index = 0,
                    // }),
                },
            );

            transfer_surface = false;

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
                        .image_layout = .rendering_local_read,
                        .resolve_mode = .{},
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_image_layout = .undefined,
                    }),
                    .p_depth_attachment = &vk.RenderingAttachmentInfo{
                        .image_view = depth_image_view,
                        .clear_value = vk.ClearValue{
                            .depth_stencil = .{
                                .depth = 0.0,
                                .stencil = 0,
                            },
                        },
                        .image_layout = .rendering_local_read,
                        .resolve_mode = .{},
                        .load_op = .clear,
                        .store_op = .store,
                        .resolve_image_layout = .undefined,
                    },
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
                    .offset = .{
                        .x = 0,
                        .y = 0,
                    },
                }));

                vkd.cmdBindPipeline(command_buffers[frame], .graphics, pipeline);

                vkd.cmdBindDescriptorBuffersEXT(
                    command_buffers[frame],
                    1,
                    @ptrCast(&vk.DescriptorBufferBindingInfoEXT{
                        .address = descriptor_buffer_device_address,
                        .usage = descriptor_buffer_create_info.usage,
                    }),
                );
                vkd.cmdSetDescriptorBufferOffsets2EXT(
                    command_buffers[frame],
                    &vk.SetDescriptorBufferOffsetsInfoEXT{
                        .first_set = 0,
                        .set_count = 1,
                        .layout = pipeline_layout,
                        .stage_flags = .{
                            .vertex_bit = true,
                        },
                        .p_buffer_indices = &[_]u32{0},
                        .p_offsets = &[_]vk.DeviceSize{0},
                    },
                );

                vkd.cmdBindVertexBuffers(
                    command_buffers[frame],
                    0,
                    1,
                    @ptrCast(&surface_buffer),
                    @ptrCast(&@as(vk.DeviceSize, 0)),
                );

                vkd.cmdDraw(command_buffers[frame], 4, surfaces_len, 0, 0);

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
            const eql = std.mem.eql;
            const global_interface = std.mem.span(global.interface);

            if (eql(u8, global_interface, "wl_compositor")) {
                data.compositor = registry.bind(
                    global.name,
                    wl.Compositor,
                    global.version,
                ) catch return;
            } else if (eql(u8, global_interface, "xdg_wm_base")) {
                data.wm_base = registry.bind(
                    global.name,
                    xdg.WmBase,
                    global.version,
                ) catch return;
            } else if (eql(u8, global_interface, "wl_seat")) {
                data.seat = registry.bind(
                    global.name,
                    wl.Seat,
                    global.version,
                ) catch return;
            } else if (eql(u8, global_interface, "wl_output")) {
                data.output = registry.bind(
                    global.name,
                    wl.Output,
                    global.version,
                ) catch return;
            } else if (eql(u8, global_interface, "zwp_relative_pointer_manager_v1")) {
                data.relative_pointer_manager_v1 = registry.bind(
                    global.name,
                    zwp.RelativePointerManagerV1,
                    global.version,
                ) catch return;
            } else if (eql(u8, global_interface, "zwp_pointer_constraints_v1")) {
                data.pointer_constraints_v1 = registry.bind(
                    global.name,
                    zwp.PointerConstraintsV1,
                    global.version,
                ) catch return;
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
                0x20 => camera_keyboard.x_pos = pressed,
                0x1e => camera_keyboard.x_neg = pressed,
                0x2a => camera_keyboard.y_pos = pressed,
                0x39 => camera_keyboard.y_neg = pressed,
                0x11 => camera_keyboard.z_pos = pressed,
                0x1f => camera_keyboard.z_neg = pressed,
                0x01 => camera_keyboard.locked_pointer_toggle = pressed,
                else => {},
            }
        },
        else => {},
    }
    // print("{?}\n\n", .{camera_keyboard.*});
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
        else => {},
    }

    // switch (event) {
    // }

    // print("{?}\n", .{event});
    // pointer_event.* = event;
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

inline fn pext(comptime T: type, src: T, mask: T) T {
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
