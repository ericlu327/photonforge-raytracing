use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use wgpu::*;
use winit::{dpi::PhysicalSize, window::Window};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct CameraUBO {
    origin: [f32; 3],
    _pad0: f32,
    dir: [f32; 3],
    _pad1: f32,
    right: [f32; 3],
    _pad2: f32,
    up: [f32; 3],
    _pad3: f32,
    img_size: [u32; 2],
    frame_index: u32,
    max_bounce: u32,
}

pub enum Movement {
    Forward,
    Backward,
    Left,
    Right,
    Up,
    Down,
}

pub struct Renderer<'w> {
    surface: Surface<'w>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,

    size: PhysicalSize<u32>,

    // accumulation ping-pong
    accum_a: Texture,
    accum_b: Texture,
    accum_a_view_storage: TextureView,
    accum_b_view_storage: TextureView,
    accum_a_view_sample: TextureView,
    accum_b_view_sample: TextureView,

    sampler: Sampler,

    compute_pipeline: ComputePipeline,
    blit_pipeline: RenderPipeline,

    // split layouts/bind groups to avoid usage conflicts
    compute_bind_layout: BindGroupLayout,
    blit_bind_layout: BindGroupLayout,

    compute_bind_a: BindGroup,
    compute_bind_b: BindGroup,
    blit_bind_a: BindGroup,
    blit_bind_b: BindGroup,

    camera_buf: Buffer,

    frame_index: u32,
    use_a_as_src: bool,

    cam_pos: Vec3,
    yaw: f32,
    pitch: f32,
    move_delta: Vec3,
    fov_y_radians: f32,
}

impl<'w> Renderer<'w> {
    pub async fn new(window: &'w Window) -> Result<Self> {
        let size = window.inner_size();

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        let surface = instance.create_surface(window)?;
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No GPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("device"),
                    // allow RGBA16F as storage on native
                    required_features: Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    required_limits: Limits::default().using_resolution(adapter.limits()),
                },
                None,
            )
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    TextureFormat::Bgra8Unorm
                        | TextureFormat::Bgra8UnormSrgb
                        | TextureFormat::Rgba8Unorm
                        | TextureFormat::Rgba8UnormSrgb
                )
            })
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            desired_maximum_frame_latency: 3,
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let (accum_a, a_storage, a_sample) = Self::make_accum(&device, size);
        let (accum_b, b_storage, b_sample) = Self::make_accum(&device, size);

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("linear sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        // --- Bind group layouts (split) ---
        // Compute: UBO + storage in + storage out
        let compute_bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("compute layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Blit: UBO + sampled tex + sampler
        let blit_bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("blit layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT | ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let camera_buf = device.create_buffer(&BufferDescriptor {
            label: Some("camera ubo"),
            size: std::mem::size_of::<CameraUBO>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Bind groups (compute) ---
        let compute_bind_a = device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_a (in=a, out=b)"),
            layout: &compute_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&a_storage),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&b_storage),
                },
            ],
        });

        let compute_bind_b = device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_b (in=b, out=a)"),
            layout: &compute_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&b_storage),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&a_storage),
                },
            ],
        });

        // --- Bind groups (blit) ---
        let blit_bind_a = device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit_bind_a (sample=a)"),
            layout: &blit_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&a_sample),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        let blit_bind_b = device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit_bind_b (sample=b)"),
            layout: &blit_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&b_sample),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        // --- Shaders & pipelines ---
        let compute_mod = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("compute"),
            source: ShaderSource::Wgsl(include_str!("../shaders/compute.wgsl").into()),
        });
        let blit_mod = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("blit"),
            source: ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
        });

        let pipeline_layout_compute = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute pipeline layout"),
            bind_group_layouts: &[&compute_bind_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("compute pipeline"),
            layout: Some(&pipeline_layout_compute),
            module: &compute_mod,
            entry_point: "cs_main",
        });

        let pipeline_layout_blit = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("blit pipeline layout"),
            bind_group_layouts: &[&blit_bind_layout],
            push_constant_ranges: &[],
        });
        let blit_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("blit pipeline"),
            layout: Some(&pipeline_layout_blit),
            vertex: VertexState {
                module: &blit_mod,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &blit_mod,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let mut r = Self {
            surface,
            device,
            queue,
            config,
            size,
            accum_a,
            accum_b,
            accum_a_view_storage: a_storage,
            accum_b_view_storage: b_storage,
            accum_a_view_sample: a_sample,
            accum_b_view_sample: b_sample,
            sampler,
            compute_pipeline,
            blit_pipeline,
            compute_bind_layout,
            blit_bind_layout,
            compute_bind_a,
            compute_bind_b,
            blit_bind_a,
            blit_bind_b,
            camera_buf,
            frame_index: 0,
            use_a_as_src: true,
            cam_pos: Vec3::new(0.0, 1.0, 4.0),
            yaw: 0.0,
            pitch: 0.0,
            move_delta: Vec3::ZERO,
            fov_y_radians: 45f32.to_radians(),
        };

        r.update_camera();
        Ok(r)
    }

    fn make_accum(device: &Device, size: PhysicalSize<u32>) -> (Texture, TextureView, TextureView) {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("accum tex"),
            size: Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::STORAGE_BINDING
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let storage_view =
            tex.create_view(&TextureViewDescriptor { label: Some("accum storage"), ..Default::default() });
        let sample_view =
            tex.create_view(&TextureViewDescriptor { label: Some("accum sample"), ..Default::default() });
        (tex, storage_view, sample_view)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);

        // Recreate accum textures and bind groups
        let (accum_a, a_storage, a_sample) = Self::make_accum(&self.device, self.size);
        let (accum_b, b_storage, b_sample) = Self::make_accum(&self.device, self.size);
        self.accum_a = accum_a;
        self.accum_b = accum_b;
        self.accum_a_view_storage = a_storage;
        self.accum_b_view_storage = b_storage;
        self.accum_a_view_sample = a_sample;
        self.accum_b_view_sample = b_sample;

        // Rebuild bind groups after resize
        self.compute_bind_a = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_a"),
            layout: &self.compute_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.accum_a_view_storage),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.accum_b_view_storage),
                },
            ],
        });
        self.compute_bind_b = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("compute_bind_b"),
            layout: &self.compute_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.accum_b_view_storage),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.accum_a_view_storage),
                },
            ],
        });
        self.blit_bind_a = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit_bind_a"),
            layout: &self.blit_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.accum_a_view_sample),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        self.blit_bind_b = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit_bind_b"),
            layout: &self.blit_bind_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&self.accum_b_view_sample),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.reset_accum();
        self.update_camera();
    }

    pub fn queue_movement(&mut self, m: Movement) {
        let amt = 0.2;
        match m {
            Movement::Forward => self.move_delta.z -= amt,
            Movement::Backward => self.move_delta.z += amt,
            Movement::Left => self.move_delta.x -= amt,
            Movement::Right => self.move_delta.x += amt,
            Movement::Up => self.move_delta.y += amt,
            Movement::Down => self.move_delta.y -= amt,
        }
        self.cam_pos += self.view_basis() * self.move_delta;
        self.move_delta = Vec3::ZERO;
        self.reset_accum();
        self.update_camera();
    }

    pub fn on_mouse_delta(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.0025;
        self.yaw -= dx * sensitivity;
        self.pitch -= dy * sensitivity;
        self.pitch = self.pitch.clamp(-1.5, 1.5);
        self.reset_accum();
        self.update_camera();
    }

    pub fn on_scroll(&mut self, delta: f32) {
        self.fov_y_radians = (self.fov_y_radians - delta * 0.02)
            .clamp(10f32.to_radians(), 90f32.to_radians());
        self.reset_accum();
        self.update_camera();
    }

    pub fn reset_accum(&mut self) {
        self.frame_index = 0;
    }

    fn view_basis(&self) -> Mat3 {
        let dir = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize();
        let right = dir.cross(Vec3::Y).normalize();
        let up = right.cross(dir).normalize();
        Mat3::from_cols(right, up, -dir)
    }

    fn update_camera(&mut self) {
        let basis = self.view_basis();
        let dir = -(basis.col(2));
        let right = basis.col(0);
        let up = basis.col(1);

        let ubo = CameraUBO {
            origin: self.cam_pos.to_array(),
            _pad0: 0.0,
            dir: dir.to_array(),
            _pad1: 0.0,
            right: right.to_array(),
            _pad2: 0.0,
            up: up.to_array(),
            _pad3: 0.0,
            img_size: [self.size.width.max(1), self.size.height.max(1)],
            frame_index: self.frame_index,
            max_bounce: 2,
        };
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&ubo));
    }

    pub fn render(&mut self) -> Result<()> {
        // compute
        let mut encoder =
            self.device
                .create_command_encoder(&CommandEncoderDescriptor { label: Some("encoder") });

        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("trace pass"),
                ..Default::default()
            });
            cpass.set_pipeline(&self.compute_pipeline);
            let cbind = if self.use_a_as_src {
                &self.compute_bind_a
            } else {
                &self.compute_bind_b
            };
            cpass.set_bind_group(0, cbind, &[]);
            let gx = (self.size.width + 7) / 8;
            let gy = (self.size.height + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // present
        let surface_tex = self.surface.get_current_texture()?;
        let view = surface_tex
            .texture
            .create_view(&TextureViewDescriptor::default());
        {
            let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("blit pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.blit_pipeline);
            let bbind = if self.use_a_as_src {
                &self.blit_bind_b // we just wrote B, sample B
            } else {
                &self.blit_bind_a // we just wrote A, sample A
            };
            rpass.set_bind_group(0, bbind, &[]);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        surface_tex.present();

        self.frame_index = self.frame_index.wrapping_add(1);
        self.use_a_as_src = !self.use_a_as_src;
        self.update_camera();
        Ok(())
    }
}
