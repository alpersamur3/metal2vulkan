//! A depth texture made under `writeSwizzleEnabled` and written as a render target alone
//! (`TextureDescriptor::write_swizzle_enabled`: Apple's OpenGL ES asks for its depth buffer so,
//! `ShaderRead | RenderTarget` with a swizzle of `(Red, Red, Red, One)`), rendered into and then
//! sampled through its swizzle.
//!
//! Each case clears a 4x4 texture's depth to 1 and draws one triangle over the whole target
//! whose depth falls across it, under a `Less` test that writes (the owned fixture
//! `cfb_sample_depth`: each vertex where buffer 0 holds it, its depth included); copies the
//! texture's depth to a buffer, its bytes, which no swizzle touches; and samples it at each
//! texel's centre through a nearest runtime sampler in a fragment function (the owned fixture
//! `tob_renders`' `tob_sample`, into an `RGBA32Float` target). The stored depths must be the
//! triangle's plane, and each sample the texture's swizzle over the channels a depth texture's
//! read presents, `(d, d, d, 1)` (Metal's read of a depth texture through `texture2d<float>`,
//! measured: `runtime_lane::depth_as_color`): `(d, d, d, 1)` through `(Red, Red, Red, One)`, and
//! `(1, 0, d, d)` through `(One, Zero, Red, Red)`, a swizzle that shows the sampled view takes it.
//! Not measured on Metal: its render through a write swizzle of a depth texture, and its answer
//! for the second swizzle; both are the runtime's rule. The formats are `Depth16Unorm`,
//! `Depth32Float` and `Depth32Float_Stencil8` (its depth plane copied with
//! `DepthFromDepthStencil`); a `Depth32Float` one of 4 samples is resolved (`Sample0`) into a
//! single-sampled one made alike, which is sampled so.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::error::{Error, Invalid, Problem, RuntimeGap};
use crate::{
    BlitOption, Buffer, BufferTextureProblem, ClearColor, CommandQueue, Device, DeviceKind,
    DeviceOptions, HeapDescriptor, HeapType, Library, LoadAction, Origin, PipelineOptions, Region,
    RenderPassDescriptor, RenderPipelineState, ResourceOptions, SamplerState, ShaderCacheOptions,
    Size, StoreAction, Texture, TextureDescriptor, TextureProblem, TextureSwizzleChannels,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{
    CompareFunction, PixelFormat, PrimitiveType, SamplerMinMagFilter, TextureSwizzle, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The textures' side.
const SIDE: u64 = 4;

/// The triangle over the whole target, `(x, y, z, w)` in clip space: its depth `0.2 + 0.1 (x + 1)
/// + 0.15 (y + 1)`, from 0.2625 to 0.6375 over the target's texel centres, each row and column
/// its own.
const PLANE: [[f32; 4]; 3] = [
    [-1.0, -1.0, 0.2, 1.0],
    [3.0, -1.0, 0.6, 1.0],
    [-1.0, 3.0, 0.8, 1.0],
];

/// The multisampled case's triangle: one depth, 0.4, at every sample, which `Sample0` resolves.
const FLAT: [[f32; 4]; 3] = [
    [-1.0, -1.0, 0.4, 1.0],
    [3.0, -1.0, 0.4, 1.0],
    [-1.0, 3.0, 0.4, 1.0],
];

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-write-swizzled-depth-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The owned engine fixture `stem` (`llvm-as`), held by `device`.
fn library(device: &Device, stem: &str) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../metal2vulkan-validation/fixtures/public/engine/{stem}.ll"
    ));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    device.new_library_with_compile_library(library)
}

fn function(library: &Library, name: &str) -> StageSource {
    StageSource::Function(
        library
            .new_function_with_name(name)
            .unwrap()
            .function_descriptor(),
    )
}

/// `(Red, Red, Red, One)` and `(One, Zero, Red, Red)`.
fn rrr1() -> TextureSwizzleChannels {
    use TextureSwizzle as S;
    TextureSwizzleChannels::new(S::Red, S::Red, S::Red, S::One)
}

fn one_zero_rr() -> TextureSwizzleChannels {
    use TextureSwizzle as S;
    TextureSwizzleChannels::new(S::One, S::Zero, S::Red, S::Red)
}

/// The test's device, queue, fixtures and sampler.
struct Lane {
    device: Device,
    queue: CommandQueue,
    depths: Library,
    renders: Library,
    nearest: SamplerState,
}

impl Lane {
    fn new(device: Device) -> Lane {
        let mut nearest = SamplerDescriptor::new();
        nearest.min_filter = SamplerMinMagFilter::Nearest;
        nearest.mag_filter = SamplerMinMagFilter::Nearest;
        Lane {
            queue: device.new_command_queue().unwrap(),
            depths: library(&device, "cfb_sample_depth"),
            renders: library(&device, "tob_renders"),
            nearest: device.new_sampler_state(&nearest).unwrap(),
            device,
        }
    }

    /// A private 4x4 texture of `pf` and `samples` under `writeSwizzleEnabled`, of `usage` and
    /// `swizzle`.
    fn texture(
        &self,
        pf: PixelFormat,
        samples: u64,
        usage: TextureUsage,
        swizzle: TextureSwizzleChannels,
    ) -> Texture {
        let mut d = TextureDescriptor::texture_2d(pf, SIDE, SIDE, false);
        if samples > 1 {
            d.texture_type = TextureType::Type2DMultisample;
            d.sample_count = samples;
        }
        d.usage = usage;
        d.swizzle = swizzle;
        d.write_swizzle_enabled = true;
        d.resource_options = ResourceOptions::private();
        self.device.new_texture(&d).unwrap()
    }

    /// One pass over `depth` (resolved into `resolve` where given): its depth cleared to 1, then
    /// `triangle` drawn under a `Less` test that writes.
    fn render(&self, depth: &Texture, resolve: Option<&Texture>, triangle: &[[f32; 4]; 3]) {
        let mut pipeline =
            RenderPipelineDescriptor::new(function(&self.depths, "cfb_depth_vertex"));
        pipeline.fragment = Some(function(&self.depths, "cfb_depth_none"));
        pipeline.depth_format = Some(depth.pixel_format());
        pipeline.raster_sample_count = depth.sample_count() as u32;
        let pipeline = self
            .device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap();
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Less;
        state.depth_write_enabled = true;
        let state = self.device.new_depth_stencil_state(&state).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(depth.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.clear_depth = 1.0;
        attachment.store_action = match resolve {
            Some(_) => StoreAction::MultisampleResolve,
            None => StoreAction::Store,
        };
        attachment.resolve_texture = resolve.cloned();
        let vertices = floats(&self.device, triangle.iter().flatten().copied());
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_depth_stencil_state(&state).unwrap();
        encoder.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 3, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
    }

    /// Each texel's depth of single-sampled `texture`, row by row, copied out as its bytes: a
    /// `Depth16Unorm` code over 65535, a 32-bit float depth as it is.
    fn stored(&self, texture: &Texture) -> Vec<f32> {
        let pf = texture.pixel_format();
        let (per, option) = match pf {
            PixelFormat::Depth16Unorm => (2, BlitOption::None),
            PixelFormat::Depth32FloatStencil8 => (4, BlitOption::DepthFromDepthStencil),
            _ => (4, BlitOption::None),
        };
        let length = SIDE * SIDE * per;
        let buffer = self
            .device
            .new_buffer(length, ResourceOptions::shared())
            .unwrap();
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            Size::new(SIDE, SIDE, 1),
            &buffer,
            0,
            SIDE * per,
            length,
            option,
        )
        .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let bytes = read(&buffer, length as usize);
        match per {
            2 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from(u16::from_le_bytes(*c)) / 65535.0)
                .collect(),
            _ => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
        }
    }

    /// The texels of a 4x4 `RGBA32Float` target that `tob_sample` draws over the whole of,
    /// sampling `texture` through the nearest sampler at each pixel's centre, row by row.
    fn sampled(&self, texture: &Texture) -> Vec<[f32; 4]> {
        let mut pipeline = RenderPipelineDescriptor::new(function(&self.renders, "tob_rect"));
        pipeline.fragment = Some(function(&self.renders, "tob_sample"));
        pipeline.color_attachments[0] =
            Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA32Float));
        let pipeline: RenderPipelineState = self
            .device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap();
        let mut d = TextureDescriptor::texture_2d(PixelFormat::RGBA32Float, SIDE, SIDE, false);
        d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        d.resource_options = ResourceOptions::shared();
        let target = self.device.new_texture(&d).unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(target.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_color = ClearColor::new(0.0, 0.0, 0.0, 0.0);
        let rect = floats(&self.device, [-1.0, -1.0, 1.0, 1.0]);
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.set_render_pipeline_state(&pipeline).unwrap();
        encoder.set_vertex_buffer(Some(&rect), 0, 0).unwrap();
        encoder.set_fragment_texture(Some(texture), 0).unwrap();
        encoder
            .set_fragment_sampler_state(Some(&self.nearest), 0)
            .unwrap();
        encoder
            .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None);
        let mut bytes = vec![0u8; (16 * SIDE * SIDE) as usize];
        target
            .get_bytes(
                &mut bytes,
                16 * SIDE,
                0,
                Region::new_2d(0, 0, SIDE, SIDE),
                0,
                0,
            )
            .unwrap();
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|c| {
                std::array::from_fn(|k| f32::from_le_bytes(c[4 * k..4 * k + 4].try_into().unwrap()))
            })
            .collect()
    }
}

/// A `Shared` buffer of `values`' bytes.
fn floats(device: &Device, values: impl IntoIterator<Item = f32>) -> Buffer {
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// The first `len` bytes of a shared buffer.
fn read(buffer: &Buffer, len: usize) -> Vec<u8> {
    let contents = buffer.contents().expect("a shared buffer");
    // SAFETY: the buffer holds at least `len` bytes at `contents`, alive with it; no command
    // buffer uses it now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), len) }.to_vec()
}

/// [`PLANE`]'s depth at the centre of texel `(x, y)` (row 0 at clip-space y = +1).
fn plane_depth(x: u64, y: u64) -> f64 {
    let side = SIDE as f64;
    let cx = -1.0 + (2.0 * x as f64 + 1.0) / side;
    let cy = 1.0 - (2.0 * y as f64 + 1.0) / side;
    0.2 + 0.1 * (cx + 1.0) + 0.15 * (cy + 1.0)
}

/// Whether `got` is `swizzle` over `(d, d, d, 1)`, `d` within 2^-20 of `depth`.
fn swizzled(got: [f32; 4], depth: f32, swizzle: TextureSwizzleChannels) -> bool {
    let near = |v: f32| (f64::from(v) - f64::from(depth)).abs() <= 1.0 / f64::from(1u32 << 20);
    [swizzle.red, swizzle.green, swizzle.blue, swizzle.alpha]
        .iter()
        .zip(got)
        .all(|(source, v)| match source {
            TextureSwizzle::Zero => v == 0.0,
            TextureSwizzle::One | TextureSwizzle::Alpha => v == 1.0,
            _ => near(v),
        })
}

/// The creators of a `Depth32Float` texture made under `writeSwizzleEnabled` and written as a
/// render target alone: the device's and a heap's make it, the shared creator stops it
/// (`SwizzleWithWrites`, as Metal's validation does), the heap size query answers it, and a
/// buffer's stops it as it stops any depth texture (`NotLinear`); written by shaders it is the
/// runtime's gap. The validation crate's lane asks the same under the validation layer
/// (`runtime_lane::texture_creators`).
#[test]
fn a_write_swizzled_depth_target_is_made_where_metal_makes_it() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let heap = device
        .new_heap(&HeapDescriptor::new(
            HeapType::Automatic,
            1 << 20,
            ResourceOptions::private(),
        ))
        .unwrap();
    let buffer = device
        .new_buffer(1 << 16, ResourceOptions::shared())
        .unwrap();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let mut d = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 16, 16, false);
    d.usage = usage;
    d.swizzle = rrr1();
    d.write_swizzle_enabled = true;
    d.resource_options = ResourceOptions::private();
    let made = device.new_texture(&d).unwrap();
    assert_eq!((made.usage(), made.swizzle()), (usage, rrr1()));
    let heaped = heap.new_texture(&d).unwrap();
    assert_eq!((heaped.usage(), heaped.swizzle()), (usage, rrr1()));
    assert!(device.heap_texture_size_and_align(&d).is_ok());
    let stop = Problem::Texture(TextureProblem::SwizzleWithWrites);
    assert_eq!(
        device.new_shared_texture(&d).err(),
        Some(Error::Invalid(Invalid::new("new_shared_texture", stop)))
    );
    let over = TextureDescriptor {
        resource_options: ResourceOptions::shared(),
        ..d
    };
    let not_linear = BufferTextureProblem::NotLinear {
        pixel_format: PixelFormat::Depth32Float,
    };
    assert_eq!(
        buffer.new_texture(&over, 0, 64).err(),
        Some(Error::Invalid(Invalid::new(
            "new_texture",
            Problem::BufferTexture(not_linear)
        )))
    );
    let written = TextureDescriptor {
        usage: TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET,
        ..d
    };
    let gap = Some(Error::NotImplemented(RuntimeGap::WriteSwizzle));
    assert_eq!(device.new_texture(&written).err(), gap);
    assert_eq!(heap.new_texture(&written).err(), gap);
    assert!(device.heap_texture_size_and_align(&written).is_ok());
}

/// A depth texture made under `writeSwizzleEnabled` and written as a render target alone is made,
/// reports its descriptor's usage, renders depth through its attachment, keeps its bytes, and
/// samples through its swizzle (the module docs).
#[test]
fn a_write_swizzled_depth_target_renders_and_samples_through_its_swizzle() {
    let scratch = Scratch::new();
    let lane = Lane::new(device(&scratch.0));
    let name = lane.device.name().to_owned();
    let usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let mut wrong = Vec::new();
    let mut compared = 0usize;
    for pf in [
        PixelFormat::Depth16Unorm,
        PixelFormat::Depth32Float,
        PixelFormat::Depth32FloatStencil8,
    ] {
        for swizzle in [rrr1(), one_zero_rr()] {
            let texture = lane.texture(pf, 1, usage, swizzle);
            assert_eq!(texture.usage(), usage, "{pf:?}");
            assert_eq!(texture.swizzle(), swizzle, "{pf:?}");
            lane.render(&texture, None, &PLANE);
            let stored = lane.stored(&texture);
            let samples = lane.sampled(&texture);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let i = (SIDE * y + x) as usize;
                    compared += 1;
                    let at = format!("{pf:?} {swizzle:?} at ({x}, {y})");
                    if (f64::from(stored[i]) - plane_depth(x, y)).abs() > 1e-4 {
                        wrong.push(format!("{at}: stored {}", stored[i]));
                    }
                    if !swizzled(samples[i], stored[i], swizzle) {
                        wrong.push(format!("{at}: sampled {:?} of {}", samples[i], stored[i]));
                    }
                }
            }
        }
    }
    // Multisampled, resolved into a single-sampled one made alike, which is sampled.
    let ms = lane.texture(
        PixelFormat::Depth32Float,
        4,
        TextureUsage::RENDER_TARGET,
        rrr1(),
    );
    let resolved = lane.texture(PixelFormat::Depth32Float, 1, usage, rrr1());
    lane.render(&ms, Some(&resolved), &FLAT);
    let stored = lane.stored(&resolved);
    let samples = lane.sampled(&resolved);
    for (i, (&depth, &sample)) in stored.iter().zip(&samples).enumerate() {
        compared += 1;
        if (depth - 0.4).abs() > 1e-6 || !swizzled(sample, depth, rrr1()) {
            wrong.push(format!(
                "resolve texel {i}: stored {depth}, sampled {sample:?}"
            ));
        }
    }
    eprintln!(
        "{name}: {compared} texels of write-swizzled depth targets compared, {} wrong",
        wrong.len()
    );
    assert!(wrong.is_empty(), "{name}: {wrong:#?}");
}
