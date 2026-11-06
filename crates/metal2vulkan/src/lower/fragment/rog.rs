//! Raster order groups: a fragment function that names a raster order group on any parameter,
//! output it reads, or argument-buffer member runs its whole body inside a fragment-interlock
//! critical section.
//!
//! Metal orders a raster order group's accesses between fragments that cover the same pixel (the
//! same sample at sample rate). AIR states the group only as `air.raster_order_group` metadata on
//! buffers, textures, color inputs and outputs, and imageblock members; the accesses themselves
//! are ordinary loads and stores. The lowering therefore orders the whole body, which is a
//! superset of every group's accesses (groups only reduce the ordering Metal must provide):
//!
//! ```text
//! main:  OpBeginInvocationInterlockEXT
//!        OpFunctionCall body
//!        OpEndInvocationInterlockEXT
//!        if discarded { OpKill | OpTerminateInvocation }      // only when a discard can set it
//!        OpReturn
//! ```
//!
//! No terminator may run between `Begin` and `End`, so inside the body a terminate-form discard
//! is `discarded ← true; OpReturn`, a predicated one only sets `discarded`, and the body's returns
//! do not kill; `main` kills after `End`. A demoted invocation stays in the body and ends it as
//! usual.
//!
//! **Color outputs alone**. A group's color output that the function does not read (no color input
//! of its attachment) asks for no ordering beyond what a render target already has: Vulkan performs
//! a fragment's color writes and blending per sample in rasterization order, as Metal does. Such an
//! output alone does not interlock the function, so a function whose groups hold nothing else
//! compiles on every device, the ones without an interlock included.
//!
//! **Color inputs**. On a device without the pixel interlock but with rasterization-order color
//! attachment access (Turnip), a group's color input and color output are ordered by that access:
//! an input attachment's read sees every earlier overlapping fragment's write. Those alone do not
//! interlock the function there ([`needs_interlock`]). A group on any other memory still asks for
//! it, and such a device has none: the function runs unordered and says so ([`mode`]), which only
//! overlapping fragments of a draw can tell from Metal; iOS' own compositor needs these functions.
//!
//! **Mode** ([`mode`]). `PixelInterlockOrderedEXT` where the profile has
//! `fragmentShaderPixelInterlock`: it orders every invocation covering the pixel, which covers
//! Metal's pixel- and sample-rate ordering. `SampleInterlockOrderedEXT` orders only invocations
//! covering the same sample, which equals Metal's ordering only when the module itself runs at
//! sample rate (it reads `SampleId` or has a sample-interpolated input); a pixel-rate module on a
//! multisampled target would leave two fragments covering disjoint samples of one pixel unordered,
//! so there it is not used. Otherwise the function is refused
//! `Unsupported{RasterOrderGroups, AnyOf[[FragmentShaderPixelInterlock], ...]}`.

use std::collections::BTreeSet;

use crate::air::abi::{MemberKind, OutputRole, ParamRole, StructRef};
use crate::air::Entry;
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::select::SelectCx;
use crate::lower::stage_io::IoObject;
use crate::pipeline::sym::Stage;
use crate::spirv::ir::{ExecMode, IoDir, Sampling};

/// The raster order group a parameter role names, if any.
fn role_group(role: &ParamRole) -> Option<u32> {
    match role {
        ParamRole::Buffer(a) | ParamRole::ArgumentBuffer(a) | ParamRole::Threadgroup(a) => {
            a.raster_order_group
        }
        ParamRole::Texture(t) => t.raster_order_group,
        ParamRole::ColorInput {
            raster_order_group, ..
        } => *raster_order_group,
        _ => None,
    }
}

/// The layouts parameter role `role` names directly: a buffer's pointee, an imageblock's cell
/// and master.
fn role_layouts(role: &ParamRole) -> [Option<StructRef>; 2] {
    match role {
        ParamRole::Buffer(a) | ParamRole::ArgumentBuffer(a) | ParamRole::Threadgroup(a) => {
            [a.layout, None]
        }
        ParamRole::Imageblock(ib) => [ib.layout, ib.master],
        _ => [None, None],
    }
}

/// Whether entry `e` names a raster order group that needs the interlock: on a parameter, a
/// color output whose attachment the function reads (a color input of the same attachment), or
/// an argument-buffer or imageblock member of a layout its parameters and outputs reach.
/// `e.structs` holds every layout of the module, so a layout only another function names does not
/// count. A group's color output the function does not read needs none ([module docs](self)).
pub(super) fn declared(e: &Entry) -> bool {
    declared_on(e, true)
}

/// Whether the function's groups need the interlock on this device: [`declared`], without the
/// color inputs and outputs on a device whose rasterization-order access orders them
/// ([module docs](self)).
pub(super) fn needs_interlock(cx: &SelectCx<'_>) -> bool {
    let f = cx.profile.fields().fragment;
    declared_on(cx.entry, f.pixel_interlock || !f.rasterization_order_color_access)
}

/// [`declared`], counting a group on a color input or output only when `colors`.
fn declared_on(e: &Entry, colors: bool) -> bool {
    let params = e.ep.params.iter().any(|b| {
        (colors || !matches!(b.role, ParamRole::ColorInput { .. })) && role_group(&b.role).is_some()
    });
    let read = |attachment| {
        e.ep.params
            .iter()
            .any(|b| matches!(b.role, ParamRole::ColorInput { index, .. } if index == attachment))
    };
    let outputs = colors
        && e.ep.outputs.iter().any(|o| match o.role {
            OutputRole::RenderTarget {
                attachment,
                raster_order_group: Some(_),
                ..
            } => read(attachment),
            _ => false,
        });
    if params || outputs {
        return true;
    }
    let mut stack: Vec<StructRef> =
        e.ep.params
            .iter()
            .flat_map(|b| role_layouts(&b.role))
            .chain(e.ep.outputs.iter().flat_map(|o| match o.role {
                OutputRole::ImageblockData(ib) => [ib.layout, ib.master],
                _ => [None, None],
            }))
            .flatten()
            .collect();
    let mut seen = BTreeSet::new();
    while let Some(r) = stack.pop() {
        if !seen.insert(r) {
            continue;
        }
        let Some(s) = usize::try_from(r.0).ok().and_then(|i| e.structs.get(i)) else {
            continue;
        };
        for m in &s.members {
            if m.raster_order_group.is_some() {
                return true;
            }
            stack.extend(m.layout);
            if let MemberKind::Indirect(a) = &m.kind {
                if role_group(&a.role).is_some() {
                    return true;
                }
                stack.extend(role_layouts(&a.role).into_iter().flatten());
            }
        }
    }
    false
}

/// Whether the planned stage interface makes the module run at sample rate: it reads `SampleId`
/// or declares a sample-interpolated input.
fn sample_rate(objects: &[IoObject]) -> bool {
    objects.iter().any(|o| match o {
        IoObject::Builtin { builtin, dir, .. } => {
            *builtin == crate::spv::BuiltIn::SampleId && *dir == IoDir::Input
        }
        IoObject::User { dir, deco, .. } => {
            *dir == IoDir::Input && deco.sampling == Sampling::Sample
        }
    })
}

/// What the first group of entry `e` that needs the interlock is on, for the line [`mode`] prints.
fn group_memory(e: &Entry) -> &'static str {
    let named = e.ep.params.iter().find_map(|b| match &b.role {
        ParamRole::Buffer(a) if a.raster_order_group.is_some() => Some("a buffer"),
        ParamRole::ArgumentBuffer(a) if a.raster_order_group.is_some() => {
            Some("an argument buffer")
        }
        ParamRole::Threadgroup(a) if a.raster_order_group.is_some() => Some("threadgroup memory"),
        ParamRole::Texture(t) if t.raster_order_group.is_some() => Some("a texture"),
        _ => None,
    });
    named.unwrap_or("a member of a layout")
}

/// The interlock mode of a function in a raster order group, `None` to run unordered, or its
/// refusal. Called once the stage interface is planned (`fragment::finish`).
pub(super) fn mode(cx: &SelectCx<'_>) -> Result<Option<ExecMode>, Error> {
    if cx.stage != Stage::Fragment {
        // Metal orders raster order groups between fragments only.
        return Err(Unsupported::not_implemented(Feature::RasterOrderGroups).into());
    }
    let f = cx.profile.fields().fragment;
    let per_sample = sample_rate(&cx.families.stage_io.objects);
    if f.pixel_interlock {
        return Ok(Some(ExecMode::PixelInterlockOrdered));
    }
    if f.sample_interlock && per_sample {
        return Ok(Some(ExecMode::SampleInterlockOrdered));
    }
    // A group alone, on a device that orders attachments but has no interlock ([module docs](self));
    // an imageblock, the storage form and a stencil shadow cannot do without one.
    let fragment = &cx.families.fragment;
    let group_alone =
        fragment.imageblock.is_none() && fragment.fetch.is_empty() && fragment.stencil.is_none();
    if group_alone && f.rasterization_order_color_access {
        eprintln!(
            "metal2vulkan: the fragment function of a render pipeline state: its raster order \
             group on {} runs unordered: the device has no pixel interlock",
            group_memory(cx.entry)
        );
        return Ok(None);
    }
    let pixel = vec![Capability::Feature(VkFeature::FragmentShaderPixelInterlock)];
    let alternatives = if per_sample {
        vec![
            pixel,
            vec![Capability::Feature(
                VkFeature::FragmentShaderSampleInterlock,
            )],
        ]
    } else {
        vec![pixel]
    };
    Err(Unsupported::needs_any_of(Feature::RasterOrderGroups, alternatives).into())
}
