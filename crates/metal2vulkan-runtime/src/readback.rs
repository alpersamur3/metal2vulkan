//! The staging buffer `Texture::get_bytes` reads an image back through (`texture.rs`'s
//! `read_region`), kept by the device between calls instead of made and freed per call, and, on a
//! device with no host-cached coherent memory, placed in host-cached memory that is not coherent.
//!
//! - **Why.** reims shows the guest's screen by reading its frame target back once per frame
//!   (24 MB at 1179 x 2556 with 8-byte texels). A buffer per frame costs a `vkAllocateMemory`
//!   and a bind that pins thousands of pages on `mali_kbase`, and where the only host-visible
//!   coherent memory is uncached (ARM Mali, PowerVR) the host's copy out of it ran at 92 MB/s:
//!   250 ms a frame.
//! - **Memory.** [`MemoryClass::Readback`]: `Shared`'s type where the device has one the host
//!   caches and keeps coherent, as before; else a type the host caches without coherence, whose
//!   range this module invalidates ([`Memory::invalidate`]) after the GPU's copy completed and
//!   before the host reads, as Vulkan requires. Only this buffer takes such a type: images and
//!   every other buffer keep `Shared`'s rule (`memory.rs`, [`host_cached`](crate::memory)).
//! - **Which reads.** A read of an image that takes host copies (`texture/host_copy.rs`) goes
//!   through this buffer, not a copy on the host, when the image's memory type lacks
//!   `HOST_CACHED` and the device has a cached type for the buffer
//!   ([`read_through_buffer`](super::memory::read_through_buffer): Mali and PowerVR, whose
//!   host-transfer images stay in uncached memory). Why: the host reads uncached memory at tens of
//!   MB/s. The cost: that read is a waited submission on the device's first queue, as on a device
//!   without host image copy. Elsewhere the host copy stays (images in cached memory; devices with
//!   no cached type). Each distinct (path, memory type) says so once ([`host_copy_line`], and the
//!   buffer's own line).
//! - **Kept.** The device keeps one idle buffer of up to [`KEPT_MAX`] bytes. A read takes it out
//!   where it is large enough, and gives it back as it ends; a read that finds none (another
//!   thread has it) or one too small makes a buffer of its own. Of two buffers given back the
//!   larger is kept, so the buffer grows to the largest frame read and no more. A buffer whose
//!   submission failed is released, not kept: its work may still be pending. The device releases
//!   its idle buffer as it is destroyed.
//! - **Stale state.** None: every byte a read returns the GPU's copy wrote just before, and the
//!   buffer is host memory of this process outside the Metal address space, so a suspend and a
//!   resume of the guest's GPU state (a quick save and its load) leave nothing in it that matters.

use super::device::lock;
use super::error::{vulkan, Error, MemoryKind};
use super::exit_guard;
use super::memory::{allocate, readback_requirements, Memory, MemoryClass, Placement};
use super::mipmap_scratch::{Spare, Take};
use super::Device;
use crate::release;
use ash::vk;
use std::sync::Mutex;

/// The most bytes of an idle buffer the device keeps: a frame of a 4K display with 8-byte texels
/// is 66 MB, so a larger read (a guest's own `getBytes` of a big texture) is made, used and
/// released as before, and holds no memory afterwards.
pub(super) const KEPT_MAX: u64 = 96 << 20;

/// The id name of the informational message that says which memory type the read-back buffer
/// landed on ([`read_back_buffer_line`]).
pub(super) const READ_BACK_BUFFER: &str = "metal2vulkan-read-back-buffer";

/// The id name of the informational message that says a read of an image took a copy on the host
/// ([`host_copy_line`]).
pub(super) const READ_BACK_HOST_COPY: &str = "metal2vulkan-read-back-host-copy";

/// How a read of an image is answered, for [`announce`].
#[derive(Clone, Copy)]
pub(super) enum ReadPath {
    /// Through the kept buffer.
    Buffer = 0,
    /// A copy on the host out of the image's memory.
    HostCopy = 1,
}

/// The (path, memory type index, flag) triples announced in this process, packed as
/// `path << 33 | index << 1 | flag` (the flag: coherent for the buffer, cached for a host copy).
static ANNOUNCED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Whether a read by `path` on memory type `index` with `flag` is to be announced: the first time
/// the triple occurs in this process, which it then records.
pub(super) fn announce(path: ReadPath, index: u32, flag: bool) -> bool {
    let key = ((path as u64) << 33) | (u64::from(index) << 1) | u64::from(flag);
    let mut seen = ANNOUNCED.lock().unwrap_or_else(|e| e.into_inner());
    if seen.contains(&key) {
        return false;
    }
    seen.push(key);
    true
}

/// The names of the memory property `flags` set, in a fixed order, joined by " | ".
fn flag_words(flags: vk::MemoryPropertyFlags) -> String {
    const NAMES: [(vk::MemoryPropertyFlags, &str); 6] = [
        (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
        (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
        (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
        (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
        (
            vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            "LAZILY_ALLOCATED",
        ),
        (vk::MemoryPropertyFlags::PROTECTED, "PROTECTED"),
    ];
    let words: Vec<&str> = (NAMES.iter())
        .filter(|(flag, _)| flags.contains(*flag))
        .map(|(_, name)| *name)
        .collect();
    if words.is_empty() {
        "no known flag".to_owned()
    } else {
        words.join(" | ")
    }
}

/// The line that says a read of an image took a copy on the host out of memory type `index` of
/// property `flags`; `, uncached` is appended where the type lacks `HOST_CACHED`, so a slow host
/// copy that was not rerouted is visible. Prefix `m2v: read-back: host image copy`.
pub(super) fn host_copy_line(index: u32, flags: vk::MemoryPropertyFlags) -> String {
    let uncached = match flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
        true => "",
        false => ", uncached",
    };
    format!(
        "m2v: read-back: host image copy from memory type {index} ({}){uncached}",
        flag_words(flags)
    )
}

impl Device {
    /// Whether a read of an image placed in memory type `image_type` goes through the kept buffer
    /// ([`memory::read_through_buffer`](super::memory::read_through_buffer)).
    pub(super) fn reads_through_buffer(&self, image_type: u32) -> bool {
        lock(&self.shared().memory).reads_through_buffer(image_type)
    }

    /// Say once per distinct memory type that a read of an image took a host copy out of type
    /// `image_type`.
    pub(super) fn say_host_copy_read(&self, image_type: u32) {
        let flags = lock(&self.shared().memory)
            .type_flags(image_type)
            .unwrap_or_default();
        let cached = flags.contains(vk::MemoryPropertyFlags::HOST_CACHED);
        if announce(ReadPath::HostCopy, image_type, cached) {
            self.inform(READ_BACK_HOST_COPY, host_copy_line(image_type, flags));
        }
    }
}

/// The line that says where a read-back buffer of `len` bytes is: its memory type `index` with the
/// type's property `flags` in words, and whether the host's reads follow an invalidation after
/// each copy (`coherent` false) or not. Plain ASCII, one line, prefix `m2v: read-back buffer:`.
pub(super) fn read_back_buffer_line(
    len: u64,
    index: u32,
    flags: vk::MemoryPropertyFlags,
    coherent: bool,
) -> String {
    let words = flag_words(flags);
    let after = if coherent {
        "coherent"
    } else {
        "invalidated after each copy"
    };
    format!("m2v: read-back buffer: {len} bytes on memory type {index} ({words}), {after}")
}

/// A buffer and the memory it is bound to.
#[derive(Debug)]
pub(super) struct Block {
    raw: vk::Buffer,
    placement: Placement,
    /// The bytes the buffer holds.
    len: u64,
    /// The device's `nonCoherentAtomSize` where the memory is not coherent, so that the host's
    /// reads follow an invalidation; `None` where it is.
    invalidate: Option<u64>,
}

impl Block {
    /// A buffer of `len` bytes.
    fn make(device: &Device, len: u64) -> Result<Block, Error> {
        let shared = device.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(len)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { raw_device.create_buffer(&info, None) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let atom = lock(&shared.memory).atom();
        let requirements = readback_requirements(&requirements, atom);
        let placement = match allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Readback,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let free = |raw, placement: &Placement| {
            // SAFETY: nothing used the buffer.
            unsafe { release::object(raw_device, raw) };
            lock(&shared.memory).free(raw_device, placement);
        };
        if placement.mapped.is_none() {
            free(raw, &placement);
            return Err(Error::OutOfMemory(MemoryKind::Host));
        }
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        if let Err(error) =
            unsafe { raw_device.bind_buffer_memory(raw, placement.memory, placement.offset) }
        {
            free(raw, &placement);
            return Err(vulkan(error));
        }
        let (invalidate, flags) = {
            let memory = lock(&shared.memory);
            (
                memory.incoherent(&placement).then_some(atom),
                memory.type_flags(placement.type_index()),
            )
        };
        let flags = flags.unwrap_or_default();
        if announce(
            ReadPath::Buffer,
            placement.type_index(),
            invalidate.is_none(),
        ) {
            device.inform(
                READ_BACK_BUFFER,
                read_back_buffer_line(len, placement.type_index(), flags, invalidate.is_none()),
            );
        }
        Ok(Block {
            raw,
            placement,
            len,
            invalidate,
        })
    }

    /// Release the buffer and its memory.
    ///
    /// # Safety
    ///
    /// No command that uses the buffer is pending.
    unsafe fn release(self, device: &ash::Device, memory: &mut Memory) {
        // SAFETY: the caller's.
        unsafe { release::object(device, self.raw) };
        memory.free(device, &self.placement);
    }
}

/// The device's idle buffer ([`DeviceShared::readback`](super::device::DeviceShared)), with the
/// bytes it holds: the spare-block rule of `generate_mipmaps`' scratch (`mipmap_scratch.rs`).
pub(super) type Kept = Spare<Block>;

/// Whether the device keeps an idle buffer of `len` bytes ([`KEPT_MAX`]).
pub(super) fn keeps(len: u64) -> bool {
    len <= KEPT_MAX
}

impl Spare<Block> {
    /// Release the idle buffer as the device is destroyed.
    pub(super) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        if let Some(block) = self.drain() {
            // SAFETY: the device is idle, so no command uses the buffer.
            unsafe { block.release(device, memory) };
        }
    }
}

/// A buffer of `len` bytes for one read, given back to its device as it drops (the module docs).
pub(super) struct ReadStaging<'a> {
    device: &'a Device,
    block: Option<Block>,
    len: usize,
    /// The GPU's copy into it completed ([`ReadStaging::copied`]): only then may it be kept.
    done: bool,
}

impl<'a> ReadStaging<'a> {
    /// A buffer of `len` bytes: the device's idle one where it is large enough, else a new one.
    pub(super) fn new(device: &'a Device, len: u64) -> Result<ReadStaging<'a>, Error> {
        let size = usize::try_from(len).map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        let shared = device.shared();
        let taken = lock(&shared.readback).take(len);
        let block = match taken {
            Take::Reuse(block, _) => block,
            Take::Make(stale) => {
                if let Some(stale) = stale {
                    // SAFETY: an idle buffer's last read completed.
                    unsafe { stale.release(shared.vk(), &mut lock(&shared.memory)) };
                }
                Block::make(device, len)?
            }
        };
        Ok(ReadStaging {
            device,
            block: Some(block),
            len: size,
            done: false,
        })
    }

    /// The buffer.
    pub(super) fn raw(&self) -> vk::Buffer {
        self.block.as_ref().map_or(vk::Buffer::null(), |b| b.raw)
    }

    /// The submission that copied into the buffer completed: make its bytes visible to the host
    /// where the memory is not coherent, and let the buffer be kept.
    pub(super) fn copied(&mut self) -> Result<(), Error> {
        self.done = true;
        let Some(block) = &self.block else {
            return Ok(());
        };
        match block.invalidate {
            Some(atom) => Memory::invalidate(
                self.device.shared().vk(),
                &block.placement,
                self.len as u64,
                atom,
            ),
            None => Ok(()),
        }
    }

    /// The bytes, through the mapping, after [`copied`](Self::copied).
    pub(super) fn bytes(&mut self) -> &mut [u8] {
        let Some(mapped) = self.block.as_ref().and_then(|b| b.placement.mapped) else {
            unreachable!("checked when the buffer was made");
        };
        // SAFETY: the mapping covers the buffer's `len` bytes at least, which only this read uses
        // (an idle buffer is leased to one read at a time); the submission that wrote them
        // completed, and the range was invalidated where the memory is not coherent.
        unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), self.len) }
    }
}

impl Drop for ReadStaging<'_> {
    fn drop(&mut self) {
        let Some(block) = self.block.take() else {
            return;
        };
        // Once the process is exiting, the buffer is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = self.device.shared();
        let released = if self.done && keeps(block.len) {
            let len = block.len;
            lock(&shared.readback).give(block, len)
        } else {
            Some(block)
        };
        if let Some(block) = released {
            // SAFETY: every submission that used the buffer completed, or failed: a buffer whose
            // submission failed is released as the per-read buffer always was.
            unsafe { block.release(shared.vk(), &mut lock(&shared.memory)) };
        }
    }
}
