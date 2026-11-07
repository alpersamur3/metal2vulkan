//! Host memory gathered from a host region's pages (a runtime extension: Metal has none): a
//! no-copy buffer or heap whose bytes, at their pointer, are no mapping the device imports, but
//! pages scattered through a region the device can import in chunks (ASoftware's Android port:
//! the guest's RAM, QEMU's one shared mapping of its file, whose 16 KiB pages a guest object
//! takes in any order).
//!
//! - **Why.** A contiguous CPU view of scattered pages costs one kernel mapping per page on Linux,
//!   and a process holds at most `vm.max_map_count` (65,530 on Android, which no app may raise).
//!   The GPU needs no such view: a sparse `VkBuffer` gathers the pages through the GPU's page
//!   tables, one bind per piece, with no mapping of the CPU's (measured on an Adreno 830, Turnip
//!   26.2.6 over KGSL: 256 scattered 4 KiB pieces of one host import bound into a sparse buffer
//!   and read back by a GPU copy, none wrong).
//! - **The source** ([`GatherSource`]) answers, for the pages of an import's window, where they
//!   lie ([`GatherAnswer::Pieces`]): each piece a run of bytes of a region that stays mapped,
//!   writable and shared for the device's life. Or that the pages are mapped at the pointer now
//!   ([`GatherAnswer::Pointer`]), which the runtime then imports as [`HostMemory::new`]'s: the
//!   caller's fallback for pages no region holds. It is called with no lock of the runtime's held,
//!   once per import.
//! - **The chunks.** The runtime imports each region in chunks of [`CHUNK`] bytes from the
//!   region's first byte (`VK_EXT_external_memory_host`, the last chunk short at the region's end),
//!   each on its first use, and shares a live chunk among every gathered buffer over it
//!   ([`Chunks`]). A chunk goes once no buffer holds it, after a while of no use ([`Chunks`]'s
//!   lingering list), so that a page freed and taken again by the guest is not imported twice in a
//!   row. Drivers pin imported pages (KGSL does), so the chunks pin what the live buffers' pages
//!   lie in, rounded out to chunks.
//! - **The buffer** is a `SPARSE_BINDING` `VkBuffer` of the window's pages, made for host
//!   allocations, each piece bound at its offset from the window's first byte to its offset in its
//!   chunk by one `vkQueueBindSparse` on the device's first queue (`Device::bind_sparse_buffer`),
//!   as the dma-buf route binds its chunks (`dma_buf.rs`): the call does not wait, and every later
//!   submission waits for it. Its placement is the dma-buf route's (`Placement::dma_buf`), holding
//!   the chunks instead of imports of its own, so every rule of that route holds for it: a texture
//!   over the buffer is a shadow, as over any host pages; a texture placed in the heap is a shadow
//!   of the heap's bytes (no image is bound to the chunks), counted and announced.
//! - **The CPU's bytes** stay the caller's pointer: `contents` answers it, and the caller makes the
//!   bytes there show the same pages before the CPU reaches them (ASoftware's port maps them on the
//!   CPU's first touch). The CPU's writes through the pointer and the GPU's through the chunks
//!   meet in the same pages, host-coherent.
//! - **Refused** as the dma-buf route refuses (`Unsupported`, `HostMemory`): a device that binds no
//!   sparse buffer, or whose sparse buffer's alignment misses the pieces, a source that fails, a
//!   chunk the driver refuses. Pieces that do not cover the window exactly, or that run past their
//!   region, are misuse (`Invalid`, `Problem::DmaBuf`).

use super::buffer::usage;
use super::buffer_descriptor::HostBuffer;
use super::device::lock;
use super::device::functions::Functions;
use super::dma_buf::DmaBufPages;
use super::error::{invalid, vulkan, Error, Problem};
use super::host_memory::{based_size, import_refused, not_importable, ImportWindow, Importer};
use super::memory::{ImportFailure, MemoryClass, Placement};
use super::Device;
use crate::release;
use ash::vk;
use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// A runtime extension (Metal has none): where the pages of a no-copy import lie, for memory whose
/// pointer the device does not import ([`HostMemory::with_gather`](crate::HostMemory::with_gather),
/// the module docs). Called on the thread making the object, with no lock of the runtime's held,
/// once for each import of the memory (a buffer, a heap, or a rebind of either).
pub type GatherSource =
    Arc<dyn Fn(GatherRequest) -> std::io::Result<GatherAnswer> + Send + Sync + 'static>;

/// What a [`GatherSource`] is asked for: the `length` bytes of host pages from `start`, both
/// multiples of the host page, as pieces each a multiple of `granule` bytes long and starting at
/// a multiple of it in its region.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherRequest {
    /// The first byte of the first page.
    pub start: NonNull<u8>,
    /// The bytes of the pages.
    pub length: u64,
    /// The bytes every piece's length and its host address are a multiple of: the alignment of a
    /// sparse buffer of host allocations, rounded up to the host page.
    pub granule: u64,
}

// SAFETY: a request is an address and two lengths, plain data: nothing is reached through the
// address by having the request.
unsafe impl Send for GatherRequest {}
// SAFETY: as above.
unsafe impl Sync for GatherRequest {}

/// A [`GatherSource`]'s answer.
#[derive(Debug)]
pub enum GatherAnswer {
    /// The requested pages, in order: the first piece holds the bytes from `start`, each next one
    /// those from where the one before ends.
    Pieces(Vec<GatherPiece>),
    /// The pages are mapped at the request's pointer now, and stay so as `HostMemory::new`
    /// requires: the runtime imports the pointer.
    Pointer,
}

/// One piece of a [`GatherAnswer::Pieces`]: `length` bytes at host address `host`, inside the
/// region of `region_length` bytes from `region_start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GatherPiece {
    /// The bytes, a multiple of the granule.
    pub length: u64,
    /// The host address of the piece's first byte, a multiple of the granule from the region's
    /// first byte.
    pub host: u64,
    /// The region's first byte, a multiple of the host page.
    pub region_start: u64,
    /// The region's bytes, a multiple of the host page.
    pub region_length: u64,
}

/// The bytes the runtime imports a region in, from its first byte: 256 KiB, or
/// `METAL2VULKAN_GATHER_CHUNK_KIB` kibibytes (a power of two, 64 to 1048576), or
/// `METAL2VULKAN_GATHER_CHUNK_MIB` mebibytes (a power of two, 1 to 1024). Turnip binds a piece in
/// time that grows with the memory object's size (measured on the Adreno 830: 39-74 µs a piece in
/// a 256 MiB import, 495-598 µs in a 2 GiB one), so the chunks are small; each costs an import.
/// A driver that imports host memory by pinning its pages (KGSL) pins a chunk whole while any
/// gathered buffer holds it, so the chunk also bounds how much more than the buffers' own pages is
/// pinned where those lie scattered. Measured with iOS 26 on a Galaxy S25 Ultra (reims, the same
/// rounds of six apps from a cold boot per size): the chunks pinned on average 2,857 MiB for
/// 138 MiB of live gathered buffers at 16 MiB (where Android's lmkd killed the app), 1,641 for 146
/// at 2 MiB, 922 for 169 at 512 KiB and 703 for 180 at 256 KiB; an import took 4.6 ms at 16 MiB
/// and 0.15 ms at 256 KiB, a bind 57 and 29 µs, the imports and binds of a buffer 3.8 and 2.2 ms
/// in all.
pub const CHUNK: u64 = 256 << 10;

/// The bytes of chunks no buffer holds that the device keeps imported for a later one
/// ([`Chunks`]): 256 MiB, or `METAL2VULKAN_GATHER_LINGER_MIB` mebibytes.
const LINGER: u64 = 256 << 20;

/// The chunk size in use ([`CHUNK`]), read once.
fn chunk_size() -> u64 {
    static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        let read = |name: &str, most: u64, shift: u32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|n| (1..=most).contains(n) && n.is_power_of_two())
                .map(|n| n << shift)
        };
        read("METAL2VULKAN_GATHER_CHUNK_KIB", 1 << 20, 10)
            .filter(|bytes| *bytes >= 64 << 10)
            .or_else(|| read("METAL2VULKAN_GATHER_CHUNK_MIB", 1024, 20))
            .unwrap_or(CHUNK)
    })
}

/// The lingering bytes in use ([`LINGER`]), read once.
fn linger_bytes() -> u64 {
    static SIZE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SIZE.get_or_init(|| {
        std::env::var("METAL2VULKAN_GATHER_LINGER_MIB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(LINGER, |mib| mib << 20)
    })
}

/// The process's counts of the gather route, for the caller's own reports ([`gather_counts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct GatherCounts {
    /// Chunks imported now.
    pub chunks_live: u64,
    /// The bytes of the chunks imported now.
    pub chunk_bytes_live: u64,
    /// Chunks imported so far, ever.
    pub chunks_imported: u64,
    /// The time the chunks' imports took on the calling threads, in microseconds, ever.
    pub import_micros: u64,
    /// Gathered buffers made so far.
    pub buffers: u64,
    /// The binds of their pieces so far (after joining neighbours).
    pub binds: u64,
    /// Gathered buffers live now.
    pub buffers_live: u64,
    /// The bytes of the live gathered buffers' windows.
    pub buffer_bytes_live: u64,
    /// The time the sparse binds took on the calling threads, in microseconds, ever.
    pub bind_micros: u64,
    /// Imports whose source answered the pointer ([`GatherAnswer::Pointer`]).
    pub pointer_answers: u64,
    /// Textures placed in heaps over gathered pages that are shadows of the heap's bytes.
    pub heap_shadows: u64,
    /// Textures placed in heaps over gathered pages that are sparse linear images of the chunks.
    pub heap_sparse_images: u64,
}

static CHUNKS_LIVE: AtomicU64 = AtomicU64::new(0);
static CHUNK_BYTES_LIVE: AtomicU64 = AtomicU64::new(0);
static CHUNKS_IMPORTED: AtomicU64 = AtomicU64::new(0);
static IMPORT_MICROS: AtomicU64 = AtomicU64::new(0);
static BUFFERS: AtomicU64 = AtomicU64::new(0);
static BINDS: AtomicU64 = AtomicU64::new(0);
static BUFFERS_LIVE: AtomicU64 = AtomicU64::new(0);
static BUFFER_BYTES_LIVE: AtomicU64 = AtomicU64::new(0);
static BIND_MICROS: AtomicU64 = AtomicU64::new(0);
static POINTER_ANSWERS: AtomicU64 = AtomicU64::new(0);
static HEAP_SHADOWS: AtomicU64 = AtomicU64::new(0);

/// Counts a texture placed in a heap over gathered pages, a shadow (`texture/heap.rs`).
pub(super) fn heap_shadow_made() {
    HEAP_SHADOWS.fetch_add(1, Ordering::Relaxed);
}

/// The process's counts of the gather route so far ([`GatherCounts`]).
pub fn gather_counts() -> GatherCounts {
    let get = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
    GatherCounts {
        chunks_live: get(&CHUNKS_LIVE),
        chunk_bytes_live: get(&CHUNK_BYTES_LIVE),
        chunks_imported: get(&CHUNKS_IMPORTED),
        import_micros: get(&IMPORT_MICROS),
        buffers: get(&BUFFERS),
        binds: get(&BINDS),
        buffers_live: get(&BUFFERS_LIVE),
        buffer_bytes_live: get(&BUFFER_BYTES_LIVE),
        bind_micros: get(&BIND_MICROS),
        pointer_answers: get(&POINTER_ANSWERS),
        heap_shadows: get(&HEAP_SHADOWS),
        heap_sparse_images: get(&HEAP_SPARSE_IMAGES),
    }
}

/// One chunk of a region, imported (module docs): freed when its last holder goes, inside a
/// section of the exit guard; left to the process's end once it is exiting.
pub(super) struct HostChunk {
    /// The device's functions, which hold nothing of the device: the device's teardown drops the
    /// lingering chunks first (`Chunks::clear`), and every other holder holds the device.
    functions: Arc<Functions>,
    memory: vk::DeviceMemory,
    type_index: u32,
    first: u64,
    size: u64,
}

impl Drop for HostChunk {
    fn drop(&mut self) {
        // SAFETY: no buffer is bound to the chunk any more (every holder's buffer was destroyed
        // before it let go of it), and no command reads it.
        if unsafe { release::object(self.functions.get(), self.memory) } {
            crate::sparse_log::chunk(true, self.memory, self.first, self.size);
            CHUNKS_LIVE.fetch_sub(1, Ordering::Relaxed);
            CHUNK_BYTES_LIVE.fetch_sub(self.size, Ordering::Relaxed);
        }
    }
}

impl std::fmt::Debug for HostChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostChunk")
            .field("first", &format_args!("{:#x}", self.first))
            .field("size", &self.size)
            .field("type_index", &self.type_index)
            .finish_non_exhaustive()
    }
}

/// The device's live chunks, by their first byte and memory class, and those it keeps a while
/// after their last buffer went (module docs): a leaf lock, nothing called under it but the
/// chunks' drops of the lingering ones let go of, which take only the exit guard.
#[derive(Default)]
pub(super) struct Chunks {
    inner: Mutex<ChunkMap>,
}

#[derive(Default)]
struct ChunkMap {
    live: HashMap<(u64, u8), Weak<HostChunk>>,
    lingering: VecDeque<Arc<HostChunk>>,
    lingering_bytes: u64,
}

impl Chunks {
    /// The live chunk of `class` from `first`, where one is: taken out of the lingering list.
    fn get(&self, first: u64, class: MemoryClass) -> Option<Arc<HostChunk>> {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let chunk = map.live.get(&(first, class_key(class))).and_then(Weak::upgrade)?;
        if let Some(at) = map.lingering.iter().position(|c| Arc::ptr_eq(c, &chunk)) {
            if let Some(gone) = map.lingering.remove(at) {
                map.lingering_bytes -= gone.size;
            }
        }
        Some(chunk)
    }

    /// Note `chunk`, of `class`, live; the chunk already there for the same bytes, where another
    /// thread imported it meanwhile, wins, and is answered.
    fn put(&self, chunk: Arc<HostChunk>, class: MemoryClass) -> Arc<HostChunk> {
        let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let key = (chunk.first, class_key(class));
        if let Some(there) = map.live.get(&key).and_then(Weak::upgrade) {
            drop(map);
            return there;
        }
        map.live.insert(key, Arc::downgrade(&chunk));
        if map.live.len() > 4096 {
            map.live.retain(|_, weak| weak.strong_count() > 0);
        }
        chunk
    }

    /// Keep `chunks`, which a gathered buffer let go of, a while: those no other holds join the
    /// lingering list, and the oldest past [`linger_bytes`] go.
    fn linger(&self, chunks: Vec<Arc<HostChunk>>) {
        let mut gone = Vec::new();
        {
            let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let keep = linger_bytes();
            for chunk in chunks {
                // Held by another buffer (or the lingering list): nothing to keep.
                if Arc::strong_count(&chunk) > 1 || keep == 0 {
                    gone.push(chunk);
                    continue;
                }
                map.lingering_bytes += chunk.size;
                map.lingering.push_back(chunk);
            }
            while map.lingering_bytes > keep {
                let Some(oldest) = map.lingering.pop_front() else {
                    break;
                };
                map.lingering_bytes -= oldest.size;
                gone.push(oldest);
            }
        }
        // Freed with no lock held.
        drop(gone);
    }

    /// Let go of every lingering chunk: the device's teardown, before it destroys the device.
    pub(super) fn clear(&self) {
        let lingering = {
            let mut map = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            map.lingering_bytes = 0;
            std::mem::take(&mut map.lingering)
        };
        drop(lingering);
    }
}

/// A memory class as the chunk map's key.
fn class_key(class: MemoryClass) -> u8 {
    match class {
        MemoryClass::Shared => 0,
        MemoryClass::Managed => 1,
        _ => 2,
    }
}

/// What a gathered buffer's placement holds (`DmaBufPages`): its chunks, let go of into the
/// lingering list as the placement goes.
pub(super) struct GatheredChunks {
    pub(super) chunks: Vec<Arc<HostChunk>>,
    /// The importer's chunks, where the held ones linger.
    pub(super) cache: Arc<Chunks>,
    /// The window's bytes, uncounted from the live buffers as it goes.
    pub(super) bytes: u64,
    /// The buffer's binds, in order: where each stretch of the window lies in the chunks, which a
    /// texture placed in a heap over the pages binds its image to ([`GatheredChunks::image_binds`]).
    pub(super) binds: Vec<(u64, u64, vk::DeviceMemory, u64)>,
}

impl GatheredChunks {
    /// The opaque binds of an image of `size` bytes at `at` of the window, to the chunks its
    /// bytes lie in: each stretch at its offset in the image, every offset a multiple of
    /// `granule` (the image's alignment) and every size too, but the last, which ends at the
    /// image's end. `None` where a stretch misses the granule or the window does not hold them.
    pub(super) fn image_binds(
        &self,
        at: u64,
        size: u64,
        granule: u64,
    ) -> Option<Vec<vk::SparseMemoryBind>> {
        let granule = granule.max(1);
        let end = at.checked_add(size)?;
        let mut binds = Vec::new();
        let mut covered = at;
        for &(resource, length, memory, offset) in &self.binds {
            let (from, to) = (resource.max(at), (resource + length).min(end));
            if from >= to {
                continue;
            }
            if from != covered {
                return None;
            }
            let into = from - at;
            let memory_offset = offset + (from - resource);
            let bytes = to - from;
            let whole = bytes.is_multiple_of(granule) || to == end;
            if !into.is_multiple_of(granule) || !memory_offset.is_multiple_of(granule) || !whole {
                return None;
            }
            binds.push(
                vk::SparseMemoryBind::default()
                    .resource_offset(into)
                    .size(bytes)
                    .memory(memory)
                    .memory_offset(memory_offset),
            );
            covered = to;
        }
        (covered == end).then_some(binds)
    }
}

/// Waits until the device's internal timeline `timeline` reached `value`: a sparse binding the
/// runtime made (`Device::bind_sparse_buffer`, `Device::bind_sparse_image`), before what it
/// bound is destroyed. A lost device ends the wait.
pub(super) fn settle_binding(device: &Device, (timeline, value): (vk::Semaphore, u64)) {
    let raw_device = device.shared().vk();
    let semaphores = [timeline];
    let values = [value];
    let info = vk::SemaphoreWaitInfo::default()
        .semaphores(&semaphores)
        .values(&values);
    loop {
        // SAFETY: the timeline is the device's, live while the device is.
        let waited = unsafe { raw_device.wait_semaphores(&info, 100_000_000) };
        crate::exit_guard::called(crate::exit_guard::Call::WaitSemaphores);
        if waited != Err(vk::Result::TIMEOUT) {
            return;
        }
    }
}

/// Textures placed in heaps over gathered pages that are sparse linear images of the chunks.
static HEAP_SPARSE_IMAGES: AtomicU64 = AtomicU64::new(0);

/// Counts a texture placed in a heap over gathered pages as a sparse linear image
/// (`texture/heap.rs`).
pub(super) fn heap_sparse_image_made() {
    HEAP_SPARSE_IMAGES.fetch_add(1, Ordering::Relaxed);
}

impl Drop for GatheredChunks {
    fn drop(&mut self) {
        BUFFERS_LIVE.fetch_sub(1, Ordering::Relaxed);
        BUFFER_BYTES_LIVE.fetch_sub(self.bytes, Ordering::Relaxed);
        let chunks = std::mem::take(&mut self.chunks);
        self.cache.linger(chunks);
    }
}

/// Whether `pieces` answer `request` ([`GatherSource`]'s contract): each a positive multiple of
/// the granule, at a multiple of it from its region's first byte, inside its region; together the
/// request's length.
fn pieces_cover(pieces: &[GatherPiece], request: &GatherRequest, page: u64) -> bool {
    let granule = request.granule.max(1);
    let page = page.max(1);
    let mut total = 0u64;
    for piece in pieces {
        let Some(end) = piece.host.checked_add(piece.length) else {
            return false;
        };
        let Some(region_end) = piece.region_start.checked_add(piece.region_length) else {
            return false;
        };
        let inside = piece.host >= piece.region_start && end <= region_end;
        let aligned = piece.length != 0
            && piece.length.is_multiple_of(granule)
            && piece.host.wrapping_sub(piece.region_start).is_multiple_of(granule)
            && piece.region_start.is_multiple_of(page)
            && piece.region_length.is_multiple_of(page);
        if !inside || !aligned {
            return false;
        }
        total = match total.checked_add(piece.length) {
            Some(total) => total,
            None => return false,
        };
    }
    !pieces.is_empty() && total == request.length
}

/// One bind of a gathered buffer ([`binds_of`]): `size` bytes at `resource` in the buffer, `offset`
/// bytes into the chunk from `first`, of `chunk` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PieceBind {
    resource: u64,
    size: u64,
    first: u64,
    offset: u64,
    chunk: u64,
}

/// The binds of a gathered buffer over `pieces`, chunks of `chunk` bytes from each region's first
/// byte (the last one short at the region's end): each piece cut at its chunks' boundaries,
/// neighbours in the same chunk that follow on there and in the buffer joined.
fn binds_of(pieces: &[GatherPiece], chunk: u64) -> Vec<PieceBind> {
    let mut binds: Vec<PieceBind> = Vec::new();
    let mut resource = 0u64;
    for piece in pieces {
        let region_end = piece.region_start + piece.region_length;
        let mut host = piece.host;
        let end = piece.host + piece.length;
        while host < end {
            let first = piece.region_start + (host - piece.region_start) / chunk * chunk;
            let chunk_end = (first + chunk).min(region_end);
            let now = end.min(chunk_end) - host;
            let offset = host - first;
            match binds.last_mut() {
                Some(last)
                    if last.first == first
                        && last.offset + last.size == offset
                        && last.resource + last.size == resource =>
                {
                    last.size += now;
                }
                _ => binds.push(PieceBind {
                    resource,
                    size: now,
                    first,
                    offset,
                    chunk: chunk_end - first,
                }),
            }
            resource += now;
            host += now;
        }
    }
    binds
}

impl Device {
    /// The `VkBuffer` of a no-copy buffer of `length` bytes at `pointer`, over `window`'s pages
    /// gathered from the regions `source` names (module docs), as memory of `class`: the buffer
    /// and its placement, which holds the chunks. Where the source answers the pointer, the
    /// pointer's import instead, as [`HostMemory::new`]'s ([`Device::import_by_pointer`] refused
    /// is refused here). Refused as the module says.
    ///
    /// # Safety
    ///
    /// The window's pages lie where the source answers for as long as the placement lives, in
    /// regions that stay mapped and writable for the device's life (`HostMemory::with_gather`);
    /// where it answers the pointer, they are mapped there as `HostMemory::new` requires.
    pub(super) unsafe fn import_gathered(
        &self,
        call: &'static str,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        window: &ImportWindow,
        source: GatherSource,
        class: MemoryClass,
    ) -> Result<(HostBuffer, Placement), Error> {
        let shared = self.shared();
        // The window starts `window.offset` bytes before the pointer, in the pages it touches.
        let start = NonNull::new(pointer.as_ptr().wrapping_sub(window.offset as usize))
            .ok_or_else(not_importable)?;
        let Some(granule) = self.gather_granule(importer) else {
            // No sparse buffer of host allocations on this device: the pages at the pointer.
            return self.gathered_by_pointer(importer, pointer, length, window, &source, class);
        };
        let request = GatherRequest {
            start,
            length: window.size,
            granule,
        };
        let pieces = match source(request).map_err(|_| not_importable())? {
            GatherAnswer::Pieces(pieces) => pieces,
            GatherAnswer::Pointer => {
                POINTER_ANSWERS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: the source answered the pages mapped at the pointer.
                return unsafe { self.import_pointer(importer, pointer, length, window, class) };
            }
        };
        if !pieces_cover(&pieces, &request, importer.page) {
            return Err(invalid(call, Problem::DmaBuf));
        }
        // A range the binding layer pads to 16 bytes from the bytes stays inside the buffer.
        if based_size(window.offset, length).is_none_or(|padded| padded > window.size) {
            return self.gathered_by_pointer(importer, pointer, length, window, &source, class);
        }
        let raw_device = shared.vk();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .flags(vk::BufferCreateFlags::SPARSE_BINDING)
                .size(window.size)
                .usage(usage(shared.device_address))
                .push_next(&mut external),
        );
        // SAFETY: the device is live, binds sparse memory (`gather_granule`), and `info` and its
        // chain are valid.
        let raw = unsafe { raw_device.create_buffer(&info, None) }.map_err(vulkan)?;
        shared.counts.buffer_made(info.usage);
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let made = (|| {
            if !granule.is_multiple_of(requirements.alignment.max(1)) {
                return Err(not_importable());
            }
            let chunk = chunk_size();
            let binds = binds_of(&pieces, chunk);
            // The chunks, each once, in the order the binds first name them.
            let mut held: Vec<Arc<HostChunk>> = Vec::new();
            let mut index: HashMap<u64, usize> = HashMap::new();
            let mut sparse: Vec<vk::SparseMemoryBind> = Vec::with_capacity(binds.len());
            for bind in &binds {
                let at = match index.get(&bind.first) {
                    Some(&at) => at,
                    None => {
                        let chunk =
                            self.host_chunk(importer, bind.first, bind.chunk, class, &requirements)?;
                        if requirements.memory_type_bits & (1 << chunk.type_index) == 0 {
                            return Err(not_importable());
                        }
                        held.push(chunk);
                        index.insert(bind.first, held.len() - 1);
                        held.len() - 1
                    }
                };
                sparse.push(
                    vk::SparseMemoryBind::default()
                        .resource_offset(bind.resource)
                        .size(bind.size)
                        .memory(held[at].memory)
                        .memory_offset(bind.offset),
                );
            }
            let type_index = held.first().map(|c| c.type_index).ok_or_else(not_importable)?;
            let began = std::time::Instant::now();
            let bound = self.bind_sparse_buffer(raw, &sparse)?;
            let micros = began.elapsed().as_micros() as u64;
            BIND_MICROS.fetch_add(micros, Ordering::Relaxed);
            BINDS.fetch_add(sparse.len() as u64, Ordering::Relaxed);
            BUFFERS.fetch_add(1, Ordering::Relaxed);
            BUFFERS_LIVE.fetch_add(1, Ordering::Relaxed);
            BUFFER_BYTES_LIVE.fetch_add(window.size, Ordering::Relaxed);
            if let Some(log) = crate::memory_log::MemoryLog::new(shared.instance.sink()) {
                log.line(format!(
                    "metal2vulkan: memory: gathered host memory 0x{:x}, {} bytes, from {} \
                     pieces in {} chunks, {} binds in {micros} us",
                    start.as_ptr() as u64,
                    window.size,
                    pieces.len(),
                    held.len(),
                    sparse.len(),
                ));
            }
            let chunks = GatheredChunks {
                binds: (sparse.iter())
                    .map(|b| (b.resource_offset, b.size, b.memory, b.memory_offset))
                    .collect(),
                chunks: held,
                cache: Arc::clone(&importer.chunks),
                bytes: window.size,
            };
            let pages = DmaBufPages::gathered(self, start, window.size, chunks, bound);
            let placement = Placement::dma_buf(
                Arc::new(pages),
                vk::DeviceMemory::null(),
                type_index,
                window.size,
                start,
            );
            Ok(placement)
        })();
        match made {
            Ok(placement) => {
                let buffer = HostBuffer {
                    raw,
                    base: window.offset,
                    size: window.size,
                    requirements,
                };
                Ok((buffer, placement))
            }
            Err(error) => {
                // SAFETY: no command uses the buffer; a failed binding left nothing pending.
                unsafe { release::object(raw_device, raw) };
                if matches!(error, Error::DeviceLost) {
                    return Err(error);
                }
                // A device that refused the sparse buffer or a chunk: the pages at the pointer,
                // which the source maps there for good.
                if let Some(log) = crate::memory_log::MemoryLog::new(shared.instance.sink()) {
                    log.line(format!(
                        "metal2vulkan: memory: no gather of host memory 0x{:x}, {} bytes                          ({error:?}): imported at its pointer",
                        start.as_ptr() as u64,
                        window.size
                    ));
                }
                self.gathered_by_pointer(importer, pointer, length, window, &source, class)
            }
        }
    }

    /// The pointer's import of the window, for a source on a device that gathers no pages: the
    /// source is asked at the granule of the window's own length, which answers the pointer once
    /// the pages are mapped there. Refused where it answers pieces.
    fn gathered_by_pointer(
        &self,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        window: &ImportWindow,
        source: &GatherSource,
        class: MemoryClass,
    ) -> Result<(HostBuffer, Placement), Error> {
        let start = NonNull::new(pointer.as_ptr().wrapping_sub(window.offset as usize))
            .ok_or_else(not_importable)?;
        let request = GatherRequest {
            start,
            length: window.size,
            granule: window.size,
        };
        match source(request).map_err(|_| not_importable())? {
            GatherAnswer::Pointer => {
                POINTER_ANSWERS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: the source answered the pages mapped at the pointer.
                unsafe { self.import_pointer(importer, pointer, length, window, class) }
            }
            GatherAnswer::Pieces(_) => Err(not_importable()),
        }
    }

    /// The chunk of `size` bytes from `first`, of `class`: the live one, or a new import of it
    /// for a sparse buffer of `requirements`.
    fn host_chunk(
        &self,
        importer: &Importer,
        first: u64,
        size: u64,
        class: MemoryClass,
        requirements: &vk::MemoryRequirements,
    ) -> Result<Arc<HostChunk>, Error> {
        if let Some(chunk) = importer.chunks.get(first, class) {
            return Ok(chunk);
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        let began = std::time::Instant::now();
        let pointer = NonNull::new(first as *mut u8).ok_or_else(not_importable)?;
        // SAFETY: the region's pages are host memory the source vouches for, at the import
        // alignment (a chunk starts at a multiple of the host page from the region's first byte).
        let types = unsafe { importer.memory_types(raw_device, pointer) }.map_err(import_refused)?;
        let allowed = types & requirements.memory_type_bits;
        // SAFETY: as above: mapped and writable for the device's life.
        let (memory, type_index) = unsafe {
            lock(&shared.memory).import_host_chunk(raw_device, pointer, size, (allowed, class))
        }
        .map_err(|failure| match failure {
            ImportFailure::NoMemoryType => not_importable(),
            ImportFailure::Vulkan(error) => import_refused(error),
        })?;
        crate::sparse_log::chunk(false, memory, first, size);
        IMPORT_MICROS.fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
        CHUNKS_LIVE.fetch_add(1, Ordering::Relaxed);
        CHUNK_BYTES_LIVE.fetch_add(size, Ordering::Relaxed);
        CHUNKS_IMPORTED.fetch_add(1, Ordering::Relaxed);
        let chunk = Arc::new(HostChunk {
            functions: shared.functions(),
            memory,
            type_index,
            first,
            size,
        });
        Ok(importer.chunks.put(chunk, class))
    }

    /// The granule a gather source's pieces are cut at, where the device makes a sparse buffer of
    /// host allocations: the alignment of one, rounded up to the host page. `None` where the
    /// device binds no sparse memory. Asked of the device once ([`Importer::gather_granule`]).
    fn gather_granule(&self, importer: &Importer) -> Option<u64> {
        *importer.gather_granule.get_or_init(|| {
            let shared = self.shared();
            if !shared.sparse.binds() {
                return None;
            }
            let raw_device = shared.vk();
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let info = shared.sharing().buffer(
                vk::BufferCreateInfo::default()
                    .flags(vk::BufferCreateFlags::SPARSE_BINDING)
                    .size(importer.page)
                    .usage(usage(shared.device_address))
                    .push_next(&mut external),
            );
            // SAFETY: the device is live, binds sparse memory, and `info` and its chain are valid.
            let raw = unsafe { raw_device.create_buffer(&info, None) }.ok()?;
            shared.counts.buffer_made(info.usage);
            // SAFETY: `raw` is a live buffer of this device.
            let alignment = unsafe { raw_device.get_buffer_memory_requirements(raw) }.alignment;
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
            alignment
                .max(1)
                .checked_next_multiple_of(importer.page)
                .filter(|granule| granule.is_multiple_of(alignment.max(1)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(host: u64, length: u64) -> GatherPiece {
        GatherPiece {
            length,
            host,
            region_start: 0x1000_0000,
            region_length: 0x1000_0000,
        }
    }

    /// Pieces are cut at the chunks' boundaries, and neighbours in a chunk that follow on in it
    /// and in the buffer are one bind
    #[test]
    fn binds_cut_at_chunks_and_join_neighbours() {
        let chunk = 0x10_0000;
        let base = 0x1000_0000;
        let pieces = [
            piece(base + 0x4000, 0x4000),
            piece(base + 0x8000, 0x4000),
            piece(base + 0x30_0000, 0x4000),
            piece(base + chunk - 0x4000, 0x8000),
        ];
        let binds = binds_of(&pieces, chunk);
        let at = |resource, size, first, offset| PieceBind {
            resource,
            size,
            first,
            offset,
            chunk,
        };
        assert_eq!(
            binds,
            vec![
                at(0, 0x8000, base, 0x4000),
                at(0x8000, 0x4000, base + 0x30_0000, 0),
                at(0xc000, 0x4000, base, chunk - 0x4000),
                at(0x1_0000, 0x4000, base + chunk, 0),
            ]
        );
        // A region's last chunk is short at its end.
        let short = GatherPiece {
            length: 0x4000,
            host: base + 0x1000_0000 - 0x4000,
            region_start: base,
            region_length: 0x1000_0000,
        };
        let odd = binds_of(&[short], 0x300_0000);
        assert_eq!(odd[0].first, base + 0xf00_0000);
        assert_eq!(odd[0].chunk, 0x100_0000);
    }

    /// Pieces cover a request exactly, inside their regions, at the granule
    #[test]
    fn pieces_cover_the_request() {
        let start = NonNull::<u8>::dangling();
        let request = GatherRequest {
            start,
            length: 0x8000,
            granule: 0x1000,
        };
        let base = 0x1000_0000;
        assert!(pieces_cover(
            &[piece(base, 0x4000), piece(base + 0x10000, 0x4000)],
            &request,
            0x1000
        ));
        assert!(!pieces_cover(&[piece(base, 0x4000)], &request, 0x1000));
        assert!(!pieces_cover(
            &[piece(base + 0x800, 0x4000), piece(base, 0x4000)],
            &request,
            0x1000
        ));
        let outside = GatherPiece {
            region_length: 0x4000,
            ..piece(base + 0x2000, 0x8000)
        };
        assert!(!pieces_cover(&[outside], &request, 0x1000));
        assert!(!pieces_cover(&[], &request, 0x1000));
    }
}
