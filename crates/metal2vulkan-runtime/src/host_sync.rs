//! Cache maintenance for host imports that lie in memory the GPU is not coherent with: a no-copy
//! buffer's pages imported into a memory type that is `HOST_VISIBLE` without `HOST_COHERENT`
//! (`memory.rs`'s `import_candidates`, second tier).
//!
//! Imported host memory is the caller's cached CPU memory, so a driver whose GPU does not snoop
//! the CPU's caches offers it only as a type without `HOST_COHERENT` (a Mali-G1-Ultra lists a
//! coherent type, which host pointers cannot be imported into, and a cached one that is not
//! coherent, which they can). A third party measured the consequence on that phone: a compute
//! dispatch's writes into a 4 MiB import read back as stale zeros for 40 to 52 percent of the
//! words, none once the range was invalidated before the host read it, and `vkMapMemory` of the
//! imported memory answered the imported pointer. Vulkan's contract for such memory is
//! `vkFlushMappedMemoryRanges` after the host's writes and before the device reads them, and
//! `vkInvalidateMappedMemoryRanges` after the device's writes and before the host reads them,
//! over ranges that are multiples of `nonCoherentAtomSize`, on memory that is host mapped
//! (`VUID-VkMappedMemoryRange-memory-00684`).
//!
//! - **Mapped once.** An import made in such a type is mapped whole right after it is made
//!   (`Memory::import_host`) and stays mapped until it is freed. A map that fails or answers any
//!   pointer but the imported one leaves the import usable and unsynced (`ImportSync::Unsynced`,
//!   a `noncoherent-map` note): the engine that booted a phone before this ran such imports with
//!   no maintenance at all.
//! - **The unit is the held set.** The runtime already knows, for each command buffer, the
//!   buffers over host pages and the heaps over host memory its commands used (`rebind::Taken`,
//!   what the rebind rule holds), at no cost while the device has none. For each such import
//!   in non-coherent memory, a [`Span`] is planned: the `VkBuffer`'s whole reach, its bytes and the
//!   pages after them it can address (`Buffer::copy_reach`), rounded out to atoms ([`mapped_range`]).
//! - **Flushed at the first submission.** The guest's CPU writes any byte of a shared buffer at any
//!   time and the GPU reads it later, so each span is flushed once, at its command buffer's first
//!   submission, on the thread that submits, before any of its batches goes to the `VkQueue`
//!   (`command_buffer/first_submission.rs`), and a segment a host step records later is
//!   flushed as it is placed (`State::failing_at`).
//! - **Invalidated at the completion.** The GPU writes buffers and the guest's CPU reads them once
//!   the command buffer completed, so each span the GPU may have written is invalidated when the
//!   command buffer's last batch finished, before its events are reached and before any
//!   completed handler or host read (`command_buffer/completion.rs`).
//!   `vkInvalidateMappedMemoryRanges` is `DC IVAC` over the range on Android and Linux drivers,
//!   which DISCARDS dirty cache lines, and the guest's CPU may be writing other bytes of the same
//!   shared buffer then. So the span is flushed first, immediately before, so that only clean
//!   lines are discarded; **a CPU write that lands between that flush and the invalidate is
//!   still lost**, and no call closes the window (Vulkan has no clean-and-invalidate). And only
//!   spans the GPU may have written are invalidated: a buffer every use of which was a read
//!   (`Taken::wrote`: no use with a write access, none of unknown access) is only flushed.
//! - **Mode.** `METAL2VULKAN_HOST_SYNC`, read once per device: `full` (the default: flush and
//!   invalidate), `flush` (flush only) or `off` (no maintenance at all, the kill switch).
//! - **Cost.** A device with no import in a non-coherent type pays one relaxed load of
//!   [`HostSync::active`] where an encoder touches a resource, where a command buffer is first
//!   submitted and where one finishes; nothing else (`WorkCounts::host_flushes` and kin stay 0).
//!
//! **Not covered**, so that nobody assumes it is: pointers the GPU reads out of a buffer
//! (indirect access reaches any live buffer through the resident set, which no use names);
//! work the device submits on its own (`Device::submit_internal`) that reads an import; a host step
//! that reads GPU-written memory between two batches of a command buffer (invalidated once, at the
//! last batch's end); a CPU read keyed on an event a batch other than the last signals; the
//! CPU's writes between a command buffer's first submission and its later batches; an import of
//! dma-bufs or of a gather's chunks (`gather.rs`: unchanged, coherent memory only).

use super::buffer::HostPages;
use super::device::DeviceShared;
use super::memory::mapped_range;
use super::Device;
use ash::vk;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// What `METAL2VULKAN_HOST_SYNC` asks of the maintenance of non-coherent imports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Flush at a command buffer's first submission; flush, then invalidate, at its completion
    /// (the default).
    Full,
    /// Flush only.
    Flush,
    /// No maintenance at all.
    Off,
}

impl Mode {
    /// The mode the variable's `value` asks for, `None` where it is unset: `full` (also the
    /// default, for no value, an empty one and anything not understood: the safe choice),
    /// `flush`, or `off` (`0`, `no`, `none`, `false` too), in any case.
    pub(crate) fn parse(value: Option<&str>) -> Mode {
        let Some(value) = value else {
            return Mode::Full;
        };
        match value.trim().to_ascii_lowercase().as_str() {
            "flush" => Mode::Flush,
            "off" | "0" | "no" | "none" | "false" => Mode::Off,
            _ => Mode::Full,
        }
    }

    /// The mode the process's environment asks for.
    pub(crate) fn from_env() -> Mode {
        Mode::parse(std::env::var("METAL2VULKAN_HOST_SYNC").ok().as_deref())
    }

    /// Whether the host's writes are flushed.
    pub(crate) fn flushes(self) -> bool {
        self != Mode::Off
    }

    /// Whether the GPU's writes are invalidated.
    pub(crate) fn invalidates(self) -> bool {
        self == Mode::Full
    }

    /// Its name, as the variable spells it.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Full => "full",
            Mode::Flush => "flush",
            Mode::Off => "off",
        }
    }
}

/// How the host's view of an import's memory is kept in step with the GPU's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ImportSync {
    /// The memory is coherent (or no import's): nothing to do.
    #[default]
    Coherent,
    /// The memory is not coherent and is host mapped: its ranges are flushed and invalidated.
    Maintained,
    /// The memory is not coherent and could not be mapped as the imported pointer: used as the
    /// field engine used such memory, with no maintenance.
    Unsynced,
}

/// A device's cache maintenance state (`DeviceShared::host_sync`).
pub(crate) struct HostSync {
    mode: Mode,
    /// The device's `nonCoherentAtomSize`.
    atom: u64,
    /// An import that is maintained was made in non-coherent memory: set once, never cleared, as
    /// the read that tells a device with none (the module's cost).
    active: AtomicBool,
    /// The one line to stderr was printed.
    announced: AtomicBool,
}

impl HostSync {
    /// The state of a device with a `nonCoherentAtomSize` of `atom`, in the mode the environment
    /// asks for (a test's forced mode first).
    pub(crate) fn new(atom: u64) -> HostSync {
        #[cfg(test)]
        let mode = forced::mode().unwrap_or_else(Mode::from_env);
        #[cfg(not(test))]
        let mode = Mode::from_env();
        HostSync::with_mode(mode, atom)
    }

    /// The state of a device in `mode`.
    pub(crate) fn with_mode(mode: Mode, atom: u64) -> HostSync {
        HostSync {
            mode,
            atom: atom.max(1),
            active: AtomicBool::new(false),
            announced: AtomicBool::new(false),
        }
    }

    /// Whether the device has an import to maintain: one relaxed load.
    pub(crate) fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// The mode.
    pub(crate) fn mode(&self) -> Mode {
        self.mode
    }

    /// The device's `nonCoherentAtomSize`.
    pub(crate) fn atom(&self) -> u64 {
        self.atom
    }

    /// An import was made in memory type `type_index`, of property flags `flags`, whose
    /// maintenance is `sync`: maintained ones switch the device's maintenance on (unless the mode
    /// is `off`), and the first of any tells the process once, on stderr.
    pub(crate) fn imported(
        &self,
        sync: ImportSync,
        type_index: u32,
        flags: vk::MemoryPropertyFlags,
    ) {
        if sync == ImportSync::Coherent {
            return;
        }
        if sync == ImportSync::Maintained && self.mode != Mode::Off {
            self.active.store(true, Ordering::Relaxed);
        }
        if self.announced.swap(true, Ordering::Relaxed) {
            return;
        }
        let kept = match sync {
            ImportSync::Unsynced => "; the first of them could not be mapped and is used unsynced",
            _ => "",
        };
        eprintln!(
            "metal2vulkan: host imports use memory type {type_index} (flags 0x{:x}): not coherent, \
             ranges are flushed at a command buffer's submission and invalidated at its completion \
             (mode {}, atom {}){kept}",
            flags.as_raw(),
            self.mode.name(),
            self.atom
        );
    }
}

/// A range of an import's memory to maintain: whole atoms of `memory`, from `offset`, `size` bytes
/// (clipped to the memory's size, where it is not a multiple of the atom).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl Span {
    /// The span of the `len` bytes at `start` of `memory`, which is `memory_size` bytes long, on a
    /// device whose atom is `atom` ([`mapped_range`]); `None` where it is empty.
    pub(crate) fn of(
        memory: vk::DeviceMemory,
        (start, len): (u64, u64),
        (atom, memory_size): (u64, u64),
    ) -> Option<Span> {
        let (offset, size) = mapped_range(start, len, atom, memory_size);
        (size > 0).then_some(Span {
            memory,
            offset,
            size,
        })
    }

    fn range(&self) -> vk::MappedMemoryRange<'static> {
        vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(self.offset)
            .size(self.size)
    }
}

/// A command buffer's planned maintenance (`State::host_sync`): the buffers over host pages its
/// commands took (`Submitting::submit` notes each) and, once resolved, the spans of those whose
/// import is not coherent.
#[derive(Default)]
pub(crate) struct SyncPlan {
    /// Pages noted and not yet resolved to spans, each with whether the GPU may have written them.
    pending: Vec<(Arc<HostPages>, bool)>,
    spans: Vec<Planned>,
}

struct Planned {
    span: Span,
    written: bool,
    flushed: bool,
}

impl SyncPlan {
    /// Note `pages`, taken by a command: written by the GPU where `written`. Resolved, under no
    /// lock of the device's buffers over host pages, by [`flush`](Self::flush).
    pub(crate) fn note(&mut self, pages: &Arc<HostPages>, written: bool) {
        self.pending.push((Arc::clone(pages), written));
    }

    /// The pages noted so far turned into spans: those of imports that are maintained, once each
    /// (imports of one buffer's bytes made by several buffers share their span), written where any
    /// use of them may have written.
    fn resolve(&mut self, atom: u64) {
        for (pages, written) in std::mem::take(&mut self.pending) {
            let Some(span) = pages.import().sync_span(atom) else {
                continue;
            };
            match self.spans.iter_mut().find(|planned| planned.span == span) {
                Some(planned) => planned.written |= written,
                None => self.spans.push(Planned {
                    span,
                    written,
                    flushed: false,
                }),
            }
        }
    }

    /// Flush the spans planned and not yet flushed, in the device's mode: the host's writes made
    /// visible to the GPU, before the batch that uses them is submitted. Counted.
    pub(crate) fn flush(&mut self, device: &Device) {
        let shared = device.shared();
        if !shared.host_sync.mode().flushes() || device.is_lost() {
            self.pending.clear();
            return;
        }
        self.resolve(shared.host_sync.atom());
        let ranges: Vec<Span> = (self.spans.iter_mut())
            .filter(|planned| !planned.flushed)
            .map(|planned| {
                planned.flushed = true;
                planned.span
            })
            .collect();
        flush_spans(shared, &ranges);
    }

    /// The command buffer's last batch finished: invalidate the spans the GPU may have written,
    /// each flushed first (the module's window), before the host reads any of it. The spans are
    /// let go, so that a second call does nothing. Counted.
    pub(crate) fn complete(&mut self, device: &Device) {
        let shared = device.shared();
        let written: Vec<Span> = (self.spans.drain(..))
            .filter(|planned| planned.written)
            .map(|planned| planned.span)
            .collect();
        self.pending.clear();
        if !shared.host_sync.mode().invalidates() || written.is_empty() || device.is_lost() {
            return;
        }
        // The memory is held by the command buffer's retained pages until its handlers ran,
        // and a process that is exiting leaves it to its end.
        let Some(_busy) = crate::exit_guard::enter() else {
            return;
        };
        flush_spans(shared, &written);
        let ranges: Vec<vk::MappedMemoryRange<'static>> = written.iter().map(Span::range).collect();
        // SAFETY: every range lies in memory of this device that is imported, mapped whole
        // (`Memory::import_host`) and held by the command buffer's retained pages; the ranges are
        // whole atoms or end at their memory's end (`mapped_range`).
        let result = unsafe { shared.vk().invalidate_mapped_memory_ranges(&ranges) };
        if result.is_ok() {
            let bytes = written.iter().map(|span| span.size).sum();
            shared.counts.host_invalidated(written.len() as u64, bytes);
        }
    }
}

/// `vkFlushMappedMemoryRanges` over `spans`, counted: nothing for none.
fn flush_spans(shared: &DeviceShared, spans: &[Span]) {
    if spans.is_empty() {
        return;
    }
    let Some(_busy) = crate::exit_guard::enter() else {
        return;
    };
    let ranges: Vec<vk::MappedMemoryRange<'static>> = spans.iter().map(Span::range).collect();
    // SAFETY: as `SyncPlan::complete`'s.
    let result = unsafe { shared.vk().flush_mapped_memory_ranges(&ranges) };
    if result.is_ok() {
        let bytes = spans.iter().map(|span| span.size).sum();
        shared.counts.host_flushed(spans.len() as u64, bytes);
    }
}

/// What a test forces on the imports its thread makes: that every host import is treated as
/// being in non-coherent memory (lavapipe, where the tests run, lists only coherent types), that
/// the map of an import fails, and the mode, which the environment otherwise gives. Never
/// compiled outside the crate's tests.
#[cfg(test)]
pub(crate) mod forced {
    use super::Mode;
    use std::cell::Cell;

    #[derive(Clone, Copy, Default)]
    struct Forcing {
        noncoherent: bool,
        map_fails: bool,
        mode: Option<Mode>,
    }

    thread_local! {
        static FORCED: Cell<Forcing> = const {
            Cell::new(Forcing { noncoherent: false, map_fails: false, mode: None })
        };
    }

    /// Whether this thread's imports are treated as non-coherent.
    pub(crate) fn noncoherent() -> bool {
        FORCED.with(Cell::get).noncoherent
    }

    /// Whether this thread's imports fail to map.
    pub(crate) fn map_fails() -> bool {
        FORCED.with(Cell::get).map_fails
    }

    /// The mode forced on this thread's new devices, if any.
    pub(crate) fn mode() -> Option<Mode> {
        FORCED.with(Cell::get).mode
    }

    /// Force the imports this thread makes, and the devices it makes, until the guard is dropped.
    #[cfg(any(m2v_gpu, m2v_cpu_icd))]
    pub(crate) fn force(map_fails: bool, mode: Option<Mode>) -> Forced {
        FORCED.with(|forced| {
            forced.set(Forcing {
                noncoherent: true,
                map_fails,
                mode,
            })
        });
        Forced
    }

    /// Ends the forcing when dropped.
    #[cfg(any(m2v_gpu, m2v_cpu_icd))]
    pub(crate) struct Forced;

    #[cfg(any(m2v_gpu, m2v_cpu_icd))]
    impl Drop for Forced {
        fn drop(&mut self) {
            FORCED.with(|forced| forced.set(Forcing::default()));
        }
    }
}
