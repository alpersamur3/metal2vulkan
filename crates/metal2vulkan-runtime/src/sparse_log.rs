//! The runtime's last sparse binds and the chunks of gathered pages it last imported and freed
//! (`gather.rs`), kept for the line a device's loss says (`Device::mark_lost`): a diagnostic that
//! records, and prints only at a loss.
//!
//! A driver that loses the device for a GPU fault or a refused bind names no object (Turnip over
//! KGSL: "GPU faulted or hung", "bind submit failed"), and the runtime's later calls say only that
//! the device was lost. The line names where the runtime first saw the loss and lists, oldest
//! first, the last [`KEPT`] binds (the resource, its ranges' bytes and memories, the internal
//! timeline's value, whether the driver refused it) and the last [`KEPT`] chunk imports and frees
//! (the memory, the region's bytes), each with how long before the loss it came: a chunk's memory
//! freed while a bind still named it, or a bind just before a fault, shows there.

use ash::vk::{self, Handle};
use std::collections::VecDeque;
use std::fmt::Write as _;
use std::panic::Location;
use std::sync::Mutex;
use std::time::Instant;

/// How many binds, and how many chunk events, the log keeps.
const KEPT: usize = 16;

/// A sparse bind the runtime made.
struct Bind {
    at: Instant,
    image: bool,
    resource: u64,
    ranges: usize,
    bytes: u64,
    /// The distinct memories its ranges name, the first few.
    memories: Vec<u64>,
    value: u64,
    refused: bool,
}

/// A chunk of gathered pages imported or freed.
struct Chunk {
    at: Instant,
    freed: bool,
    memory: u64,
    first: u64,
    size: u64,
}

#[derive(Default)]
struct Log {
    binds: VecDeque<Bind>,
    chunks: VecDeque<Chunk>,
}

/// The log: a leaf lock, taken for a push or the loss's line alone.
static LOG: Mutex<Log> = Mutex::new(Log {
    binds: VecDeque::new(),
    chunks: VecDeque::new(),
});

fn log() -> std::sync::MutexGuard<'static, Log> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// Note a `vkQueueBindSparse` of `ranges` to `resource` (an image where `image`), which signals
/// the internal timeline's `value`; `refused` where it failed.
pub(crate) fn bound(
    image: bool,
    resource: u64,
    ranges: &[vk::SparseMemoryBind],
    value: u64,
    refused: bool,
) {
    let bytes = ranges.iter().map(|r| r.size).sum();
    let mut memories: Vec<u64> = Vec::new();
    for range in ranges {
        let memory = range.memory.as_raw();
        if !memories.contains(&memory) {
            memories.push(memory);
            if memories.len() == 4 {
                break;
            }
        }
    }
    let bind = Bind {
        at: Instant::now(),
        image,
        resource,
        ranges: ranges.len(),
        bytes,
        memories,
        value,
        refused,
    };
    let mut log = log();
    if log.binds.len() == KEPT {
        log.binds.pop_front();
    }
    log.binds.push_back(bind);
}

/// Note the chunk `memory` of `size` bytes from `first` imported, or freed where `freed`.
#[cfg(unix)]
pub(crate) fn chunk(freed: bool, memory: vk::DeviceMemory, first: u64, size: u64) {
    let chunk = Chunk {
        at: Instant::now(),
        freed,
        memory: memory.as_raw(),
        first,
        size,
    };
    let mut log = log();
    if log.chunks.len() == KEPT {
        log.chunks.pop_front();
    }
    log.chunks.push_back(chunk);
}

/// The line a device's loss says, first seen at `at`: the log's binds and chunks, oldest first,
/// each with how long before now it came.
pub(crate) fn lost_line(at: &Location<'_>) -> String {
    let now = Instant::now();
    let ago = |then: Instant| now.saturating_duration_since(then).as_secs_f64() * 1000.0;
    let log = log();
    let mut line = format!(
        "metal2vulkan: the device was lost, first seen at {}:{}",
        at.file(),
        at.line()
    );
    if log.binds.is_empty() && log.chunks.is_empty() {
        line.push_str("; no sparse bind or chunk of gathered pages before it");
        return line;
    }
    let _ = write!(
        line,
        "; the runtime's last {} sparse binds:",
        log.binds.len()
    );
    for b in &log.binds {
        let memories: Vec<String> = b.memories.iter().map(|m| format!("{m:#x}")).collect();
        let _ = write!(
            line,
            "\n  {:.1} ms before: {} {:#x}, {} ranges of {} bytes in memory {}, timeline {}{}",
            ago(b.at),
            if b.image { "image" } else { "buffer" },
            b.resource,
            b.ranges,
            b.bytes,
            memories.join(" "),
            b.value,
            if b.refused { ", refused" } else { "" }
        );
    }
    let _ = write!(
        line,
        "\n  and its last {} chunks of gathered pages:",
        log.chunks.len()
    );
    for c in &log.chunks {
        let _ = write!(
            line,
            "\n  {:.1} ms before: memory {:#x} {}, {} bytes from {:#x}",
            ago(c.at),
            c.memory,
            if c.freed { "freed" } else { "imported" },
            c.size,
            c.first
        );
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The loss's line names its place, then each bind and chunk event noted before it, with the
    /// bind's ranges, bytes, memories and value, and whether the driver refused it.
    #[test]
    fn the_loss_line_lists_the_binds_and_chunks_before_it() {
        let memory = vk::DeviceMemory::from_raw(0xabc0);
        let ranges = [
            vk::SparseMemoryBind::default().size(0x4000).memory(memory),
            vk::SparseMemoryBind::default()
                .resource_offset(0x4000)
                .size(0x1000)
                .memory(memory),
        ];
        bound(true, 0x5150, &ranges, 77, true);
        #[cfg(unix)]
        chunk(true, memory, 0x7000_0000, 16 << 20);
        let line = lost_line(Location::caller());
        assert!(
            line.starts_with("metal2vulkan: the device was lost, first seen at "),
            "{line}"
        );
        assert!(line.contains(file!()), "{line}");
        assert!(
            line.contains(
                "image 0x5150, 2 ranges of 20480 bytes in memory 0xabc0, timeline 77, refused"
            ),
            "{line}"
        );
        #[cfg(unix)]
        assert!(
            line.contains("memory 0xabc0 freed, 16777216 bytes from 0x70000000"),
            "{line}"
        );
    }
}
