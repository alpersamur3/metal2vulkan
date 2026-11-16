//! Why an import of host memory was refused: a one-line note for each refusal, made where the
//! refusal is.
//!
//! A no-copy buffer's import can be refused at a dozen places (`host_memory.rs`,
//! `buffer_descriptor.rs`, `memory.rs`), and most of them answer the same typed
//! `Unsupported`, `HostMemory`, so the caller cannot tell which check said no. A phone whose
//! Vulkan driver refused a 0x34000-byte import, with its dma-buf fallback closed by the system,
//! left only that typed refusal and a dead process.
//!
//! - **What a note holds.** `host import refused at <stage>: <numbers>`: the stage names the check
//!   (`window`, `alignment`, `skipped`, `unwritable`, `superset`, `buffer`, `binding`,
//!   `pointer-types`, `max-allocation`, `no-memory-type`, `allocate`, `bind`, `dma-buf`), and the
//!   numbers are the ones the check compared, in hex, so a refusal is read without the code at
//!   hand. A note is one line, at most [`LONGEST`] characters. The one note that is no refusal
//!   is `host import unsynced at noncoherent-map: ...` ([`note_as`]): an import in memory that is
//!   not coherent whose `vkMapMemory` failed or answered another pointer, made all the same and
//!   used with no cache maintenance (`host_sync.rs`). `no-memory-type` and `allocate` tell both
//!   tiers of the types an import tries (`memory.rs`'s `import_candidates`).
//! - **Where it goes.** To the calling thread's last [`KEPT`] notes, which
//!   [`take_host_import_notes`] drains (the caller reads them when it reports the refusal, on
//!   the thread that made the call: every refusal is made there), and once to stderr as
//!   `metal2vulkan: <note>`, at most [`PRINTED`] lines in a process, then one line saying the rest
//!   are not printed.
//! - **Only on a refusal.** A check that passes makes no note, and the numbers are formatted by a
//!   closure that runs only for a note ([`note`]), so an import the device takes costs nothing
//!   here.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicU32, Ordering};

/// The notes a thread keeps: its last few refusals, oldest dropped first.
pub(crate) const KEPT: usize = 4;

/// The notes a process prints, before it says that it prints no more.
pub(crate) const PRINTED: u32 = 64;

/// The longest note, in characters: a longer one ends in `...`.
pub(crate) const LONGEST: usize = 399;

/// What the process has printed: the notes, then the one line that says it stopped.
static PRINT_COUNT: AtomicU32 = AtomicU32::new(0);

/// A thread's last notes.
#[derive(Default)]
pub(crate) struct Ring {
    lines: VecDeque<String>,
}

impl Ring {
    /// An empty ring.
    pub(crate) const fn new() -> Ring {
        Ring {
            lines: VecDeque::new(),
        }
    }

    /// Keep `line`, dropping the oldest where [`KEPT`] are kept already.
    pub(crate) fn push(&mut self, line: String) {
        if self.lines.len() == KEPT {
            self.lines.pop_front();
        }
        self.lines.push_back(line);
    }

    /// Every note kept, oldest first, the ring empty after.
    pub(crate) fn take(&mut self) -> Vec<String> {
        self.lines.drain(..).collect()
    }
}

thread_local! {
    /// This thread's notes.
    static NOTES: RefCell<Ring> = const { RefCell::new(Ring::new()) };
}

/// What the next note does on stderr.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Printing {
    /// Print the note.
    Line,
    /// Print the line that says no more are printed, this once.
    Last,
    /// Print nothing.
    Quiet,
}

/// What the next note does on stderr, after `printed` notes were counted and with room for `cap`
/// of them: the counter stops at `cap + 1`, so it never wraps.
pub(crate) fn printing(printed: &AtomicU32, cap: u32) -> Printing {
    let counted = printed.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        (count <= cap).then_some(count.saturating_add(1))
    });
    match counted {
        Ok(count) if count < cap => Printing::Line,
        Ok(_) => Printing::Last,
        Err(_) => Printing::Quiet,
    }
}

/// `text` as one line of at most [`LONGEST`] characters: line breaks become spaces, and what
/// does not fit is cut where a character ends, ending in `...`.
pub(crate) fn one_line(text: String) -> String {
    let mut line: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    if line.chars().count() > LONGEST {
        let cut = line
            .char_indices()
            .nth(LONGEST - 3)
            .map_or(line.len(), |(at, _)| at);
        line.truncate(cut);
        line.push_str("...");
    }
    line
}

/// Say that a check of an import of host memory refused it, at `stage`: `detail`, run only here,
/// gives the numbers it compared. Keeps the note on this thread ([`take_host_import_notes`]) and
/// prints it once to stderr, as the module says.
pub(crate) fn note(stage: &'static str, detail: impl FnOnce() -> String) {
    note_as("refused", stage, detail);
}

/// [`note`] of an import that is not refused but made with a caveat: `host import <outcome> at
/// <stage>: <numbers>`. The only one so far is `unsynced` at `noncoherent-map`: an import in
/// memory that is not coherent whose map failed, used with no cache maintenance (`host_sync.rs`).
pub(crate) fn note_as(outcome: &'static str, stage: &'static str, detail: impl FnOnce() -> String) {
    let line = one_line(format!("host import {outcome} at {stage}: {}", detail()));
    // A thread that is ending keeps none.
    let _ = NOTES.try_with(|ring| ring.borrow_mut().push(line.clone()));
    let mut stderr = std::io::stderr().lock();
    // A closed stderr loses the line; a refusal is not made worse by it.
    let _ = match printing(&PRINT_COUNT, PRINTED) {
        Printing::Line => writeln!(stderr, "metal2vulkan: {line}"),
        Printing::Last => writeln!(
            stderr,
            "metal2vulkan: further host import refusals are not printed"
        ),
        Printing::Quiet => Ok(()),
    };
}

/// The notes this thread's refusals of host memory made since it last took them, oldest first: the
/// last four, each one line, `host import refused at <stage>: <the numbers the check
/// compared>`.
///
/// A no-copy buffer's import is refused as `Unsupported`, `HostMemory`, `NotRepresentable` by many
/// checks that answer alike (the import window, the buffer's binding in it, the memory types, the
/// driver's refusal of the pointer, the writability of the pages); the notes say which one, with
/// its numbers. Call it on the thread that made the call that failed, as the caller reports the
/// refusal: a thread's notes are its own, and a refusal that a dma-buf source then served leaves
/// its notes too. Each note is also printed once to stderr (`metal2vulkan: ...`), the first 64 in
/// a process.
pub fn take_host_import_notes() -> Vec<String> {
    NOTES
        .try_with(|ring| ring.borrow_mut().take())
        .unwrap_or_default()
}
