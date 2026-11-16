//! The import probe (`host_probe.rs`, `Device::probe_host_import`) on the device the cfg selects:
//! one facts line and one line for each of eight shapes, each starting with its id; on a CPU
//! device (lavapipe), which imports every shape the probe makes, the first four ok; and the
//! device as it was after: no import of the probe's left, no note.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on Unix, on the kind of device the cfg selects. Where no device exists it
//! says so and passes.

use super::blit_holds::{until, Scratch};
use crate::host_import_note::take_host_import_notes;
use crate::{Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions};

/// The lines' ids, after the facts line.
const IDS: [&str; 8] = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"];

/// The device the cfg selects, as `blit_holds::device` makes it, or `None` where the machine has
/// none.
fn device_here(cache: &std::path::Path) -> Option<Device> {
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
    match Device::system_default_with(&options) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("HOST-PROBE: no device on this machine: {error}: not run");
            None
        }
    }
}

#[test]
fn the_probe_answers_a_line_for_each_shape_and_leaves_the_device_as_it_was() {
    let scratch = Scratch::new("host-probe");
    let Some(device) = device_here(&scratch.0) else {
        return;
    };
    let live = device.work_counts().host_imports_live;
    let lines = device.probe_host_import();
    eprintln!("HOST-PROBE on {}:\n{}", device.name(), lines.join("\n"));

    if lines.len() == 1 {
        // A device that imports no host memory says why, in one line.
        assert!(
            lines[0].starts_with("facts: the device imports no host memory"),
            "{lines:?}"
        );
        return;
    }
    assert_eq!(lines.len(), 1 + IDS.len(), "{lines:#?}");
    // The importer's facts first.
    let facts = &lines[0];
    assert!(facts.starts_with("facts: page=0x"), "{facts}");
    for field in [" alignment=0x", " dma_buf=", " types=[0:0x"] {
        assert!(facts.contains(field), "{field}: {facts}");
    }
    // The memory types, then how many of the probe's imports lay in a type that is not coherent:
    // none on a device whose host-visible types are all coherent (lavapipe).
    let (types, noncoherent) = facts
        .rsplit_once(" noncoherent_imports=")
        .unwrap_or_else(|| panic!("{facts}"));
    assert!(types.ends_with(']'), "{facts}");
    let noncoherent: u64 = noncoherent.parse().unwrap_or_else(|_| panic!("{facts}"));
    if cfg!(m2v_cpu_icd) {
        assert_eq!(noncoherent, 0, "{facts}");
    }
    // A line for each shape, in order, an answer of one of three kinds.
    for (line, id) in lines[1..].iter().zip(IDS) {
        let answer = line
            .strip_prefix(&format!("{id} "))
            .unwrap_or_else(|| panic!("{id}: {line}"));
        let (_, answer) = answer
            .split_once(": ")
            .unwrap_or_else(|| panic!("{id}: {line}"));
        let kind = ["ok", "refused: ", "error: "]
            .iter()
            .any(|kind| answer.starts_with(kind));
        assert!(kind, "{line}");
        // One line each, whatever the notes said.
        assert!(!line.contains('\n'), "{line}");
        // A refusal says which check refused.
        if let Some(notes) = answer.strip_prefix("refused: ") {
            assert!(
                notes.contains("host import refused at ") || notes.contains("no note"),
                "{line}"
            );
        }
    }
    // A CPU ICD imports the pointers of these shapes: private and shared, one mapping and many.
    if cfg!(m2v_cpu_icd) {
        for line in &lines[1..5] {
            assert!(line.contains(": ok"), "{lines:#?}");
        }
    }

    // The device as it was: the probe's imports are gone, no note is left for the caller to read
    // as its own, and the device makes a buffer.
    until("the probe's imports went", || {
        device.work_counts().host_imports_live == live
    });
    assert_eq!(take_host_import_notes(), Vec::<String>::new());
    let buffer = device.new_buffer(4096, ResourceOptions::default());
    assert!(buffer.as_ref().err().is_none(), "{:?}", buffer.err());
}
