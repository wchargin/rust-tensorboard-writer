use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use prost::Message;

use crate::proto::tensorboard as pb;
use crate::tf_record::TfRecord;

/// Utility for writing TensorBoard event files.
///
/// A TensorBoard event file contains a sequence of (raw, undelimited) TFRecords. The data of each
/// TFRecord should be a serialized [`tensorboard.Event`][pb::Event] protocol buffer. Most events
/// contain summary values; you can use the [`SummaryBuilder`][crate::SummaryBuilder] utility to
/// build those.
pub struct Writer<W> {
    writer: W,
}

static GLOBAL_UID: AtomicU64 = AtomicU64::new(0);

/// Creates a unique name for an event file, incorporating sources of entropy including the
/// timestamp, hostname, process ID, and a per-process global counter.
fn event_file_name() -> OsString {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |dt| dt.as_secs());
    let hostname = hostname::get().unwrap_or_default();
    let pid = std::process::id();
    let uid = GLOBAL_UID.fetch_add(1, Ordering::Relaxed);

    let mut result = OsString::from(format!("events.out.tfevents.{now:010}."));
    result.push(hostname);
    result.push(format!(".{pid}.{uid}"));
    result
}

impl Writer<BufWriter<File>> {
    /// Creates a new TensorBoard event file in the given run directory.
    ///
    /// The run directory and its ancestors will be created if they do not exist.
    ///
    /// # Errors
    ///
    /// Errors if the run directory cannot be created, or in the unlikely event that the newly
    /// chosen name for the event file is already taken.
    pub fn new<P: AsRef<Path>>(run_directory: P) -> io::Result<Self> {
        let run_directory = run_directory.as_ref();
        std::fs::create_dir_all(run_directory)?;
        let filename = run_directory.join(event_file_name());
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(filename)?;
        Ok(Self::wrap(BufWriter::new(file)))
    }
}

impl<W> Writer<W> {
    /// Wraps an existing writer object. Usually you will want to use [`Writer::new`]; this method
    /// is appropriate if not writing to a file.
    pub fn wrap(writer: W) -> Self {
        Self { writer }
    }

    /// Gets a reference to the underlying writer.
    pub fn get_ref(&self) -> &W {
        &self.writer
    }

    /// Gets a mutable reference to the underlying writer.
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Unwraps this TensorBoard writer, returning the underlying writer.
    pub fn into_inner(self) -> W {
        self.writer
    }
}

fn time_f64(time: SystemTime) -> std::io::Result<f64> {
    Ok(time
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(io::Error::other)?
        .as_secs_f64())
}

impl<W: Write> Writer<W> {
    /// [Flushes][std::io::Write::flush] the underlying writer.
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    /// Writes a raw TFRecord to the output stream. You may find it more convenient to use
    /// [`write_event`][Self::write_event] instead, which computes the record checksum for you.
    pub fn write_record(&mut self, record: &TfRecord) -> io::Result<()> {
        record.write(&mut self.writer)
    }

    /// Writes an `Event` to the output stream.
    pub fn write_event(&mut self, event: &pb::Event) -> io::Result<()> {
        let data = event.encode_to_vec();
        let record = TfRecord::from_data(data);
        self.write_record(&record)
    }

    /// Writes a file version header event. This reads the current system time.
    pub fn write_file_version(&mut self) -> io::Result<()> {
        const FILE_VERSION: &str = "brain.Event:2";
        const WRITER: &str = "wchargin/rust-tensorboard-writer";

        let mut event = pb::Event::default();
        event.wall_time = time_f64(SystemTime::now())?;
        event.what = Some(pb::event::What::FileVersion(FILE_VERSION.to_string()));
        let mut source_metadata = pb::SourceMetadata::default();
        source_metadata.writer = WRITER.to_string();
        event.source_metadata = Some(source_metadata);
        self.write_event(&event)
    }

    /// Writes a summary to the output stream, wrapped in an `Event` with the given step and wall
    /// time.
    ///
    /// You may find it helpful to use the [`SummaryBuilder`][crate::SummaryBuilder] utility to
    /// construct the `summary` value.
    pub fn write_summary(
        &mut self,
        wall_time: SystemTime,
        step: i64,
        summary: pb::Summary,
    ) -> io::Result<()> {
        let mut event = pb::Event::default();
        event.wall_time = time_f64(wall_time)?;
        event.step = step;
        event.what = Some(pb::event::What::Summary(summary));
        self.write_event(&event)
    }
}
