use std::io::{self, Write};
use std::time::SystemTime;

use prost::Message;

use crate::proto::tensorboard as pb;
use crate::tf_record::TfRecord;

pub struct TensorboardWriter<W> {
    writer: W,
}

impl<W> TensorboardWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

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

impl<W: Write> TensorboardWriter<W> {
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
        const WRITER: &str = "rust:wchargin/tensorboard-writer";

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
