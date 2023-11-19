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

    /// Writes a summary to the output stream, wrapped in an `Event` with the given step and wall
    /// time.
    pub fn write_summary(
        &mut self,
        wall_time: SystemTime,
        step: i64,
        summary: pb::Summary,
    ) -> io::Result<()> {
        let mut event = pb::Event::default();
        event.wall_time = wall_time
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_secs_f64();
        event.step = step;
        event.what = Some(pb::event::What::Summary(summary));
        self.write_event(&event)
    }
}