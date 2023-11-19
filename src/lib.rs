//! Simple utilities to write data that can be read by TensorBoard.
//!
//! A TensorBoard dashboard is made up of one or more time series, each of which describes a
//! different aspect of your model's training progress: e.g., you might have scalar time series
//! showing your loss and cross-entropy, and histogram time series for the weights and biases of
//! each layer of your model.
//!
//! The name of each time series is called its *tag*: e.g., the above time series might have names
//! `loss`, `xent`, `weights/0`, `biases/0`, `weights/1`, etc. The data in each time series is
//! indexed by *step* and/or *wall time*: the step is an arbitrary integer (interpretation is up to
//! you, but generally it's something that you increment once per training step or epoch or
//! something like that), and the wall time is a [`SystemTime`][std::time::SystemTime]. The
//! individual data points in these time series are called *summary values*, because they're meant
//! to summarize the progress of your model.
//!
//! Summaries are written to *TensorFlow event files*, which are exactly those files whose name
//! contains the literal string `tfevents`. (Yes, this is a bit jank. No, it can't be changed.) The
//! rest of the filename doesn't matter, but it generally contains information like timestamp,
//! hostname, and PID to make it unique and somewhat inspectable. An event file contains serialized
//! *TFRecords*, which are basically [protocol buffers][] with an accompanying CRC-32C (Castagnoli)
//! checksum.
//!
//! Event files are organized by directory into *runs*. One run typically represents one "training
//! run" of your model. Time series with the same name in different runs will be overlaid on the
//! same graph, so it is useful to use multiple runs when revising your model architecture,
//! changing hyperparameters, etc. so that you can compare the results easily in TensorBoard. Some
//! people also use separate runs for training, validation, and/or testing.
//!
//! So, for instance, you might have a filesystem structure like this:
//!
//! ```text
//! my_model/
//!     20230101/events.out.tfevents.1672561234.hostname
//!     20230102.layers=3/events.out.tfevents.1672646456.hostname
//!     20230102.layers=5/events.out.tfevents.1672646456.hostname
//!     20230102.layers=7/events.out.tfevents.1672646456.hostname
//! ```
//!
//! If each run has a scalar time series with tag `loss`, then TensorBoard will show a line chart
//! labeled "loss" with four series, called `20230101`, `20230102.layers=3`, etc. Nested
//! directories simply become runs with slashes in the name: e.g., `20230101/eval`.
//!
//! TensorBoard dashboards update live. You can launch TensorBoard before or after your event files
//! have been written, and the backend will poll the event files for new data periodically. Your
//! frontend can be set to reload data automatically or manually.
//!
//! This package provides ergonomic utilities for writing event files from Rust.
//!
//! [protocol buffers]: https://protobuf.dev/
//!
//! # Examples
//!
//! (See also `examples/simple.rs` for a script that you can run.)
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::{BufWriter, Write};
//! use std::time::SystemTime;
//!
//! use tensorboard_writer::{TensorboardWriter, SummaryBuilder};
//!
//! # fn main() -> std::io::Result<()> {
//! let file = File::create("run_123/tfevents.TIMESTAMP")?;
//! let mut writer = TensorboardWriter::new(BufWriter::new(file));
//!
//! writer.write_file_version()?;
//!
//! // train your model...
//! for step in 0..10 {
//!     let summ = SummaryBuilder::new()
//!         .scalar("loss", 0.1234)
//!         .histogram("weights", 30, &[0.123, 0.234, 0.345])
//!         .build();
//!     writer.write_summary(SystemTime::now(), step, summ)?;
//!     // flush the underlying `BufWriter` at each step
//!     // so that results show up in TensorBoard immediately
//!     writer.get_mut().flush()?;
//! }
//! # Ok(())
//! # }
//! ```

pub mod proto {
    pub mod tensorboard {
        include!("tensorboard.pb.rs");
    }
}

mod masked_crc;
mod summary;
mod writer;

pub mod tf_record;

pub use masked_crc::MaskedCrc;
pub use summary::SummaryBuilder;
pub use writer::Writer as TensorboardWriter;

#[cfg(test)]
mod scripted_reader;
