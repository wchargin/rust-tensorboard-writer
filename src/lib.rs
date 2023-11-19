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
