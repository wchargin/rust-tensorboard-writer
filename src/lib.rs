pub mod proto {
    pub mod tensorboard {
        include!("tensorboard.pb.rs");
    }
}

pub mod masked_crc;
mod scripted_reader;
pub mod summary;
pub mod tf_record;
pub mod writer;
