pub mod proto {
    pub mod tensorboard {
        include!("tensorboard.pb.rs");
    }
}

pub mod masked_crc;
pub mod summary;
pub mod tf_record;
pub mod writer;

#[cfg(test)]
mod scripted_reader;
