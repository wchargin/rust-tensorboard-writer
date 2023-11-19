pub mod proto {
    pub mod tensorboard {
        include!("tensorboard.pb.rs");
    }
}

pub mod masked_crc;
mod scripted_reader;
pub mod tf_record;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
