use std::fs::File;
use std::io::BufWriter;
use std::time::{Duration, SystemTime};

use rand::prelude::*;

use tensorboard_writer as tw;

fn main() -> std::io::Result<()> {
    let outfile = std::env::args_os().nth(1).unwrap_or_else(|| {
        eprintln!("fatal: specify OUTFILE as first argument");
        std::process::exit(1);
    });
    if !outfile.to_string_lossy().contains("tfevents") {
        eprintln!("warn: filenames must include 'tfevents' to show up in TensorBoard");
    }

    let writer = BufWriter::new(File::create(outfile)?);
    let mut writer = tw::writer::TensorboardWriter::new(writer);

    writer.write_file_version()?;
    const STEPS: usize = 50;
    for step in 0..STEPS {
        // get your values from somewhere... here, we just make them up
        let loss: f32 = 10.0 / (step + 1) as f32;
        let weights_layer1: [f64; 10000] = normal(step as f64, 10.0 / (step as f64 + 1.0).sqrt());
        let weights_final: [f64; 10000] = normal(3.0, 10.0);

        const NUM_HISTOGRAM_BINS: usize = 30;
        let sum = tw::summary::SummaryBuilder::new()
            .scalar("loss", loss)
            .histogram("weights/layer1", NUM_HISTOGRAM_BINS, &weights_layer1)
            .histogram("weights/final", NUM_HISTOGRAM_BINS, &weights_final)
            .build();
        let fake_time = SystemTime::now() + Duration::from_secs(step as u64);
        writer.write_summary(fake_time, step as i64, sum)?;
    }

    // Make sure we can flush to disk without error.
    let bufw = writer.into_inner();
    bufw.into_inner().map_err(|e| e.into_error())?.sync_all()?;

    println!("wrote event file with {} steps", STEPS);

    Ok(())
}

fn normal<const N: usize>(mu: f64, sigma: f64) -> [f64; N] {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::new(mu, sigma).unwrap();
    std::array::from_fn(|_| dist.sample(&mut rng))
}
