use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::{Duration, SystemTime};

use rand::prelude::*;

use tensorboard_writer::{SummaryBuilder, TensorboardWriter};

fn main() -> std::io::Result<()> {
    let outfile = std::env::args_os().nth(1).unwrap_or_else(|| {
        eprintln!("fatal: specify OUTFILE as first argument");
        std::process::exit(1);
    });
    if !outfile.to_string_lossy().contains("tfevents") {
        eprintln!("warn: filenames must include 'tfevents' to show up in TensorBoard");
    }

    // Open a file and bind a writer to it.
    let writer = BufWriter::new(File::create(outfile)?);
    let mut writer = TensorboardWriter::new(writer);

    // Write a file header---not strictly necessary, but useful for troubleshooting.
    writer.write_file_version()?;

    // Go through your training loop, and at each step log some TensorBoard summaries...
    const STEPS: usize = 50;
    for step in 0..STEPS {
        let nweights = 10000;
        // get your values from somewhere... here, we just make them up
        let loss: f32 = 10.0 / (step + 1) as f32;
        let weights_layer1 = normal(nweights, step as f64, 10.0 / (step as f64 + 1.0).sqrt());
        let weights_final = normal(nweights, 3.0, 10.0);
        // (suppose this is a generative model trying to generate names of colors...)
        let samples: Vec<String> =
            std::iter::repeat_with(|| sample_color_name((1.0 / (step + 1) as f64).sqrt()))
                .take(5)
                .collect();

        const NUM_HISTOGRAM_BINS: usize = 30;
        // On every step, write the current loss and the weights distributions.
        let mut sb = SummaryBuilder::new()
            .scalar("loss", loss)
            .histogram("weights/layer1", NUM_HISTOGRAM_BINS, &weights_layer1)
            .histogram("weights/final", NUM_HISTOGRAM_BINS, &weights_final)
            .text_ndarray("sample_outputs", samples.as_slice(), &[samples.len()]);
        // On the first step, additionally dump some hyperparameter info.
        if step == 0 {
            let desc = format!(
                "\
                # Hyperparameters\n\
                \n\
                - `layers`: 2\n\
                - `nweights`: {nweights}\n\
                ",
                nweights = nweights
            );
            sb = sb.text("hparams", &desc);
        }
        let summ = sb.build();

        // Write summaries to file.
        writer.write_summary(SystemTime::now(), step as i64, summ)?;
        writer.get_mut().flush()?;

        println!("finished training step {}", step);
        // Simulate training taking a while, so that you can see your TensorBoard updating live.
        std::thread::sleep(Duration::from_millis(250));
    }

    // Make sure we can flush to disk without error.
    let bufw = writer.into_inner();
    bufw.into_inner().map_err(|e| e.into_error())?.sync_all()?;

    println!("wrote event file with {} steps", STEPS);

    Ok(())
}

fn normal(n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let dist = rand_distr::Normal::new(mu, sigma).unwrap();
    let mut result = Vec::with_capacity(n);
    for _ in 0..n {
        result.push(dist.sample(&mut rng));
    }
    result
}

fn sample_color_name(p_error: f64) -> String {
    use rand::distributions::{Bernoulli, Slice};
    const GROUND_TRUTH: &[&str] = &["red", "orange", "yellow", "green", "blue", "purple"];

    let flip = Bernoulli::new(p_error).unwrap();
    let replacements = Slice::new("bcdfghjklmnpqrstvwxyz".as_bytes()).unwrap();

    let mut rng = rand::thread_rng();
    let base_word = GROUND_TRUTH.choose(&mut rng).unwrap();
    base_word
        .chars()
        .map(|c| {
            if rng.sample(flip) {
                *rng.sample(replacements) as char
            } else {
                c
            }
        })
        .collect()
}
