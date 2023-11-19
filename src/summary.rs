use super::proto::tensorboard as pb;
use pb::summary::value::Value as InnerValue;

#[derive(Default)]
pub struct SummaryBuilder {
    summary: pb::Summary,
}

impl SummaryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn build(self) -> pb::Summary {
        self.summary
    }

    pub fn value(mut self, value: pb::summary::Value) -> Self {
        self.summary.value.push(value);
        self
    }

    fn tag_and_inner_value(self, tag: &str, inner: InnerValue) -> Self {
        let mut outer = pb::summary::Value::default();
        outer.tag = tag.to_string();
        outer.value = Some(inner);
        self.value(outer)
    }

    pub fn scalar(self, tag: &str, scalar: f32) -> Self {
        self.tag_and_inner_value(tag, InnerValue::SimpleValue(scalar))
    }

    pub fn histogram(self, tag: &str, bins: usize, values: &[f64]) -> Self {
        let mut histo = pb::HistogramProto::default();
        if !values.is_empty() && bins > 0 {
            histo.min = values.iter().copied().min_by(f64::total_cmp).unwrap();
            histo.max = values.iter().copied().max_by(f64::total_cmp).unwrap();
            // `bucket` has the counts in each bucket
            histo.bucket = vec![0.0; bins];
            // `bucket_limit` has the right edge of each bucket
            histo.bucket_limit = Vec::with_capacity(bins);
            let bucket_width = (histo.max - histo.min) / bins as f64;
            for i in 0..bins {
                histo
                    .bucket_limit
                    .push(histo.min + (i + 1) as f64 * bucket_width);
            }
            for &z in values {
                let idx = f64::floor((z - histo.min) / bucket_width);
                // Clamp in case of any floating point weirdness.
                let idx = idx.clamp(0.0, (bins - 1) as f64);
                histo.bucket[idx as usize] += 1.0;
            }
            // `histo` has other fields, like `sum` and `sum_squares`,
            // but they don't actually matter :^)
        }
        self.tag_and_inner_value(tag, InnerValue::Histo(histo))
    }
}
