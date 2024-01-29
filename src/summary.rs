use super::proto::tensorboard as pb;
use pb::summary::value::Value as InnerValue;

/// Builder for constructing TensorBoard `Summary` protocol buffers.
///
/// To use this builder, construct an instance with [`new`][Self::new], chain builder methods like
/// [`scalar`][Self::scalar] and [`histogram`][Self::histogram], and then retrieve the underlying
/// summary value with [`build`][Self::build]:
///
/// ```
/// use tensorboard_writer::SummaryBuilder;
///
/// # let weights_layer1 = [1.0, 2.0, 3.0];
/// let summ = SummaryBuilder::new()
///     .scalar("loss", 123.0)
///     .histogram("weights/layer1", 30, &weights_layer1)
///     .build();
/// ```
///
/// For more precise control over the values created, you can use the [`value`][Self::value]
/// builder to pass a raw TensorBoard `Summary.Value` protobuf that you've prepared ahead of time.
#[derive(Default)]
pub struct SummaryBuilder {
    summary: pb::Summary,
}

impl SummaryBuilder {
    /// Creates an empty summary with no values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Finishes this builder and returns the summary that's been constructed.
    pub fn build(self) -> pb::Summary {
        self.summary
    }

    /// Adds an arbitrary [`tensorboard.Summary.Value`][pb::summary::Value] protobuf value.
    pub fn value(mut self, value: pb::summary::Value) -> Self {
        self.summary.value.push(value);
        self
    }

    fn build_value(self, tag: &str, inner: InnerValue, meta: Option<pb::SummaryMetadata>) -> Self {
        let mut outer = pb::summary::Value::default();
        outer.tag = tag.to_string();
        outer.value = Some(inner);
        outer.metadata = meta;
        self.value(outer)
    }

    /// Adds a scalar summary.
    pub fn scalar(self, tag: &str, scalar: f32) -> Self {
        self.build_value(tag, InnerValue::SimpleValue(scalar), None)
    }

    /// Adds a rank-0 text summary with a single string. The text is interpreted as Markdown.
    ///
    /// This can be used to log actual model outputs (e.g., predictions on some sample data at each
    /// step) or just as a general "escape hatch" to dump any human-readable data to TensorBoard.
    pub fn text<T: AsRef<[u8]>>(self, tag: &str, text: &T) -> Self {
        self.text_ndarray(tag, &[text], &[])
    }

    /// Adds a text summary with a string or tensor of strings. The `text` vector should be in
    /// row-major order with shape given by `shape`. Text strings are interpreted as Markdown.
    ///
    /// # Panics
    ///
    /// In `cfg(debug)`, panics if the product of `shape` does not equal `text.len()`.
    pub fn text_ndarray<T: AsRef<[u8]>>(self, tag: &str, text: &[T], shape: &[usize]) -> Self {
        let string_val = text
            .iter()
            .map(|t| prost::bytes::Bytes::copy_from_slice(t.as_ref()))
            .collect();
        self.text_ndarray_mono(tag, string_val, shape)
    }

    fn text_ndarray_mono(
        self,
        tag: &str,
        string_val: Vec<prost::bytes::Bytes>,
        shape: &[usize],
    ) -> Self {
        let mut tensor = pb::TensorProto::default();

        // Check dimensions.
        if cfg!(debug) {
            let dim_product = shape
                .iter()
                .map(|&d| d as i64)
                .try_fold(1i64, |x, y| x.checked_mul(y));
            match dim_product {
                Some(n) if n == string_val.len() as i64 => (),
                None => panic!("bad shape: dimension product overflowed"),
                Some(n) => {
                    panic!(
                        "bad shape: dimension product is {} but vector has length {}",
                        n,
                        string_val.len()
                    );
                }
            }
        }

        tensor.tensor_shape = Some({
            let mut shape_pb = pb::TensorShapeProto::default();
            shape_pb.dim = shape
                .iter()
                .map(|&d| pb::tensor_shape_proto::Dim {
                    size: d as i64,
                    ..Default::default()
                })
                .collect();
            shape_pb
        });

        tensor.dtype = pb::DataType::DtString.into();
        tensor.string_val = string_val;

        let mut meta = pb::SummaryMetadata::default();
        const TEXT_PLUGIN_NAME: &str = "text";
        meta.plugin_data = Some(pb::summary_metadata::PluginData {
            plugin_name: TEXT_PLUGIN_NAME.to_string(),
            content: Default::default(),
        });

        self.build_value(tag, InnerValue::Tensor(tensor), Some(meta))
    }

    /// Adds a histogram summary, linearly bucketing the given `values` into the given number of
    /// `bins`.
    ///
    /// The `values` may be `f32`s or `f64`s, or any type that can be copied into an `f64`.
    pub fn histogram<T>(self, tag: &str, bins: usize, values: &[T]) -> Self
    where
        T: Into<f64> + Copy,
    {
        let mut histo = pb::HistogramProto::default();
        if !values.is_empty() && bins > 0 {
            histo.min = values
                .iter()
                .map(|z| Into::<f64>::into(*z))
                .min_by(f64::total_cmp)
                .unwrap();
            histo.max = values
                .iter()
                .map(|z| Into::<f64>::into(*z))
                .max_by(f64::total_cmp)
                .unwrap();
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
            for z in values {
                let idx = f64::floor((Into::<f64>::into(*z) - histo.min) / bucket_width);
                // Clamp in case of any floating point weirdness.
                let idx = idx.clamp(0.0, (bins - 1) as f64);
                histo.bucket[idx as usize] += 1.0;
            }
            // `histo` has other fields, like `sum` and `sum_squares`,
            // but they don't actually matter :^)
        }
        self.build_value(tag, InnerValue::Histo(histo), None)
    }
}
