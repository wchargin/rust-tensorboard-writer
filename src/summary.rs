use super::proto::tensorboard as pb;
use pb::summary::value::Value as InnerValue;

pub struct SummaryBuilder {
    summary: pb::Summary,
}

impl SummaryBuilder {
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
}
