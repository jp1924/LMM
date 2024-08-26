from transformers import LlavaNextConfig


class LlavaNextDocConfig(LlavaNextConfig):
    def __init__(
        self,
        hreducer_hidden_size=1024,
        hreducer_layer_norm=1e-6,
        hreducer_conv_shape="1x4",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hreducer_hidden_size = hreducer_hidden_size
        self.hreducer_layer_norm = hreducer_layer_norm
        self.hreducer_conv_shape = hreducer_conv_shape
