import math

import torch
import torch.nn as nn
from einops import rearrange

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    LlavaNextForConditionalGeneration,
)
from transformers.activations import ACT2FN

from .configuration_llava_next_doc import LlavaNextDocConfig


class LlavaNextDocHReducer(nn.Module):
    def __init__(self, config: LlavaNextDocConfig) -> None:
        super().__init__()
        self.config = config
        self.conv_shape = (
            int(self.config.hreducer_conv_shape.split("x")[0]),
            int(self.config.hreducer_conv_shape.split("x")[1]),
        )
        self.layer_norm = torch.nn.LayerNorm(self.config.hreducer_hidden_size, eps=self.config.hreducer_layer_norm)
        self.conv_patch = self.conv_shape[0] * self.conv_shape[1]
        # Wav2Vec와 같이
        self.reducer_before = torch.nn.Sequential(
            nn.Conv2d(
                self.config.hreducer_hidden_size,
                self.conv_patch * self.config.hreducer_hidden_size,
                kernel_size=self.conv_shape,
                stride=self.conv_shape,
                bias=False,
            ),
            nn.GELU(),
        )
        ## reduce visual feature length with a conv layer
        self.reducer = nn.Conv2d(
            self.config.hreducer_hidden_size,
            self.config.hreducer_hidden_size,
            kernel_size=self.conv_shape,
            stride=self.conv_shape,
            bias=False,
        )
        ## align visual features with language embedding with fc
        self.visual_fc = torch.nn.Linear(self.config.hreducer_hidden_size, config.text_config.hidden_size)
        self.vit_eos = torch.nn.Parameter(torch.randn(1, 1, config.text_config.hidden_size))
        self.gradient_checkpointing = False

    def forward(self, encoder_hidden_states=None):
        r"""
        Processes the encoder hidden states to reduce visual feature length and align them with language embeddings.

        Args:
            encoder_hidden_states (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional):
                Batch size is the number of all images (global+crop) in a batch.
                Sequence of hidden-states at the output of the last layer of the encoder.

        Returns:
            torch.FloatTensor: The processed sequence output with reduced visual feature length and aligned with language embeddings.

        Example:
            >>> config = Config()  # Assuming Config is already defined
            >>> model = MPLUGDocOwlHReducer(config)
            >>> encoder_hidden_states = torch.randn(batch_size, sequence_length, hidden_size)  # Example tensor
            >>> output = model.forward(encoder_hidden_states)
        """

        # Remove the first cls token
        # Shape: (batch_size, sequence_length - 1, hidden_size)
        encoder_hidden_states = encoder_hidden_states[:, 1:, :]

        # B - batch_size, L - sequence_length, C - hidden_size
        batch_size, seq_len, hidden_size = encoder_hidden_states.shape

        # Calculate height assuming seq_len is a square number
        encoder_hidden_states = rearrange(encoder_hidden_states, "B (H W) D -> B D H W", H=int(math.sqrt(seq_len)))

        # Apply reducer (e.g., a convolution)
        # Shape: (batch_size, reduced_depth, height, width_reduced)
        reduced_states = self.reducer_before(encoder_hidden_states)

        # B - batch_size, reduced_depth - reduced depth dimension, height - height, width_reduced - reduced width
        batch_size, reduced_depth, height, width_reduced = reduced_states.shape

        # Number of patches in width
        num_patches = self.conv_patch

        # New depth dimension
        depth = reduced_depth // num_patches

        # Reshape reduced states
        # Shape: (batch_size, num_patches, depth, height, width_reduced)
        reduced_states = reduced_states.view(batch_size, num_patches, depth, height, width_reduced)

        # Shape: (batch_size, depth, height, width_reduced, num_patches)
        reduced_states = reduced_states.permute(0, 2, 3, 4, 1)

        # Shape: (batch_size, depth, height, width)
        reduced_states = reduced_states.reshape(batch_size, depth, height, width_reduced * num_patches)

        # Apply final reducer (e.g., a convolution)
        sequence_output = self.reducer(reduced_states)  # Shape: (batch_size, final_depth, final_height, final_width)

        # Flatten and transpose to (batch_size, seq_length_reduced, final_depth)
        # Shape: (batch_size, seq_length_reduced, final_depth)
        sequence_output = sequence_output.flatten(2).transpose(1, 2)

        # Shape: (seq_length_reduced, batch_size, final_depth)
        sequence_output = sequence_output.transpose(0, 1).contiguous()

        # Apply final fully connected layer
        # Shape: (seq_length_reduced, batch_size, final_hidden_size)
        sequence_output = self.visual_fc(sequence_output)

        # Shape: (batch_size, seq_length_reduced, final_hidden_size)
        sequence_output = sequence_output.transpose(0, 1).contiguous()

        # Concatenate end-of-sequence token
        # Shape: (batch_size, seq_length_reduced + 1, final_hidden_size)
        sequence_output = torch.cat([sequence_output, self.vit_eos.repeat(batch_size, 1, 1)], dim=1)

        return sequence_output


# Copied from transformers.models.llava.modeling_llava.LlavaMultiModalProjector with Llava->LlavaNext
class LlavaNextDocMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaNextDocConfig):
        super().__init__()

        self.config = config
        self.linear_1 = nn.Linear(
            self.config.vision_config.hidden_size, self.config.text_config.hidden_size, bias=True
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.hidden_size, bias=True)

        self.reducer = LlavaNextDocHReducer(config)

    def forward(self, image_features):
        hidden_states = self.reducer(image_features)
        # hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaNextDocForConditionalGeneration(LlavaNextForConditionalGeneration):
    config_class = LlavaNextDocConfig

    def __init__(self, config: LlavaNextDocConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaNextDocMultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"  # set it to left by default, user can use setter to change padding_sides
        self.post_init()

    def pack_image_features(self, image_features, _, image_newline=None):
        """
        Reshape, unpad and then pack each image_feature into a single image_features tensor containing all visual vectors.

        Args:
            image_features (`List[torch.Tensor]` of length num_images, each of shape `(num_patches, image_length, embed_dim)`)
                List of image feature tensor, each contains all the visual feature of all patches.
            _image_sizes_ (`torch.Tensor` of shape `(num_images, 2)`)
                Actual image size of each images (H, W).
                LlavaNextDoc 에선 필요 없는 값.
            image_newline (`torch.Tensor` of shape `(embed_dim)`)
                New line embedding vector.
        Returns:
            image_features (`torch.Tensor` of shape `(all_feat_len, embed_dim)`)
            feature_lens (`List[int]`)
                token length of each image in image_features
        """
        new_image_features = []
        feature_lens = []
        for _, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[None, None, :].expand(1, *image_feature.shape[1:]).to(image_feature.dtype),
                        ),
                        dim=0,
                    )
                image_feature = image_feature.flatten(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens
