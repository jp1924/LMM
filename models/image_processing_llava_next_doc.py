import numpy as np

from transformers import LlavaNextImageProcessor
from transformers.image_transforms import resize
from transformers.image_utils import ChannelDimension


class LlavaNextDocImageProcessor(LlavaNextImageProcessor):
    def _resize_for_patching(
        self, image: np.array, target_resolution: tuple, resample, input_data_format: ChannelDimension
    ) -> np.array:
        """
        Resizes an image to a target resolution while maintaining aspect ratio.

        Args:
            image (np.array):
                The input image.
            target_resolution (tuple):
                The target resolution (height, width) of the image.
            resample (`PILImageResampling`):
                Resampling filter to use if resizing the image.
            input_data_format (`ChannelDimension` or `str`):
                The channel dimension format of the input image.

        Returns:
            np.array: The resized and padded image.
        """
        # Resize the image
        # 이렇게 하면 resize를 최대한 하지 않음.
        resized_image = resize(image, target_resolution, resample=resample, input_data_format=input_data_format)

        return resized_image
