"""
This component that calculates depth maps from input images
"""
import io
import itertools
import logging
import toolz

import dask
import dask.dataframe as dd
from PIL import Image
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, Pipeline

from fondant.component import TransformComponent
from fondant.logger import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


def transform(bytes: bytes,
              pipe: Pipeline) -> bytes:
    """transforms an image to a depth map

    Args:
        bytes (bytes): input image
        pipe (Pipeline): depth estimation pipeline

    Returns:
        bytes: depth map
    """
    # load image
    image = Image.open(io.BytesIO(bytes)).convert("RGB")

    # calculate depth map
    image = pipe(image)['depth']
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    depth_image = Image.fromarray(image)

    # save image to bytes
    output_bytes = io.BytesIO()
    depth_image.save(output_bytes, format='JPEG')

    return output_bytes.getvalue()


class DepthProcessorComponent(TransformComponent):
    """
    Component that calculates the depth map of an image
    """

    def transform(
        self,
        dataframe: dd.DataFrame,
    ) -> dd.DataFrame:
        """
        Args:
            dataframe: Dask dataframe

        Returns:
            Dask dataframe
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load depth estimation pipeline
        pipe = pipeline('depth-estimation', device=device)

        # calculate depth map
        dataframe['depth_data'] = dataframe['images_data'].apply(lambda x: transform(x, pipe), meta=('depth_data', 'bytes'))

        return dataframe


if __name__ == "__main__":
    component = DepthProcessorComponent.from_file()
    component.run()
