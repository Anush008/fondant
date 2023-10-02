"""
This component downloads images based on URLs, and resizes them based on various settings like
minimum image size and aspect ratio.
"""
import io
import logging
import typing as t
import urllib.request
from multiprocessing.pool import ThreadPool
from threading import Semaphore

import pandas as pd
from fondant.component import PandasTransformComponent
from resizer import Resizer

logger = logging.getLogger(__name__)


class DownloadImagesComponent(PandasTransformComponent):
    """Component that downloads images based on URLs."""

    def __init__(self,
         *_,
         timeout: int,
         retries: int,
         n_connections: int,
         image_size: int,
         resize_mode: str,
         resize_only_if_bigger: bool,
         min_image_size: int,
         max_aspect_ratio: float,
    ) -> None:
        """Component that downloads images from a list of URLs and executes filtering and resizing.

        Args:
            timeout: Maximum time (in seconds) to wait when trying to download an image.
            retries: Number of times to retry downloading an image if it fails.
            n_connections: Number of concurrent connections opened per process. Decrease this
                number if you are running into timeout errors. A lower number of connections can
                increase the success rate but lower the throughput.
            image_size: Size of the images after resizing.
            resize_mode: Resize mode to use. One of "no", "keep_ratio", "center_crop", "border".
            resize_only_if_bigger: If True, resize only if image is bigger than image_size.
            min_image_size: Minimum size of the images.
            max_aspect_ratio: Maximum aspect ratio of the images.

        Returns:
            Dask dataframe
        """
        self.timeout = timeout
        self.retries = retries
        self.thread_count = n_connections
        self.resizer = Resizer(
            image_size=image_size,
            resize_mode=resize_mode,
            resize_only_if_bigger=resize_only_if_bigger,
            min_image_size=min_image_size,
            max_aspect_ratio=max_aspect_ratio,
        )

    def download_image(self, url: str) -> t.Optional[io.BytesIO]:
        url = url.strip()

        user_agent_string = (
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        )
        user_agent_string += " (compatible; +https://github.com/ml6team/fondant)"
        headers = {"User-Agent": user_agent_string}

        image_stream = None
        try:
            request = urllib.request.Request(url, data=None, headers=headers)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                image_stream = io.BytesIO(response.read())
            return image_stream
        except Exception as e:
            logger.warning(f"Skipping {url}: {repr(e)}")
            if image_stream is not None:
                image_stream.close()
            image_stream = None
        return image_stream

    def download_image_with_retry(self, id_: str,  url: str) \
            -> t.Tuple[str, t.Optional[io.BytesIO]]:
        for _ in range(self.retries + 1):
            image_stream = self.download_image(url)
            if image_stream is not None:
                return id_, image_stream
        return id_, None

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Downloading {len(dataframe)} images...")

        semaphore = Semaphore(self.thread_count * 2)

        def data_generator():
            for row in dataframe["images"]["url"].items():
                semaphore.acquire()
                yield row

        results: t.List[t.Tuple[str, bytes, int, int]] = []
        with ThreadPool(self.thread_count) as thread_pool:
            for id_, image_stream in thread_pool.imap_unordered(
                lambda row: self.download_image_with_retry(*row),
                data_generator(),
            ):
                if image_stream is not None:
                    image, width, height = self.resizer(image_stream)
                    image_stream.close()
                    del image_stream
                else:
                    image, width, height = None, None, None

                semaphore.release()
                results.append((id_, image, width, height))

        columns = ["id", "data", "width", "height"]
        if results:
            results_df = pd.DataFrame(results, columns=columns)
        else:
            results_df = pd.DataFrame(columns=columns)

        results_df = results_df.dropna()
        results_df = results_df.set_index("id", drop=True)
        results_df.columns = pd.MultiIndex.from_product([["images"], results_df.columns])

        return results_df
