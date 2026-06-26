from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
import imageio.v3 as iio

if TYPE_CHECKING:
    import os
    from typing import Any, Literal, Protocol

    import numpy.typing as npt

    class FileLike(Protocol): ...

    class LockType(Protocol):
        def __enter__(self) -> Any: ...

        def acquire(self, blocking: bool = True, timeout: int = -1) -> bool: ...
        def release(self) -> None: ...

    IndexerType = int | slice | npt.NDArray[np.integer]
    FilenameOrObjectType = str | os.PathLike | bytes | FileLike


@dataclass
class ImageIOBackendArray(xr.backends.BackendArray):
    filename_or_obj: FilenameOrObjectType

    shape: tuple[int]
    dtype: npt.DtypeLike

    lock: LockType

    def __getitem__(self, key: tuple[IndexerType]):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self.basic_indexing,
        )

    def basic_indexing(self, key: tuple[IndexerType]) -> npt.NDArray:
        import imageio.v3 as iio

        with self.lock, iio.imopen(self.filename_or_obj, io_mode='r') as f:
            if key == (slice(None),) * len(self.shape):
                return f.read()

            first_indexer = key[0]
            if isinstance(first_indexer, int):
                data = f.read(index=first_indexer, mode='P', writable=True)

                remaining_indexers = key[1:]
            else:
                if isinstance(first_indexer, slice):
                    indices = range(*first_indexer.indices(self.shape[0]))
                else:
                    indices = first_indexer

                data = np.concatenate([f.read(index=index, mode='P') for index in indices], axis=0)

                remaining_indexers = (..., *key[1:])

            return data[remaining_indexers]


class ImageIOBackend(xr.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj: FilenameOrObjectType,
        *,
        drop_variables: bool | None = None,
        mode: Literal['grayscale', 'color'] = 'color',
    ) -> xr.Dataset:

        with iio.imopen(filename_or_obj, io_mode='r') as f:
            properties = f.properties()
            metadata = f.metadata()

            dims = ['time', 'height', 'width', 'color']

            background = metadata['background']
            duration = metadata['duration']
            loop = metadata['loop']

            shape = properties.shape
            dtype = properties.dtype

        if isinstance(duration, (int, float)):
            time_values = np.timedelta64(duration, 'ms') * np.arange(shape[0])
        else:
            time_values = np.array(duration, dtype='timedelta64[ms]')

        time = xr.indexes.PandasIndex(pd.Index(time_values), dim='time')

        backend_array = ImageIOBackendArray(
            filename_or_obj=filename_or_obj,
            shape=shape,
            dtype=dtype,
            lock=xr.backends.locks.SerializableLock(),
        )
        data = xr.core.indexing.LazilyIndexedArray(backend_array)

        var = xr.Variable(
            dims=dims,
            data=data,
            attrs={'loop': loop},
            encoding={
                'preferred_chunks': dict(zip(dims, (1, *shape[1:]))),
                'fill_value': background,
            },
        )
        coords = xr.Coordinates.from_xindex(time).assign(color=['red', 'green', 'blue'])

        return xr.Dataset({'data': var}, coords=coords)

    
    def write_dataset_gif(self,
                  dataset: xr.Dataset,
                  out_filename: str,
                  variable: str,
                  time_dim: str,
                  plot: bool = False):
        """Writes a xr.Dataset to a GIF. xr.Dataset must have a time dimension greater than 1.

        :param dataset: xr.Dataset containing data variables
        :param out_filename: GIF filename to write out to. 
        :param variable: data variable to plot and covert to GIF
        :param time_dim: name of time dimension
        :param plot: when True, writes and plot image first as JPG and create GIF from plots, defaults to False
        """
     
        frames_array = np.stack([dataset[variable][x] for x in range(len(dataset[time_dim]))], axis=0)
        iio.imwrite(out_filename, frames_array, extension='.gif', loop=0, duration=100, background=1)
        

    def write_jpg_gif(self,
                  image_names: list,
                  out_filename: str):

        """ Writes a list of images to a GIF.

        :param out_filename: GIF filename to write out to. 
        """
        print(image_names)
        frames = np.stack([iio.imread(image) for image in image_names], axis=0)
        print(frames)
        iio.imwrite(out_filename, frames, extension='.gif', loop=0, duration=100, background=1)
