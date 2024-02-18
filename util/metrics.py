"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib

import numpy as np
import torch

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d, interpolate
import random


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def get_activations(
    dataloader,
    model,
    domain,
    batch_size,
    dims,
    device,
    nb_max_img,
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- dataloader  : JG-like dataloader (it can be a list of tensors too)
    -- model       : Instance of inception model
    -- domain      : image domain
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- nb_max_img  : number max of images used for activations compute

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(dataloader):
        print(
            (
                "Warning: batch size is bigger than the data size. "
                "Setting batch size to data size"
            )
        )
        batch_size = len(dataloader)

    pred_arr = torch.empty((len(dataloader), dims))

    start_idx = 0

    if nb_max_img < len(dataloader):
        if isinstance(dataloader, list):
            dataloader = dataloader[:nb_max_img]
        else:
            print(
                "Number of images limitation doesn't work with pytorch dataloaders, the full dataset will be used instead for activations computation."
            )

    for batch in tqdm(
        dataloader, total=len(dataloader) // batch_size, desc="activations"
    ):
        if isinstance(batch, dict) and domain is not None:
            img = batch[domain].to(device)
        else:
            img = batch.to(device)

        if len(img.shape) == 5:  # we're using temporal successive frames
            img = img[:, 1]

        with torch.no_grad():
            # Inceptionv3 works with 299 resolution images.
            img = interpolate(img, size=299, mode="bilinear")
            pred = model(img)
            if isinstance(pred, list):
                pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.

        if len(pred.shape) == 4:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def _compute_statistics_of_dataloader(
    path_sv,
    model,
    domain,
    batch_size,
    dims,
    device,
    dataloader,
    nb_max_img=float("inf"),
    root=None,
):
    if path_sv is not None and os.path.isfile(path_sv):
        print("Activations loaded for domain %s, from %s." % (domain, path_sv))
        f = torch.load(path_sv)
        a = f["activation"][:]
    else:
        a = get_activations(
            dataloader=dataloader,
            model=model,
            domain=domain,
            batch_size=batch_size,
            dims=dims,
            device=device,
            nb_max_img=nb_max_img,
        )

    if path_sv:
        torch.save({"activation": a}, path_sv)

    return a
