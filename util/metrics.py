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

    pred_arr = np.empty((len(dataloader), dims))

    start_idx = 0

    if nb_max_img < len(dataloader):
        if isinstance(dataloader, list):
            dataloader = dataloader[:nb_max_img]
        else:
            index = random.sample(np.range(nb_max_img))
            subset = torch.utils.data.Subset(dataloader, index)
            dataloader = subset

    # print("batchsize", batch_size)

    for batch in tqdm(dataloader, total=len(dataloader) // batch_size):
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

        pred = pred.cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    dataloader,
    model,
    domain,
    batch_size,
    dims,
    device,
    nb_max_img,
):
    """Calculation of the statistics used by the FID.
    Params:
    -- dataloader  : JG-like dataloader (it can be a list of tensors too)
    -- model       : Instance of inception model
    -- domain      : image domain
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """

    act = get_activations(
        dataloader=dataloader,
        model=model,
        domain=domain,
        batch_size=batch_size,
        dims=dims,
        device=device,
        nb_max_img=nb_max_img,
    )

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


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
        print("Mu and sigma loaded for domain {domain}, from %s." % path_sv)
        f = np.load(path_sv)
        m, s, a = f["mu"][:], f["sigma"][:], f["activation"][:]
        f.close()
    else:
        m, s, a = calculate_activation_statistics(
            dataloader=dataloader,
            model=model,
            domain=domain,
            batch_size=batch_size,
            dims=dims,
            device=device,
            nb_max_img=nb_max_img,
        )

    if path_sv:
        np.savez(
            path_sv,
            mu=m,
            sigma=s,
            activation=a,
        )

    return m, s, a
