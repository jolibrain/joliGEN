.. _docker:

############################
 Running JoliGEN with Docker
############################

Jolibrain publishes the GPU-capable JoliGEN server image at
https://docker.jolibrain.com/#!/taglist/joligen_server. The server exposes training
and GAN/diffusion inference APIs, with API documentation at ``/docs``.

Install Docker and the NVIDIA container runtime, then run a published version
(replace ``vX.Y.Z`` with the desired release tag):

.. code:: bash

   docker pull docker.jolibrain.com/joligen_server:vX.Y.Z
   docker run --rm --gpus all -p 8000:8000 docker.jolibrain.com/joligen_server:vX.Y.Z

``latest`` tracks stable releases. For reproducible deployments, use the
version tag or image digest. Mount your datasets and output directories as
needed for training. The server does not enable development auto-reload.
Optional TensorRT deployment tools require a separate TensorRT environment.

*****************
Validated builds
*****************

Build from a committed checkout and run the container smoke tests plus the full
Jenkins test suite without publishing:

.. code:: bash

   DOCKER_GPU_RUN_ARGS='--gpus device=0' ci/build-docker-images.sh \
     --tag v5.0.0 --no-push --report-dir /tmp/joligen-validation

This requires a GPU, dataset/model download access, and sufficient disk space.
Only committed files enter the build context. The release image records its
version and source commit without including Git metadata.

The development and release images share dependency installation and use CUDA
12.6.2 with the repository's Torch and torchvision requirements. FlashAttention
is optional; the default image uses the existing PyTorch fallback.

See ``ci/README.md`` in the repository for the staged release process,
unpublished rehearsals, reports, publication, and recovery instructions.
