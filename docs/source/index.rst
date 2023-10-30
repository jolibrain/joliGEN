####################################################################################
 JoliGEN: Generative AI Toolset with GANs and Diffusion for Real-World Applications
####################################################################################

.. image:: https://github.com/jolibrain/joliGEN/actions/workflows/github-actions-black-formatting.yml/badge.svg
   :target: https://github.com/jolibrain/joliGEN/actions/workflows/github-actions-black-formatting.yml

`JoliGEN <https://github.com/jolibrain/joliGEN/>`_ is an integrated framework for training custom generative AI image-to-image models

***************
 Main Features
***************

-  JoliGEN support both **GAN and Diffusion models** for unpaired and paired
   image to image translation tasks, including domain and style
   adaptation with conservation of semantics such as image and object
   classes, masks, ...

-  JoliGEN generative capabilities are targeted at real world
   applications such as **Controled Image Generation**, **Augmented Reality**, **Dataset Smart Augmentation**
   and object insertion, **Synthetic to Real** transforms.

-  JoliGEN allows for fast and stable training with astonishing results.
   A **server with REST API** is provided that allows for simplified
   deployment and usage.

-  JoliGEN has a large scope of options and parameters. To not get
   overwhelmed, start with :ref:`Quick Start <quickstart-gan-dataset>`. There are
   then links to more detailed documentation on models, dataset formats,
   and data augmentation.

***********
 Use cases
***********

-  **AR and metaverse**: replace any image element with super-realistic
   objects
-  **Smart data augmentation**: test / train sets augmentation
-  **Image manipulation**: seamlessly insert or remove objects/elements in
   images
-  **Image to image translation** while preserving semantic, e.g. existing
   source dataset annotations
-  Simulation to reality translation while preserving elements, metrics,
   ...
-  **Image generation to enrich datasets**, e.g. counter dataset imbalance, increase test sets, ...

This is achieved by combining conditioned generator architectures for
fine-grained control, bags of discriminators, configurable neural
networks and losses that ensure conservation of fundamental elements
between source and target images.

*****************
 Example results
*****************

Image translation while preserving the class
============================================

Mario to Sonic while preserving the action (running, jumping, ...)

.. image:: https://user-images.githubusercontent.com/3530657/196461791-9ff55a47-1e74-4ee7-ad3b-0a915dee6ae6.png

.. image:: https://user-images.githubusercontent.com/3530657/196461802-21d3015b-f5e8-467b-9096-78fcabd1f57b.png

Object insertion
================

Virtual Try-On with Diffusion

.. image:: https://github.com/jolibrain/joliGEN/assets/3530657/e3fc87a9-66be-4921-8abe-7dae479a2b10

Car insertion (BDD100K) with Diffusion

.. image:: https://user-images.githubusercontent.com/3530657/196428508-3eae3415-8e15-4505-9e97-41c0ba99350e.png

.. image:: https://user-images.githubusercontent.com/3530657/196428593-6ad8e229-368a-4714-a1cc-8aa8210beaad.png

Glasses insertion (FFHQ) with Diffusion

.. image:: https://github.com/jolibrain/joliGEN/assets/3530657/eba7920d-4430-4f46-b65c-6cf2267457b0
.. image:: https://github.com/jolibrain/joliGEN/assets/3530657/ef908a7f-375f-4d0a-afec-72d1ee7eaafe
	   
Object removal
==============

Glasses removal with GANs

.. image:: https://user-images.githubusercontent.com/3530657/200873590-6d1abe9a-7d86-458a-a9a5-97a1bcf4b816.png

.. image:: https://user-images.githubusercontent.com/3530657/200873601-e8c2d165-af58-4b39-a0bf-ecab510981c5.png
	   
Style transfer while preserving label boxes (e.g. cars, pedestrians, street signs, ...)
=======================================================================================

Day to night (BDD100K) with Transformers and GANs

.. image:: https://user-images.githubusercontent.com/3530657/196472056-b342c326-056f-4680-ad8d-4bf932b1404a.png

Clear to snow (BDD100K) by applying a generator multiple times to add
snow incrementally

.. image:: https://user-images.githubusercontent.com/3530657/196426503-bfaee698-b135-493f-81a6-644881cc1a5c.png

Clear to overcast (BDD100K)

.. image:: https://user-images.githubusercontent.com/3530657/196426571-f7e6189b-fa3a-4f5e-b6d5-fd580d14f29c.png

Clear to rainy (BDD100K)

.. image:: https://user-images.githubusercontent.com/3530657/196426461-e983c48f-ce19-4e83-a490-7a73b28c8181.png

.. image:: https://user-images.githubusercontent.com/3530657/196426623-deb7c00d-77e7-448e-827f-2423fd76b0ef.png

Authors
-------

**JoliGEN** is created and maintained by `Jolibrain
<https://www.jolibrain.com/>`_.

Code is making use of `pytorch-CycleGAN-and-pix2pix
<https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix>`_, `CUT
<https://github.com/taesungp/contrastive-unpaired-translation>`_,
`AttentionGAN <https://github.com/Ha0Tang/AttentionGAN>`_, `MoNCE
<https://github.com/fnzhan/MoNCE>`_ among others.

Elements from JoliGEN are supported by the French National AI
program `"Confiance.AI" <https://www.confiance.ai/en/>`_

Contact: contact@jolibrain.com

.. toctree::
   :maxdepth: 4
   :caption: Get Started
   :glob:
   :hidden:

   install
   choose
   quickstart_gan
   quickstart_ddpm

.. toctree::
   :maxdepth: 2
   :caption: Data
   :glob:
   :hidden:

   datasets
   dataloaders

.. toctree::
   :maxdepth: 2
   :caption: Options
   :glob:
   :hidden:

   options
   models

.. toctree::
   :maxdepth: 2
   :caption: Models
   :glob:
   :hidden:

   models

.. toctree::
   :maxdepth: 2
   :caption: Training
   :glob:
   :hidden:

   training
   losses
   
.. toctree::
   :maxdepth: 2
   :caption: Inference
   :glob:
   :hidden:

   export
   inference
   metrics
   
.. toctree::
   :maxdepth: 2
   :caption: Server & API
   :glob:
   :hidden:

   server
   API
   
.. toctree::
   :maxdepth: 2
   :caption: Docker
   :glob:
   :hidden:

   docker

.. toctree::
   :maxdepth: 2
   :caption: Tips
   :glob:
   :hidden:

   tips
   FAQ

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :glob:
   :hidden:

   tutorial_viton
   tutorial_styletransfer_bdd100k
