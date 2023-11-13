# joliGEN: Generative AI Toolset (Changelog)

## [2.0.0](https://github.com/jolibrain/joliGEN/compare/v1.0.0...v2.0.0) (2023-11-13)


### Features

* **ml:** dinov2 discriminator with registers ([7fcf790](https://github.com/jolibrain/joliGEN/commit/7fcf790661e64d458238c2ac68f440098caa0bae))
* **ml:** DinoV2 feature-based projected discriminator ([c67ffa8](https://github.com/jolibrain/joliGEN/commit/c67ffa88cbfd7638b82c4e5b5cce6ba66a88d872))
* **ml:** SigLIP based projected discriminators ([5e10a86](https://github.com/jolibrain/joliGEN/commit/5e10a86b124ee1f9845216ef5372910671f54d86))
* optimization eps value control ([0556987](https://github.com/jolibrain/joliGEN/commit/0556987cb079a4325b30cdc9dbe55a332f119e01))
* pix2pix task for palette ([7e47139](https://github.com/jolibrain/joliGEN/commit/7e47139d1f0a5eb47849e6b36f50bf5d8df88b9d))
* **scripts:** adding a video generation script for gans ([85d1922](https://github.com/jolibrain/joliGEN/commit/85d192273940d253ec2e55fda4527e3c85dcf8ad))


### Bug Fixes

* amp for discriminators ([811ba3d](https://github.com/jolibrain/joliGEN/commit/811ba3dee80782a1073ee05354e78ebb27c1769f))
* APA augmentation on multiple discriminators ([becb3eb](https://github.com/jolibrain/joliGEN/commit/becb3eb1b3f2d4a5c1a71332e26b082563480066))
* docker release script ([f1c56de](https://github.com/jolibrain/joliGEN/commit/f1c56de0e2f8468f0ae49ae75714271e2a28a7fa))
* end of training metrics computation ([e1f213c](https://github.com/jolibrain/joliGEN/commit/e1f213cb0bb49148b517813690400ea3cc2285c3))
* init_metrics directory and metrics on CPU ([0b77943](https://github.com/jolibrain/joliGEN/commit/0b7794395271a8af77fedb0a97dbba98207829e8))
* load size for rectangular images, resize ref image for inference ([965e1bf](https://github.com/jolibrain/joliGEN/commit/965e1bfe3723c03bd589cd5371f9237a847500d0))
* **ml:** inference for diffusion with reference image ([df8c504](https://github.com/jolibrain/joliGEN/commit/df8c50454a03247999553720e19d2ddb8ee84635))
* save whole metrics_dict so we can reload it ([5d57c4c](https://github.com/jolibrain/joliGEN/commit/5d57c4cb83883678b392648b7e9b435fda7af429))
