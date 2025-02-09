# joliGEN: Generative AI Toolset (Changelog)

## [4.0.0](https://github.com/jolibrain/joliGEN/compare/v3.0.0...v4.0.0) (2024-12-19)


### Features

* adding separate control of vertical and horizontal flips as augmentation ([a7a6109](https://github.com/jolibrain/joliGEN/commit/a7a61098b7e707ad0b80231c4d63278cae9dcde6))
* aligned crops for super-resolution ([8418470](https://github.com/jolibrain/joliGEN/commit/8418470d43cc79ba135e9206a8cbb9445319de44))
* allow tf32 on cudnn ([367cd91](https://github.com/jolibrain/joliGEN/commit/367cd91e8ccc0bb6e301bfc49e6087c8892f5ca0))
* better Canny for cond image with background ([c3c7de6](https://github.com/jolibrain/joliGEN/commit/c3c7de6151a029a13939fdbeb32712c2beb7c3f0))
* consistency models with supervised losses ([ed701ad](https://github.com/jolibrain/joliGEN/commit/ed701add9c1c8c5881c9347ae0d1db9d89aa5538))
* **data:** random bbox for inpainting ([764646d](https://github.com/jolibrain/joliGEN/commit/764646d40da714bc0b54bb88e3f8b389cc9d34c5))
* input and output multiple and different channels ([6bcd64c](https://github.com/jolibrain/joliGEN/commit/6bcd64c0d074320443fe1f793d55b2631de8c377))
* load models without stricness ([073d57c](https://github.com/jolibrain/joliGEN/commit/073d57c6061f20b026dd3b5ad3368938e0f186e3))
* max number of visualized images from train/test set ([24f0e81](https://github.com/jolibrain/joliGEN/commit/24f0e814e38d28cc16c87f290f95cf7874980071))
* **ml:** add option for vid inference ([c3f83b7](https://github.com/jolibrain/joliGEN/commit/c3f83b75f8465fb6f8dd3c37f959c6d24d128ecc))
* **ml:** add supervised loss with GANs with aligned datasets ([d7f5119](https://github.com/jolibrain/joliGEN/commit/d7f5119ef17ee385f67a2ab7999ca2c02b13f7f5))
* **ml:** added LPIPS supervised loss with GANs ([70e8ee4](https://github.com/jolibrain/joliGEN/commit/70e8ee47c92cab9ccd04a55a6c50e21a4b24004c))
* **ml:** adding example of CM+discriminator ([b6b8b64](https://github.com/jolibrain/joliGEN/commit/b6b8b6497df2eb6b8edd7b4c6db0988201ac70b5))
* **ml:** batched prompts for turbo ([023dd54](https://github.com/jolibrain/joliGEN/commit/023dd54c02985751841c3df0db6c37fa418392d2))
* **ml:** Canny can use a range of dropout probabilities ([7b4c860](https://github.com/jolibrain/joliGEN/commit/7b4c860de507223b33ddea05c5c5098720caa888))
* **ml:** canny dropout for vid ([06ce7d7](https://github.com/jolibrain/joliGEN/commit/06ce7d79a78b41c3da98a976382800155b0c2913))
* **ml:** CM with added discriminator ([10516e0](https://github.com/jolibrain/joliGEN/commit/10516e0290b5f2a407a6baa36650a3ff98ad9784))
* **ml:** consistency models for pix2pix ([cd92712](https://github.com/jolibrain/joliGEN/commit/cd927127055abee8cd8750faed83b30e9c647e56))
* **ml:** CUT turbo ([cdd508f](https://github.com/jolibrain/joliGEN/commit/cdd508fbfbbd76feeb4f6dd46269939f0b2e2c8c))
* **ml:** debug args ([a11172b](https://github.com/jolibrain/joliGEN/commit/a11172bd1ef43b5e0fe1f458ebd6bc4c73f20536))
* **ml:** debug crop ([51c9fd6](https://github.com/jolibrain/joliGEN/commit/51c9fd630df57fac7206205b2bde2788076822ad))
* **ml:** debug for canny inference ([930f3ce](https://github.com/jolibrain/joliGEN/commit/930f3ce90428b28e20705b405624fa4b177f88d1))
* **ml:** debug for canny threshold ([dca0bfa](https://github.com/jolibrain/joliGEN/commit/dca0bfabd6aea22bbc617a0b57a5430cf8fc7336))
* **ml:** debug for vid metrics ([7c57471](https://github.com/jolibrain/joliGEN/commit/7c574711ccdffb0fdc6fae946863f892c9e3437a))
* **ml:** debug inference_vid for canny ([17b9a29](https://github.com/jolibrain/joliGEN/commit/17b9a29fbea78c3d225fa25b437b7409a64f88cb))
* **ml:** debug vid for frame limit ([ff97c03](https://github.com/jolibrain/joliGEN/commit/ff97c031d0686f31c216b5a96b47fe418f6ba1e3))
* **ml:** debug vid metric ([ba43725](https://github.com/jolibrain/joliGEN/commit/ba43725cf9a34dfa01fc8b4c030cad9373d28ad4))
* **ml:** DISTS supervised loss for aligned data ([56273ef](https://github.com/jolibrain/joliGEN/commit/56273eff9ccad54d499d7f04557cc4fb5cba50d4))
* **ml:** FID,KID,MSID for multiple test sets and non 8 bit images ([74b0e65](https://github.com/jolibrain/joliGEN/commit/74b0e65519c093ad40f3e91127962a853354203c))
* **ml:** fix canny range option ([c102ee0](https://github.com/jolibrain/joliGEN/commit/c102ee0a14b07255c3dc5c7f670bc8e949b81938))
* **ml:** fix inference regeneration and crop canny ([f75196f](https://github.com/jolibrain/joliGEN/commit/f75196fd409c5190b38d50768eb3d9cfc11bfe8d))
* **ml:** HDiT for GANs ([58bedff](https://github.com/jolibrain/joliGEN/commit/58bedffff203bc2468df124545fe7830eb54e005))
* **ml:** HDiT generator ([9a95f1f](https://github.com/jolibrain/joliGEN/commit/9a95f1f40dd2f09a1d2ac1801bd89731e4ee5933))
* **ml:** jenkins test inference print ([b68ab53](https://github.com/jolibrain/joliGEN/commit/b68ab53e303b45c977ab0812b12f4d13fd3647e8))
* **ml:** L1 or MSE for diffusion multiscale loss ([06e3d6a](https://github.com/jolibrain/joliGEN/commit/06e3d6aa4fdbb01c0ab29bccd1e0d125dcc63e00))
* **ml:** metric fvd for video ([6d458a3](https://github.com/jolibrain/joliGEN/commit/6d458a3f7d911d88fe0a222a2b8f79b47904903d))
* **ml:** min-SNR loss weight for diffusion, 2303.09556 ([c802119](https://github.com/jolibrain/joliGEN/commit/c80211936002ed69bfadd17c151bad71c73e4c7c))
* **ml:** modif for horse2zebra prompt ([b66a954](https://github.com/jolibrain/joliGEN/commit/b66a954981b22eae33e949dbad078dad75e51048))
* **ml:** multiple test sets ([6db745c](https://github.com/jolibrain/joliGEN/commit/6db745c253a19bb587edd265f30192020fb9fe97))
* **ml:** option for max_sequence_lenght of video generation ([12cfc1b](https://github.com/jolibrain/joliGEN/commit/12cfc1b3c416306294e5020534332d7b2f05c414))
* **ml:** prompt for inference horze2zebra ([b8e9929](https://github.com/jolibrain/joliGEN/commit/b8e9929b20e44509cc6c8916c2be9b70f8d3c48d))
* **ml:** random canny inside batch ([70919cd](https://github.com/jolibrain/joliGEN/commit/70919cd3c4c57260ebc7e8b82d3a48b0d13cf5da))
* **ml:** rename dataloader for video generation ([98b1315](https://github.com/jolibrain/joliGEN/commit/98b13154e3fc1d4722f3a4dec3a0511c4c157d10))
* **ml:** The implementation of UNetVid for generating video with temporal consistency and inference ([43b7018](https://github.com/jolibrain/joliGEN/commit/43b70188f130de4009942f5cb5f237a7c2eb8af0))
* **ml:** unchange fill_img_with_canny with random drop canny ([a2ed3fc](https://github.com/jolibrain/joliGEN/commit/a2ed3fc32ac5f7e0f26c0701380ebd9c725a83f6))
* **ml:** UNetVid for generating video with bs > 1 ([00f11bc](https://github.com/jolibrain/joliGEN/commit/00f11bcf5331844e680cc3213814197d450d7591))
* **ml:** vid try autoregressive inference ([5b92031](https://github.com/jolibrain/joliGEN/commit/5b92031ce3e7650a1343a30c4cf5810547f2f103))
* multi-prompt local works ([b98746a](https://github.com/jolibrain/joliGEN/commit/b98746aaf5c6857620a6ec7070270cbda5d2a48d))
* multiprompt ([2bffc8b](https://github.com/jolibrain/joliGEN/commit/2bffc8bcc4520410536b0c170b4e541c37212c24))
* multistep lr scheduler ([01c3558](https://github.com/jolibrain/joliGEN/commit/01c3558a973eec593b73f882c81d3c86ddae1df7))
* train_finetune for finetuning gans/others and removing / adding losses and networks ([2f26503](https://github.com/jolibrain/joliGEN/commit/2f265036169458ac676452c193b3c7e460c82861))
* unet_vid motion module fine-grained configuration ([813e435](https://github.com/jolibrain/joliGEN/commit/813e4358303df4a18254bdb4125a271435d896dc))


### Bug Fixes

* aligne dataset, resize domain A only if necessary ([4127571](https://github.com/jolibrain/joliGEN/commit/41275710c91fef9cf19a844bbf84516fd1fbd961))
* allowing for no NCE with cut ([9d8ff9b](https://github.com/jolibrain/joliGEN/commit/9d8ff9b37cadb7fe4aacaa05408c85c75abf5c51))
* clamp bbox to image size during inference ([fc3874d](https://github.com/jolibrain/joliGEN/commit/fc3874db5994780a0a62c96aa8b1ea9cc03fddf2))
* cm at test time ([706356b](https://github.com/jolibrain/joliGEN/commit/706356b70cb273575eb4bdba80ffdbbe1798e5a0))
* cm with conditioning ([0fd2d14](https://github.com/jolibrain/joliGEN/commit/0fd2d14acc1011e3319f0bff3843ebcda55952e3))
* consistency model schedule upon resume ([88d03f9](https://github.com/jolibrain/joliGEN/commit/88d03f9466950f55ad589bff7597171e14829371))
* consistency models with input/output different channels ([db61821](https://github.com/jolibrain/joliGEN/commit/db618210a733e33aa0037f101019cd66850e145b))
* crash in inference script, errors in documentation ([f99dd34](https://github.com/jolibrain/joliGEN/commit/f99dd348fd7a3b2a4314012d0b2d4de08d774408))
* cut options at test time ([dcd2438](https://github.com/jolibrain/joliGEN/commit/dcd24383fe2e4da7a2c7b96977fe98ff93a01d0a))
* D input is G output size with gans ([194f42b](https://github.com/jolibrain/joliGEN/commit/194f42b32dc7aef632b6e01f45fb1c1ca76dfe46))
* diff across input/output channels in gans ([6845816](https://github.com/jolibrain/joliGEN/commit/68458168d44ff2bd94c7cd8ee7048739d8468ab4))
* diff real/fake not needed + cleanup ([5cbd1f0](https://github.com/jolibrain/joliGEN/commit/5cbd1f04890d250f752a39034bc38e42b82ff07b))
* diffusion inference for images > 8bits ([aefdc38](https://github.com/jolibrain/joliGEN/commit/aefdc384c8e973b83c7b9ac661d069face054208))
* diffusion with input and output of different channel size ([cd264de](https://github.com/jolibrain/joliGEN/commit/cd264de330c15785288aeb658de93bd94230d82a))
* disable hdit flop count ([8c449f8](https://github.com/jolibrain/joliGEN/commit/8c449f8baf8311db1555c484bd9bd54c42bbe596))
* fix pytest rootdir ([1fe0e80](https://github.com/jolibrain/joliGEN/commit/1fe0e805579abd9c09779b3dac68795780a510b3))
* further lowering the input test size of cut-turbo ([6914731](https://github.com/jolibrain/joliGEN/commit/6914731c3a700514cae20b3accfb539763217c43))
* gan inference script with prompts ([cef7681](https://github.com/jolibrain/joliGEN/commit/cef7681cc701e270fde5ab3a3104ef46f11b809c))
* gan metrics reference ([d5570b6](https://github.com/jolibrain/joliGEN/commit/d5570b686fd0fadc970a0fc0454179d8c3e14d66))
* GAN semantic visual output ([d3a5565](https://github.com/jolibrain/joliGEN/commit/d3a5565403bb6994741df6e0d968006f1f17fb74))
* GAN semantic visual output ([e7ee6bd](https://github.com/jolibrain/joliGEN/commit/e7ee6bd856480efbd1877153ed76bde218c9ac5c))
* gen_single_image.py for images with channels > 3 ([9ad4aaa](https://github.com/jolibrain/joliGEN/commit/9ad4aaae0f441d037ef022de97fa0f88cf05b3ab))
* hdit out_channel ([84473fc](https://github.com/jolibrain/joliGEN/commit/84473fcbc10e16eb124c99e4a3d8d863ab84f4bf))
* identity with cut turbo ([2538c00](https://github.com/jolibrain/joliGEN/commit/2538c008e9a733831bf6eb344853ded6a2fb5e24))
* inference with images > 8bit and GANs ([34e6c96](https://github.com/jolibrain/joliGEN/commit/34e6c96cc30352b330dd70d55b5c99287cc92810))
* input size of cut-turbo test ([2c024c2](https://github.com/jolibrain/joliGEN/commit/2c024c2b92c2c8d96d965d9f15efbc5ab6efb7e2))
* interpolation size selection for projected discriminators ([ef045d0](https://github.com/jolibrain/joliGEN/commit/ef045d0e7450a21768eea09f832ae6e3f3ef7268))
* load_image replacement ([5af5803](https://github.com/jolibrain/joliGEN/commit/5af5803cbf921e99dcf3598e54fdf75a0259b735))
* loading of ema models ([995c5eb](https://github.com/jolibrain/joliGEN/commit/995c5eba5e71234bdd1980dce6a6c8c1e6cd3274))
* lora config saving with multiple gpus ([c98617d](https://github.com/jolibrain/joliGEN/commit/c98617d1e534018d095c2c0ee96a3f8f9980fa8e))
* lower img2img turbo test memory footprint ([54a6ab4](https://github.com/jolibrain/joliGEN/commit/54a6ab4a4bcc7ab7b8925233b64bea8c888f6fb5))
* missing SSIM metric option ([8530851](https://github.com/jolibrain/joliGEN/commit/8530851eca49a63a23638fbe64a8257c61f66eda))
* **ml:** multiscale diffusion loss for any input resolution ([5c9f997](https://github.com/jolibrain/joliGEN/commit/5c9f9975de2f63ad89b84c4622176c4ba537f61e))
* multi-gpu ddp collective mismatch upon resume ([471fbbc](https://github.com/jolibrain/joliGEN/commit/471fbbca598661c2b4ebcc06ba623c384e1aa6a3))
* multi-gpu with frozen base network ([1a07342](https://github.com/jolibrain/joliGEN/commit/1a0734253fd02fb6be98e5066994ae44ede643f4))
* multiple test sets with test.py + SSIM ([06762fb](https://github.com/jolibrain/joliGEN/commit/06762fb051ddfbc68b496180f37509b5e0410fc4))
* option default cut_nce_idt ([4c5ec6d](https://github.com/jolibrain/joliGEN/commit/4c5ec6d8c195169381a369f90cf2fbbd446d1a23))
* palette options at test time ([75f7b04](https://github.com/jolibrain/joliGEN/commit/75f7b04572a934a66672646bfe6d0e97eb869b2e))
* parser uses model_type for model level options ([76095b5](https://github.com/jolibrain/joliGEN/commit/76095b5a8ed6fb96dfdcc2166b3c892d09d9547c))
* paths are only required for video generation ([eb39ec5](https://github.com/jolibrain/joliGEN/commit/eb39ec521b5d01ee40add84317ec24cea49a82ab))
* paths loading prompts file ([35d2ef3](https://github.com/jolibrain/joliGEN/commit/35d2ef3e0f803898110f905fe0f2ed68578d64dd))
* perceptual loss for cm when input and output channels differ ([ca81789](https://github.com/jolibrain/joliGEN/commit/ca81789385ccaa4c284a11aa48051bdd84ce226d))
* potential bug in gen_single_diffusion model path ([0cf63fe](https://github.com/jolibrain/joliGEN/commit/0cf63fe27db36e51bf62c0a607dd37e6b5feda8a))
* projected discriminator allows grayscale input ([44fb458](https://github.com/jolibrain/joliGEN/commit/44fb4586ff2b2701505d5e83bba4489f1e47be1f))
* prompt unaligned loading ([e25d4b1](https://github.com/jolibrain/joliGEN/commit/e25d4b1d707a490f7e41daa36da8b2f7eb5f8167))
* rename sketch options in examples ([6930d00](https://github.com/jolibrain/joliGEN/commit/6930d007052b51aa283a69cf5e552e31cfc14b54))
* RGB order for diffusion inpainting ([eff8a57](https://github.com/jolibrain/joliGEN/commit/eff8a577f482ccb5bb7efc2c74c57c48db85ec8b))
* rgbn cut lpips supervision ([17cfbb2](https://github.com/jolibrain/joliGEN/commit/17cfbb27df3a3c1c52a9a318c68b05c5ec1d4e35))
* sam for single channel inputs ([397f837](https://github.com/jolibrain/joliGEN/commit/397f837decabec3f05a2c04a9e3fe30f3a19e49d))
* segformer generator for single channel inputs ([1eb6695](https://github.com/jolibrain/joliGEN/commit/1eb66955830e6e84a22270d5e5d16e0435d2042d))
* show full test set output with GANs ([31efdcd](https://github.com/jolibrain/joliGEN/commit/31efdcdd1a3d4799a78a1c919e7c91ee10b9f93d))
* single dataset ([a6266d8](https://github.com/jolibrain/joliGEN/commit/a6266d85b1a1305894c18a829dd745b9bc00a539))
* supervised loss for aligned GANs, with unit tests ([e21ddd3](https://github.com/jolibrain/joliGEN/commit/e21ddd3ddb97ed1221f22b10b7574be6d42ade7a))
* supervised perceptual metrics all with piq and configurable + lambda weight ([d77c3c5](https://github.com/jolibrain/joliGEN/commit/d77c3c54e078a78f37ab2770d4087481facfb840))
* test image output tensor visuals ([19596b2](https://github.com/jolibrain/joliGEN/commit/19596b2997922218a2aa8cbf75685101e5e93f1a))
* tifffile import ([a09b5ed](https://github.com/jolibrain/joliGEN/commit/a09b5edbf4a5ecf69565c62778dbf48e172de585))
* total_iters wrong variable ([066dc1b](https://github.com/jolibrain/joliGEN/commit/066dc1bc3d423a2f854b6fd0c3d6a91c6e4d3f94))
* train batch visuals ([24adb61](https://github.com/jolibrain/joliGEN/commit/24adb614e65d009659304b412ab92f147bee8741))
* typo in semantic threshold test variable ([5082c36](https://github.com/jolibrain/joliGEN/commit/5082c3622bd3541a0da7383168d0579af9f22e04))
* unet mha output for GANs ([075b6c6](https://github.com/jolibrain/joliGEN/commit/075b6c60094ca8d4029fd5705e709801c7fdf2d1))
* visualizer for mixed single and multi channel images ([34aefe0](https://github.com/jolibrain/joliGEN/commit/34aefe0719cc811a1529b8db7cb6d32d9f09f129))
