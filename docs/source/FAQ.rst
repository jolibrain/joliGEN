.. _faq:

############################
 Frequently Asked Questions
############################

- **What are the training times for the examples in JoliGEN training ?**

  It is reasonable to train for 10 to 15 days on 2 to 4 GPUs in 256x256 or 360x360. In 64x64 or 128x128, a couple days may suffice, always a good starting point.

  In general:

  - With GANs, convergence can be visually assessed within 1 or 2 days, then fine-grained details start to appear
  - With diffusion for object insertion, training is smoother due to the straight supervision, and good results are obtained with a couple of days, then 300 to 400 epochs are reasonable
	  
