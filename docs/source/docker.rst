.. _docker:

############################
 Running JoliGEN with Docker
############################


***************
Docker images
***************

Jolibrain maintains JoliGEN server docker images, available from https://docker.jolibrain.com/#!/taglist/joligen_server

JoliGEN server images support both CPU and GPU training (inference is external):

.. code:: bash

   docker pull docker.jolibrain.com/joligen_server

To run joliGEN server with docker:

.. code:: bash

   nvidia-docker run jolibrain/myjoligen
			      
**************
Docker builds
**************

To build a docker for joliGEN server:

.. code:: bash

   docker build -t jolibrain/joligen_build -f docker/Dockerfile.build .
   docker build -t jolibrain/joligen_server -f docker/Dockerfile.server .
