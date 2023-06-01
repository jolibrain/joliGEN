#######################
 JoliGEN client python
#######################

Here are calls examples that you can use to make API calls to a JoliGEN
server. Please note that you have to run a server first.

****************
 JoliGEN server
****************

Ensure everything is installed

.. code:: bash

   pip install fastapi uvicorn

Then run server:

.. code:: bash

   server/run.sh --host localhost --port 8000

**************
 Docker build
**************

To build a docker for joliGEN server:

.. code:: bash

   docker build -t jolibrain/joligen_build -f docker/Dockerfile.build .
   docker build -t jolibrain/joligen_server -f docker/Dockerfile.server .

To run the joliGEN docker:

.. code:: bash

   nvidia-docker run jolibrain/myjoligen

************
 Unit tests
************

To launch tests before new commits:

.. code:: bash

   bash scripts/run_tests.sh /path/to/dir

*******************
 Launch a training
*******************

.. code::

   python client.py --host jg_server_host --port jg_server_port \
   [joligen commandline options eg --dataroot /path/to/data --model_type cut]

NB: the name given in joligen commandline options will also be the name
of the training process.

****************************
 List trainings in progress
****************************

.. code::

   python client.py --method training_status --host jg_server_host --port jg_server_port

*****************
 Stop a training
*****************

.. code::

   python client.py --method training_status --host jg_server_host --port jg_server_port --name training_name
