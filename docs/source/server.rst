.. _server:

############################
 JoliGEN Server & Client
############################

JoliGEN has a built-in server with REST API

**********************
Running JoliGEN server
**********************

Ensure everything is installed

.. code:: bash

   pip install fastapi uvicorn

Then run server:

.. code:: bash

   server/run.sh --host localhost --port 8000

******************************
 Client: server-based training
******************************

.. code:: bash

   python client.py --host jg_server_host --port jg_server_port  [joligen commandline options eg --dataroot /path/to/data --model_type cut --name mymodel]

NB: the `--name name` passed to joligen server commandline options becomes the name
of the training job.

*****************************************
Client: listing training jobs in progress
*****************************************

.. code:: bash

   python client.py --method training_status --host jg_server_host --port jg_server_port

*******************************
Client: stopping a training job
*******************************

.. code::

   python client.py --method training_status --host jg_server_host --port jg_server_port --name training_name
