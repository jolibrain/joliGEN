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

.. _client_stop:

*******************************
Client: stopping a training job
*******************************

.. code:: bash

   python client.py --method stop_training --host jg_server_host --port jg_server_port --name training_name

.. _client_stop:

**********************
Curl: inference on GAN
**********************

Using the following `curl` command, it will create `$PATH_TO_FILES/output.jpg` file and return it inside `curl_response["base64"]` in a base64 formatted string.

Base64 string will be saved into `output.jpg` file.

.. code:: bash

  PATH_TO_MODEL=/home/joligen/models/horse2zebra/
  PATH_TO_FILES=/home/joligen/files/

  JOLIGEN_SERVER=localhost:18100

  BASE64_IMG = curl 'http://$JOLIGEN_SERVER/predict' -X POST \
    --data-raw  \
    '{
       "predict_options": {
         "model_in_file": "$PATH_TO_MODEL/latest_net_G_A.pth",
         "img_in": "$PATH_TO_FILES/source.jpg",
         "img_out": "$PATH_TO_FILES/output.jpg"
       },
       "server": {
         "sync": true,
         "base64": true
       }
    }' | jq .base64[0]

  cat $BASE64_IMG | base64 -d > output.jpg

If `payload["server"]["base64"]` is not enabled, file will be created on disk but it won't be returned inside `curl_response`.

If `payload["server"]["sync"]` is not enabled, the inference process will run in an asynchronous mode, `curl_response` will only return a message status stating that the process has started.

In `async` mode, process status can be followed using websocket:

.. code:: bash

  JOLIGEN_SERVER=localhost:18100

  PREDICT_NAME = curl 'http://$JOLIGEN_SERVER/predict' -X POST \
    --data-raw  \
    '{
       "predict_options": {
         "model_in_file": "$PATH_TO_MODEL/latest_net_G_A.pth",
         "img_in": "$PATH_TO_FILES/source.jpg",
         "img_out": "$PATH_TO_FILES/output.jpg"
       }
    }' | jq .name

  WEBSOCKET_URL='http://$JOLIGEN_SERVER/ws/predict/$PREDICT_NAME'

  curl -N -i \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket"
    $WEBSOCKET_URL | jq .

Websocket message will be returned by api server. Websocket connection will be closed when the inference is finished or if an error has been encountered

****************************
Curl: inference on Diffusion
****************************

Using the following `curl` command, it will create `$PATH_TO_FILES/output.jpg` file and return it inside `curl_response["base64"]` in a base64 formatted string.

Base64 string will be saved into `output.jpg` file.

.. code:: bash

  PATH_TO_MODEL=/home/joligen/models/horse2zebra/
  PATH_TO_FILES=/home/joligen/files/

  JOLIGEN_SERVER=localhost:18100

  BASE64_IMG = curl 'http://$JOLIGEN_SERVER/predict' -X POST \
    --data-raw  \
    '{
       "predict_options": {
         "model_in_file": "$PATH_TO_MODEL/latest_net_G_A.pth",
         "img_in": "$PATH_TO_FILES/source.jpg",
         "dir_out": "$PATH_TO_FILES"
       },
       "server": {
         "sync": true,
         "base64": true
       }
    }' | jq .base64[0]

  cat $BASE64_IMG | base64 -d > output.jpg

If `payload["server"]["base64"]` is not enabled, file will be created on disk but it won't be returned inside `curl_response`.

If `payload["server"]["sync"]` is not enabled, the inference process will run in an asynchronous mode, `curl_response` will only return a message status stating that the process has started.

In `async` mode, process status can be followed using websocket:

.. code:: bash

  JOLIGEN_SERVER=localhost:18100

  PREDICT_NAME = curl 'http://$JOLIGEN_SERVER/predict' -X POST \
    --data-raw  \
    '{
       "predict_options": {
         "model_in_file": "$PATH_TO_MODEL/latest_net_G_A.pth",
         "img_in": "$PATH_TO_FILES/source.jpg",
         "dir_out": "$PATH_TO_FILES"
       }
    }' | jq .name

  WEBSOCKET_URL='http://$JOLIGEN_SERVER/ws/predict/$PREDICT_NAME'

  curl -N -i \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket"
    $WEBSOCKET_URL | jq .

Websocket message will be returned by api server. Websocket connection will be closed when the inference is finished or if an error has been encountered.
