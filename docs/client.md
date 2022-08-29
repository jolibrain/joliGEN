# JoliGAN client python

Here are calls examples that you can use to make API calls to a JoliGAN server. Please note that you have to run a server first.

#### Launch a training

```
python client.py --host jg_server_host --port jg_server_port [joligan commandline options eg --dataroot /path/to/data --model_type cut] 

```

NB: the name given in joligan commandline options will also be the name of the training process.

#### List trainings in progress

```
python client.py --method training_status --host jg_server_host --port jg_server_port 

```

#### Stop a training

```
python client.py --method training_status --host jg_server_host --port jg_server_port --name training_name

```