############
 JoliGEN UI
############

Web interface to control JoliGEN process.

**********
How to run
**********

::

  git clone git@git.jolibrain.com:jolibrain/joligen_ui.git
  cd joligen_ui

**********
Production
**********

Production env for Joligen UI will run a joligen server with a web ui on top of it.

1. Setup joligen data folder at ``/path/to/joligen/``
2. open another terminal and run the following commands:

::

  export JG_USER=$(id -u)
  export JG_PLATFORM_DATA=/path/to/joligen/
  docker-compose up

3. open `http://127.0.0.1:1912 <http://127.0.0.1:1912>`_

Other services will be available when running ``joligen_ui`` in production env:

- `Filebrowser <https://filebrowser.org>`_: `<http://127.0.0.1:1912/logs/>`_
- `Dozzle <https://dozzle.dev/>`_: `<http://127.0.0.1:1912/filebrowser/>`_
- `JoliGEN Docs <https://joligen.com/>`_: `<http://127.0.0.1:1912/docs/>`_

***********
Development
***********

Development env will allow you to modify ``joligen_ui`` code and validate these modifications directly in your web browser.

1. open a terminal and run ``yarn install && yarn start`` in project root
2. open another terminal and run the following commands:

::

  export JG_USER=$(id -u)
  export JG_PLATFORM_DATA=/opt/joligen/
  export JG_HOST=server_host
  export JG_PORT=18110
  docker-compose -f ./docker-compose-dev.yml up --no-deps --build joligen_ui

3. open `<http://server_host:18100>`_

************
Visdom Proxy
************

Visdom server should be running on host on port 8097 - `<http://server_host:8097>`_ - in order to display visdom iframe when clicking on `visdom` menu item in header.

If 8097 is not the right port number, it should be changed in ``./deploy/nginx-dev.conf`` in ``visdom_proxy`` location.
