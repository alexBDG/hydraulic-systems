|Python311|

.. |Python311| image:: https://img.shields.io/badge/python-3.11.9-blue.svg


============================
Monitoring Hydraulic Systems
============================


Installation
============

Install the requirements in a custom Python 3.11.9 environement.

.. code-block:: console

   usr@desktop:~$ python -m venv my_env
   usr@desktop:~$ source my_env/bin/activate
   (my_env) usr@desktop:~$ pip install -r requirements.txt

Then you can create a model running the `modeling file <modeling.ipynb>`_.


Testing
=======

To test the package:

.. code-block:: console

   (my_env) usr@desktop:~$ cd tests
   (my_env) usr@desktop:~$ python -W ignore test.py
