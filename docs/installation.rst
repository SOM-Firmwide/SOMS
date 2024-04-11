.. _installation:

********************************************************************************
Installation
********************************************************************************

Create a conda environment
==========================

The basic prerequisite to install ``soms`` is `Anaconda <https://www.anaconda.com/products/individual>`_. Install Anaconda on you machine and go to the next step.

After you conclude the Anaconda Installation, open the ``Anaconda Prompt`` that comes with it. We are going to create an environment using the prompt and install ``soms`` in this environment. To learn more about Conda Environments check out this `link <https://edcarp.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html>`_.

We will create a new environment named ``soms`` with Python 3. We will install ``soms`` package and the open-source dependency `COMPAS <https://compas.dev>`_ and Git.

Open the Anaconda Prompt/terminal and type the following to create a new environmenment and install the dependencies:


.. code-block:: bash

    conda create -n soms -c conda-forge python=3.10 COMPAS git

If you are in OSX use:

.. code-block:: bash

    conda create -n soms -c conda-forge python=3.10 python.app COMPAS git

The base environment is active by default. Activate the ``soms`` environment with:

.. code-block:: bash

    conda activate soms

Installing soms
=====================

The most direct way to work with ``SOMS`` is by cloning the repository to your local drive. This is the equivalend of downloading the source files and unzipping it in a predetermined location in your machine. 

Create a folder to store the code. We suggest something like ``C:/Code/`` if you are in Windows. Navigate in the terminal to the folder. If you are new to using the teminal, check out this guide to orient yourself  `here <https://gomakethings.com/navigating-the-file-system-with-terminal/>`_. When you are in the folder type:

.. code-block:: bash

    git clone https://github.com/SOM-Firmwide/SOMS.git
    cd soms
    pip install -r requirements.txt -e .

This install ``soms``, the base required COMPAS packages and Git.

.. Standalone viewer
.. =================

.. The Standalone viewer `COMPAS View 2 <https://github.com/compas-dev/compas_view2.git>`_ is used currently to display 3D solutions directly from the terminal. The installation can be done through conda:

.. .. code-block:: bash

..     conda install -c conda-forge compas_view2

.. To finalise the installation you need to install a few additonal :ref:`Solvers <solvers>` to your environmenment following the additional guide.

.. Currently, a work-in-progress UI is being develeoped for :ref:`Rhino  <rhino>` 6+ and an installation guide is provided.
