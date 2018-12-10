:orphan:


.. _tutorials-index:

Multiplicative Latent Force Models
==================================
This module provides methods to carry out simulation and fitting of
latent force models, which are broadly taken to be time dependent
linear ODEs driven by a set of smooth Gaussian processes which are
allowed to interact multiplicatively with the state
variable, and so the name Multiplicative Latent Force Models (MLFM)
to differ them from the case with an additive forcing term
which are discussed in :ref:`lfm-tutorials-index`

.. _tutorials-mlfm:

Multiplicative Latent Force Model
---------------------------------

Multiplicative latent force models are time dependent linear ODEs of the form

.. math::

   \dot{X}(t) = A(t)X(t), \qquad A(t) = A_0 + \sum_{r=1}^R g_r(t) A_r,

where :math:`\{ g_r(t) \}_{r=1}^R` are a set of independent smooth scalar Gaussian processes, and :math:`\{A_r \}_{r=0}^{R}` are a set of square :math:`K\times K` matrices. Furthermore it may also be the case that for each of the structure matrices :math:`A_r` we have

.. math::

   A_r = \sum_{d} \beta_{rd} L_d,

for some common set of shared basis matrices :math:`\{ L_d \}_{d=1}^{D}` -- typically these will be chosen to form a basis of some Lie algebra.

The following tutorials demonstrate the process of constructing these models as well as demonstrating the possible structure preserving properties of this model as well as how to carry out inference.

MLFM AdapGrad
-------------

Joint Density
~~~~~~~~~~~~~

For the MLFM AdapGrad model the likelihood of a set of state variables may be written

.. math::

   p(\mathbf{X} \mid \mathbf{g} ) \propto \prod_{k=1}^{K} \exp\left\{-\frac{1}{2}(f_k - m_{\dot{x}_k|x_k})^{\top}(C_{\dot{x}_k|\dot{x}} + \gamma I)^{-1}(f_k - m_{\dot{x}_k|x_k})\right\}

.. _tutorials-index-mlfmag-par:

Parameters
~~~~~~~~~~
The model parameters we are most interested in include the

+-----------------------------+-----------------------------------------+
| :math:`\mathbf{g}`          | The (vectorised) latent force variable  |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\psi}`   | Hyperparameters of the latent force GPs |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\phi}`   | Hyperparameters of the latent state GPs |
+-----------------------------+-----------------------------------------+
| :math:`\boldsymbol{\gamma}` | ODE model `temperature`       parameter |
+-----------------------------+-----------------------------------------+



.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This note decribes how to carry out the porcess of carrying out MAP parameter estimation for th...">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_mlfmagfit_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_mlfmagfit.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_mlfmagfit

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Introduction to the LFM Mix Model">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_mlfmmixsa_intro_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_mlfmmixsa_intro.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_mlfmmixsa_intro

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This note provides more detail into the process of carrying out model fitting using the MLFM mo...">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_mlfm_wpriors_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_mlfm_wpriors.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_mlfm_wpriors

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fitting of the MLFM">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_mlfmmixsa_fit_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_mlfmmixsa_fit.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_mlfmmixsa_fit

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fitting the LFM Using the MLFM">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_lfm_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_lfm.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_lfm

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Fitting of the MLFM">

.. only:: html

    .. figure:: /tutorials/llfm_tutorials/images/thumb/sphx_glr_plot_mlfmsamix_thumb.png

        :ref:`sphx_glr_tutorials_llfm_tutorials_plot_mlfmsamix.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorials/llfm_tutorials/plot_mlfmsamix
.. raw:: html

    <div style='clear:both'></div>



.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-gallery


  .. container:: sphx-glr-download

    :download:`Download all examples in Python source code: llfm_tutorials_python.zip <//Users/danieltait/Desktop/pydygp/doc/source/tutorials/llfm_tutorials/llfm_tutorials_python.zip>`



  .. container:: sphx-glr-download

    :download:`Download all examples in Jupyter notebooks: llfm_tutorials_jupyter.zip <//Users/danieltait/Desktop/pydygp/doc/source/tutorials/llfm_tutorials/llfm_tutorials_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
