.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_mlfm_adapgrad_tutorials_plot_mlfmaggibbs.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_mlfm_adapgrad_tutorials_plot_mlfmaggibbs.py:


Gibbs Sampling
==============

This example presents an illustration of the MLFM to learn the model

.. math::

   \dot{\mathbf{x}}(t)    

We do the usual imports and generate some simulated data




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_004.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_005.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_006.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_007.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_008.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_009.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/mlfm_adapgrad_tutorials/images/sphx_glr_plot_mlfmaggibbs_010.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[ 0.09  -0.1    0.045]
     [-0.797  0.86  -0.396]]




|


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pydygp.probabilitydistributions import (Normal,
                                                 GeneralisedInverseGaussian,
                                                 InverseGamma)
    from sklearn.gaussian_process.kernels import RBF
    from pydygp.liealgebras import so
    from pydygp.linlatentforcemodels import GibbsMLFMAdapGrad

    np.random.seed(15)


    gmlfm = GibbsMLFMAdapGrad(so(3), R=1, lf_kernels=(RBF(), ))

    beta = np.row_stack(([0.]*3,
                         np.random.normal(size=3)))

    x0 = np.eye(3)

    # Time points to solve the model at
    tt = np.linspace(0., 6., 9)

    # Data and true forces
    Data, lf = gmlfm.sim(x0, tt, beta=beta, size=3)

    # vectorise and stack the data
    Y = np.column_stack((y.T.ravel() for y in Data))

    logpsi_prior = GeneralisedInverseGaussian(a=5, b=5, p=-1).logtransform()
    loggamma_prior = InverseGamma(a=0.001, b=0.001).logtransform() * gmlfm.dim.K
    beta_prior = Normal(scale=1.) * beta.size

    fitopts = {'logpsi_is_fixed': True, 'logpsi_prior': logpsi_prior,
               'loggamma_is_fixed': True,# 'loggamma_prior': loggamma_prior,
               'beta_is_fixed': False, 'beta_prior': beta_prior,
               'beta0': beta,
               }
    gibbsRV = gmlfm.gibbsfit(tt, Y,
                             sample=('g', 'x'),
                             size=1000,
                             **fitopts)
    mapres = gmlfm.fit(tt, Y, **fitopts)

    A = [sum(brd*Ld for brd, Ld in zip(br, gmlfm.basis_mats))
         for br in beta]

    aij = []

    _aa = 0.

    for g in gibbsRV['g']:# in zip(gibbsRV['g']):#, gibbsRV['beta']):
        _beta = mapres.beta #b.reshape((2, 3))
        _A = [sum(brd*Ld for brd, Ld in zip(br, gmlfm.basis_mats))
              for br in _beta]
        aij.append(_A[0][0, 2] + _A[1][0, 2]*g)
        _aa += gmlfm._component_functions(g, _beta)
    aij = np.array(aij)

    print(_beta)

    #print(np.mean(gibbsRV['beta'], axis=0))
      
    _aa /= gibbsRV['g'].shape[0]

    fig, ax = plt.subplots()
    ttd = np.linspace(0., tt[-1], 100)
    ax.plot(ttd, A[0][0, 2] + A[1][0, 2]*lf[0](ttd), 'b-')
    ax.plot(tt, aij.T, 'k+')
    ax.plot(tt, np.mean(aij, axis=0), '+')
    ax.plot(tt, _aa[0, 1, :], '-.')

    fig2, ax2 = plt.subplots()
    ax2.plot(ttd, lf[0](ttd), 'k-')
    ax2.plot(tt, np.mean(gibbsRV['g'], axis=0), '+')
    ax2.plot(tt, mapres.g.T, 'o')
    ax2.plot(tt, gibbsRV['g'].T, 'k+')

    fig3, ax3 = plt.subplots()
    #ax3.hist(gibbsRV['beta'][:, 2], density=True)

    aaTrue = gmlfm._component_functions(lf[0](ttd), beta, N=ttd.size)
    aaMap = gmlfm._component_functions(mapres.g.ravel(), mapres.beta)

    fig4, ax4 = plt.subplots()
    for i, j in zip([0, 0, 1], [1, 2, 2]):
        ax4.plot(tt, _aa[i, j, :], '+-',
                 label=r'$a{}{}$'.format(i, j))
        ax4.plot(ttd, aaTrue[i, j, :], alpha=0.4)
        ax4.plot(tt, aaMap[i, j, :], 's')
    ax4.legend()

    xrv = gibbsRV['x']

    for m in range(3):
        fig, ax = plt.subplots()

        xm = xrv[..., m].reshape(xrv.shape[0],
                                 gmlfm.dim.K,
                                 gmlfm.dim.N)


        ax.plot(tt, xm[:, 2, :].T, 'k+')
        ax.plot(tt, Data[m], 's')


    from scipy.interpolate import interp1d
    u = interp1d(tt, mapres.g.ravel(), kind='cubic', fill_value='extrapolate')

    for m, x0m in enumerate(x0):
    
        sol, _ = gmlfm.sim(x0m, ttd, beta=mapres.beta, latent_forces=(u, ))

        fig, ax = plt.subplots()
        ax.plot(ttd, sol, 'k-')
        ax.plot(tt, Data[m], 's')
    
    plt.show()

**Total running time of the script:** ( 0 minutes  33.574 seconds)


.. _sphx_glr_download_tutorials_mlfm_adapgrad_tutorials_plot_mlfmaggibbs.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mlfmaggibbs.py <plot_mlfmaggibbs.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mlfmaggibbs.ipynb <plot_mlfmaggibbs.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
