.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_llfm_tutorials_plot_mlfmmixsa_fit.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_llfm_tutorials_plot_mlfmmixsa_fit.py:


Fitting of the MLFM-MixSA Model
===============================




.. rst-class:: sphx-glr-horizontal


    *

      .. image:: /tutorials/llfm_tutorials/images/sphx_glr_plot_mlfmmixsa_fit_001.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/llfm_tutorials/images/sphx_glr_plot_mlfmmixsa_fit_002.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/llfm_tutorials/images/sphx_glr_plot_mlfmmixsa_fit_003.png
            :class: sphx-glr-multi-img

    *

      .. image:: /tutorials/llfm_tutorials/images/sphx_glr_plot_mlfmmixsa_fit_004.png
            :class: sphx-glr-multi-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning: Maximum number of iterations has been exceeded.
             Current function value: 4064.989745
             Iterations: 20
             Function evaluations: 35
             Gradient evaluations: 35
    iter 1. Delta g: 1.8552149996007172
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3544.970321
             Iterations: 20
             Function evaluations: 33
             Gradient evaluations: 33
    iter 2. Delta g: 0.6942741738260573
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3474.727820
             Iterations: 20
             Function evaluations: 33
             Gradient evaluations: 33
    iter 3. Delta g: 0.2754362341607394
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3455.213064
             Iterations: 20
             Function evaluations: 33
             Gradient evaluations: 33
    iter 4. Delta g: 0.18548610603320562
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3439.270930
             Iterations: 20
             Function evaluations: 35
             Gradient evaluations: 35
    iter 5. Delta g: 0.1791234971922251
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3437.628236
             Iterations: 20
             Function evaluations: 33
             Gradient evaluations: 33
    iter 6. Delta g: 0.07936143502704221
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3436.726844
             Iterations: 20
             Function evaluations: 34
             Gradient evaluations: 34
    iter 7. Delta g: 0.06053151114696112
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3436.595303
             Iterations: 20
             Function evaluations: 32
             Gradient evaluations: 32
    iter 8. Delta g: 0.02836669114088064
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3436.533693
             Iterations: 20
             Function evaluations: 35
             Gradient evaluations: 35
    iter 9. Delta g: 0.01540882510578598
    Warning: Maximum number of iterations has been exceeded.
             Current function value: 3436.521780
             Iterations: 20
             Function evaluations: 31
             Gradient evaluations: 31
    iter 10. Delta g: 0.00944540653163691
    (28, 3, 3)




|


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pydygp.linlatentforcemodels import MLFMMixSA
    from sklearn.gaussian_process.kernels import RBF
    from pydygp.liealgebras import so
    from collections import namedtuple
    np.set_printoptions(precision=3, suppress=True)

    mlfm = MLFMMixSA(so(3), R=1, order=10, lf_kernels=[RBF(), ])

    x0 = np.eye(3)  # initial conditions the std. basis vectors of R3
    N_outputs = 3

    beta = np.random.normal(size=6).reshape(2, 3)
    beta /= np.linalg.norm(beta, axis=1)[:, None]

    tt = np.linspace(0., 6., 10)
    Data, gtrue = mlfm.sim(x0, tt, beta=beta, size=N_outputs)

    mlfm._setup_times([tt]*N_outputs, h=.25)

    # set the fixed points
    _ifix = np.linspace(0, tt.size-1, 5, dtype=np.intp)[1:-1]
    ifix = [mlfm.data_inds[0][i] for i in _ifix]

    import autograd.numpy as anp
    Normal = namedtuple('Normal', 'logpdf')
    betaprior = Normal(lambda x: -0.5 * anp.sum(x**2) - \
                       0.5 * x.size * np.log(2 * np.pi * 1.))



    init_opts = {'g0': gtrue[0](mlfm.ttc),
                 'g_is_fixed': False,
                 'beta_is_fixed': True,
                 'beta0': beta,
                 'mu_ivp_is_fixed': True,
                 'beta_prior': betaprior
                 }

    optim_opts = {
        'options': {'disp': True,
                    'maxiter': 20}
        }

    mlfm.fit([(tt, Y) for Y in Data],
             ifix,
             max_nt=10,
             optim_opts=optim_opts,
             verbose=True,
             **init_opts)

    mu_ivp = np.array([
        np.dstack([Y[_ifx, :] for Y in Data])
        for _ifx in _ifix])
    mu_ivp = mu_ivp[:, 0, ...]


    pi = np.ones(len(ifix)) / len(ifix)
    r = mlfm._get_responsibilities(pi, gtrue[0](mlfm.ttc), beta, mu_ivp, 1000)

    fig, ax = plt.subplots()
    ax.plot(tt, r[0])

    fig2, ax2 = plt.subplots()
    ax2.plot(tt, mlfm.g_[mlfm.data_inds[0]], '+')
    ax2.plot(mlfm.ttc, gtrue[0](mlfm.ttc), 'k-', alpha=0.5)

    layer = mlfm._forward(
        gtrue[0](mlfm.ttc),
        #mlfm.g_,
        beta,
        mlfm.mu_ivp_[0],
        mlfm._ifix[0])
    print(layer.shape)
    fig3, ax3 = plt.subplots()
    #ax3.plot(mlfm.ttc, layer[..., 0], 'k-', alpha=0.3)
    ax3.plot(mlfm.ttc, layer[:, 0, 0], 'b-.')
    #ax3.plot(tt, layer[mlfm.data_inds[0], :, 0], 'b-.')
    ax3.plot(tt, Data[0][:, 0], 's')
    ax3.set_ylim((-1.1, 1.1))

    fig4, ax4 = plt.subplots()
    Ar = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
          for br in beta]
    Ar_ = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
           for br in mlfm.beta_]

    ax4.plot(mlfm.ttc, Ar[0][0, 1] + Ar[1][0, 1]*gtrue[0](mlfm.ttc), 'k-')
    ax4.plot(mlfm.ttc, Ar_[0][0, 1] + Ar_[1][0, 1]*mlfm.g_, 'C0-')

    plt.show()

**Total running time of the script:** ( 0 minutes  52.220 seconds)


.. _sphx_glr_download_tutorials_llfm_tutorials_plot_mlfmmixsa_fit.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mlfmmixsa_fit.py <plot_mlfmmixsa_fit.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mlfmmixsa_fit.ipynb <plot_mlfmmixsa_fit.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
