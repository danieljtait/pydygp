.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_tutorials_llfm_tutorials_plot_mlfmsamix.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_tutorials_llfm_tutorials_plot_mlfmsamix.py:


Fitting of the MLFM-MixSA Model 2
=================================




.. code-block:: pytb

    Traceback (most recent call last):
      File "/Users/danieltait/Desktop/pydygp/tutorials/llfm_tutorials/plot_mlfmsamix.py", line 39, in <module>
        mu_ivp = np.array([d[_ifix, :] for d in Data])
      File "/Users/danieltait/Desktop/pydygp/tutorials/llfm_tutorials/plot_mlfmsamix.py", line 39, in <listcomp>
        mu_ivp = np.array([d[_ifix, :] for d in Data])
    NameError: name '_ifix' is not defined





.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from pydygp.linlatentforcemodels import MLFMSAMix
    from sklearn.gaussian_process.kernels import RBF
    from pydygp.probabilitydistributions import Normal
    from pydygp.liealgebras import so
    np.set_printoptions(precision=3, suppress=True)

    mlfm = MLFMSAMix(so(3), R=1, order=4, lf_kernels=[RBF(), ])

    x0 = np.eye(3) # inital conditions the std. basis vectors of R3
    N_repl = 3     # outputs of the experiment

    beta = np.random.normal(size=6).reshape(2, 3)
    beta /= np.linalg.norm(beta, axis=1)[:, None]
    #beta = np.array([[0., 0., 0.],
    #                 [-0.5, 0.31, 0.11]])

    tt = np.linspace(0., 2., 5)

    Data, gtrue = mlfm.sim(x0, tt, beta=beta, size=N_repl)
    experiments = [(tt, y) for y in Data]

    # setup the time vectors and augment them so maximum
    # time step is h
    mlfm._setup_times([tt]*N_repl, h=None)

    # indicies in complete time vector of the initial value problem
    #  -> super ugly and hacky at the moment
    #_ifix = np.linspace(0, tt.size-1, 3, dtype=np.intp)
    #_ifix = [tt.size // 3, 2*tt.size // 3]
    #ifix = [mlfm.data_inds[0][i] for i in _ifix]


    mu_ivp = np.array([d[_ifix, :] for d in Data])

    import time
    # Model fitting
    beta_prior = Normal(scale=5.)*6
    t0 = time.time()
    g, beta_,r  = mlfm.fit(experiments, ifix,
                           beta0=beta, beta_is_fixed=False,
                           beta_prior=beta_prior,
                           mu_ivp_is_fixed=True, mu_ivp0=mu_ivp)
    t1 = time.time()
    print("Mixture model took {}".format(t1-t0))


    from pydygp.linlatentforcemodels import MLFMAdapGrad
    print("Starting AG fit...")
    tstart = time.time()
    Y = np.column_stack((y.T.ravel() for y in Data))
    mlfmag = MLFMAdapGrad(so(3), R=1, lf_kernels=[RBF(), ])
    res_ag = mlfmag.fit(tt, Y, beta0=beta, beta_is_fixed=False,
                        beta_prior=beta_prior,
                        logpsi_is_fixed=True, logtau_is_fixed=True,
                        optim_options={'disp': True,})
    print(res_ag.optimres.nfev, res_ag.optimres.nit)
    tstop = time.time()
    print("... Done. {}".format(tstop-tstart))

    from pydygp.linlatentforcemodels import MLFMSA
    obj = MLFMSA(so(3), R=1, lf_kernels=[RBF(),], order=3)
    obj._setup_times([tt]*N_repl, h=.25, multi_output=True)
    obj.y_train_ = [d.T.ravel() for d in Data]
    basis_funcs = [(lambda x, t0: (x-t0)**2, )]*len(ifix)
    obj._setup_softmax(ifix, basis_funcs)
    v = [-.0]*len(ifix)
    alf = 2000
    pi = [obj.softmax_activs(v, x[:, None]) for x in obj.x_train_]
    r = [p.copy() for p in pi]
    """
    print("Start mixture fit...")
    tstart = time.time()
    ghat = gtrue[0](obj.ttc) #
    for nt in range(5):
        ghat, mu_ivp = obj._optim_g(ghat, beta, mu_ivp, alf, r, ifix, None)
        pi = sum(np.mean(rm, axis=0) for rm in r) / len(r)
        pi = [np.row_stack([np.mean(rm, axis=0)]*rm.shape[0])
              for rm in r]
        # update repsonsibilitees
        r = obj._get_responsibilities(pi, ghat, beta, mu_ivp, alf, ifix)

    print("-----")
    print(r[0].sum(axis=0))
    print(np.mean(r[0], axis=0))
    print("-----")
    tstop = time.time()    
    print("... Done. {}".format(tstop-tstart))
    """

    fig, ax = plt.subplots()
    ax.plot(tt, g[mlfm.data_inds[0]], 'C0o')
    ax.plot(mlfm.ttc, gtrue[0](mlfm.ttc), '-')
    ax.plot(mlfmag.ttc, res_ag.g.T, 'ks')


    print("=========================")
    print(beta)
    print(beta_)
    print(res_ag.beta)

    gmix = g[mlfm.data_inds[0]]
    Amix = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
                for br in beta_]
    Aag = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
               for br in res_ag.beta]
    Atrue = [sum(brd*Ld for brd, Ld in zip(br, mlfm.basis_mats))
             for br in beta]

    a12 = Amix[0][0, 1] + Amix[1][0, 1] * gmix
    a13 = Amix[0][0, 2] + Amix[1][0, 2] * gmix
    a23 = Amix[0][1, 2] + Amix[1][1, 2] * gmix

    b12 = Aag[0][0, 1] + Aag[1][0, 1] * res_ag.g.ravel()
    b13 = Aag[0][0, 2] + Aag[1][0, 2] * res_ag.g.ravel()
    b23 = Aag[0][1, 2] + Aag[1][1, 2] * res_ag.g.ravel()

    c12 = Atrue[0][0, 1] + Atrue[1][0, 1] * gtrue[0](tt)
    c13 = Atrue[0][0, 2] + Atrue[1][0, 2] * gtrue[0](tt)
    c23 = Atrue[0][1, 2] + Atrue[1][1, 2] * gtrue[0](tt)

    fig, ax = plt.subplots()
    ax.plot(tt, a13, label=r'$a_{13}$ mix')
    ax.plot(tt, b13, '--', label=r'$a_{13}$ ag')
    ax.plot(tt, c13, label=r'$a_{13}$ true')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(tt, a12, label=r'$a_{12}$ mix')
    ax.plot(tt, b12, label=r'$a_{12}$ ag')
    ax.plot(tt, c12, label=r'$a_{12}$ true')
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(tt, a23, label=r'$a_{23}$ mix')
    ax.plot(tt, b23, label=r'$a_{23}$ ag')
    ax.plot(tt, c23, label=r'$a_{23}$ true')
    ax.legend()
    plt.show()




**Total running time of the script:** ( 0 minutes  0.000 seconds)


.. _sphx_glr_download_tutorials_llfm_tutorials_plot_mlfmsamix.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_mlfmsamix.py <plot_mlfmsamix.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_mlfmsamix.ipynb <plot_mlfmsamix.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
