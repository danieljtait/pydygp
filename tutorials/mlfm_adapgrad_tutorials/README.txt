
Adaptive Gradient Matching Methods
==================================

For the MLFM AdapGrad model the likelihood of a set of state variables may be written

.. math::

   \begin{align}
   p(\mathbf{X} \mid \mathbf{g} ) \propto
   \prod_{k=1}^{K} \exp\bigg\{
   &-\frac{1}{2}
   (\mathbf{f}_k - \mathbf{m}_{\dot{x}_k|x_k})^{\top}
   (\mathbf{C}_{\dot{x}_k|\dot{x}} + \gamma I)^{-1}
   (\mathbf{f}_k - \mathbf{m}_{\dot{x}_k|x_k}) \\
   &-\frac{1}{2}\mathbf{x}_k^{\top}\mathbf{C}_{x_k}\mathbf{x}
   \bigg\},
   \end{align}

where :math:`\mathbf{f}_k` is the vector of components of the evolution equation,
the entries of which are given by

.. math::

   f_{kn} = 
   \sum_{r=0}^{R} g_{rn} \sum_{d=1}^D \beta_{rd}
   \sum_{j=1}^{K} L_{dkj} x_{jn},

One interesting point from the perspective of identifiability of models of these
types is that the likelihood-term will remain invariant under choices of
:math:`\mathbf{g}, \boldsymbol{\beta}` such that :math:`\mathbf{f}_k` remains
invariant.

All of the adative gradient matching methods proceed from this conditional density

Model Parameters
~~~~~~~~~~~~~~~~

The following table gives the complete collection of variables that appear in the adaptive gradient
matching method for the MLFM, along with a brief description of this variables, how this variable
is referred to when using the package, along with transformation that is applied to this variable
to give it a more natural support.

+-----------------------------+-----------------------------+------------------+---------------------+---------------+ 
| Parameter name              | Description                 | Variable name    | Transform           | Is Fixed      |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+
| :math:`\mathbf{g}`          | The (vectorised) latent GPs | :code:`g`        | :math:`\mathrm{Id}` | :code:`False` |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+
| :math:`\boldsymbol{\psi}`   | latent GP hyperparameters   | :code:`logpsi`   | :math:`\log`        | :code:`False` |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+
| :math:`\boldsymbol{\beta}`  | Basis coefficients          | :code:`beta`     | :math:`\mathrm{Id}` | :code:`False` |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+
| :math:`\boldsymbol{\tau}`   | Observation precisions      | :code:`logtau`   | :math:`\log`        | :code:`False` |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+
| :math:`\boldsymbol{\gamma}` | ODE model regularisation    | :code:`loggamma` | :math:`\log`        | :code:`True`  |
+-----------------------------+-----------------------------+------------------+---------------------+---------------+

MAP Estimation
~~~~~~~~~~~~~~

    .. include:: ../../doc/source/gen_modules/backreferences/pydygp.linlatentforcemodels.MLFMAdapGrad.examples
    .. raw:: html

        <div style='clear:both'></div>
