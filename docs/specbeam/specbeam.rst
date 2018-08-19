********
specbeam
********

Contents

.. toctree::


What is specbeam?
=================

**specbeam** is a python module developed for Walker and Indraratna (2018ish) [2]_
article " Moving loads on a viscoelastic foundation with special reference to
railway transition zones" published in the International Journal of Geomechanics.

It models a finite elastic Euler-Bernoulli beam resting on
viscoelastic foundation subjected to a moving load(s), with piecewise-linear
spatially varying properties.  It uses the spectral Galerkin method to solve the
dynamic system deflections over time. Ding et al. (2012) [1]_  models a similar
situation with constant spatial properties.


See :class:`geotecha.beam_on_foundation.specbeam.SpecBeam` for details
and input variables. for specbeam

There are various use case examples floating around.  In particular
see the article_figure functions in :mod:`geotecha.beam_on_foundation.specbeam`
module and also the testing routines
in :mod:`geotecha.beam_on_foundation.test.test_specbeam`
There is also a few examples can be found in the :ref:`examples-index` section.

The equations of Ding et al(2012 [1]_ are implemented in
:mod:`geotecha.beam_on_foundation.dingetal2012.py`.



References
==========

.. [1] Ding, H., Chen, L.-Q., and Yang, S.-P. (2012).
   "Convergence of Galerkin truncation for dynamic response of
   finite beams on nonlinear foundations under a moving load."
   Journal of Sound and Vibration, 331(10), 2426-2442.

.. [2] Walker, R.T.R. and Indraratna, B, (in press) "Moving loads on a
   viscoelastic foundation with special reference to railway
   transition zones". International Journal of Geomechanics.




