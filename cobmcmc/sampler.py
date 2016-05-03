import numpy as np


class Sampler(object):
    """
    A generic class for samplers.
    """
    def __init__(self, dim, lnprobfn, lnprobargs, lnprobkwargs,
                 verbose=False):
        """
        :param int dim: the dimension of the problem.

        :param function lnprobfn: a function to compute the natural log of the
        pdf of the target distribution up to a constant. It takes as input the
        a parameter vector, and the optional *args, and *kwargs.

        :param list lnprobargs: extra optional arguments to pass to lnprobfn.

        :param dict lnprobkwargs: extra optional keyword arguments to pass to
        lnprobfn
        """
        self.dim = dim
        self.lnprobfn = lnprobfn
        self.lnprobfnargs = lnprobargs
        self.lnprobfnkwargs = lnprobkwargs

        # Initialize chain with no steps
        self._chain = np.zeros((0, self.dim))
        self._lnprob = np.zeros(0)

        # Set number of links to zero
        self.nlinks = 0

        # Verbose output to screen
        self.verbose = verbose

    @property
    def chain(self):
        """
        A pointer to the Markov chain.

        """
        return self._chain

    @property
    def lnprob(self):
        """
        A pointer to the lnprobability array.

        """
        return self._lnprob

    @property
    def acceptancerate(self):
        """
        A pointer to the acceptance rate
        :return:
        """
        return self.compute_acceptancerate()

    def get_lnprob(self, p):
        """
        Compute probability at a given position p

        :param array-like p: position vector where lnprob is computed.
        """
        return self.lnprobfn(p, *self.lnprobfnargs, **self.lnprobfnkwargs)

    def sample(self, *args, **kwargs):
        raise NotImplementedError("The sampling routine must be implemented "
                                  "by subclasses")

    def run_mcmc(self, n, p0):
        """
        Run a Markov Chain.
        :param int n: number of steps in the chain.
        :param array-like p0: parameter vector.
        :return:
        """

    def reset(self):
        self.nlinks = 0
        self._chain = np.zeros((0, self.dim))
        self._lnprob = np.zeros(0)

    def compute_acceptancerate(self):
        return np.cumsum(self.naccepted)/np.arange(1, self.nlinks+2,
                                                   dtype=float)


