# coding=utf-8
import numpy as np
import scipy.stats
from .cobsampler import ChangeOfBasisSampler


class Test(object):
    """
    Super class implementing tests for CoBSampler. Sub-classes should specify
    target distribution.
    """

    def __init__(self, ndim, target, nsteps, cobparams={}):
        self.ndim = ndim
        self.targetdist = target
        self.niterations = nsteps

        self.firscob = cobparams.pop('firstcob', 1000)
        self.ncob = cobparams.pop('ncob', 1000)
        self.updatecob = cobparams.pop('updatecob', 1000)

        self.sampler = ChangeOfBasisSampler

    def run(self):
        # initialise sampler
        sampler = self.sampler(self.ndim, self.targetdist.logpdf, (),
                               {}, startpca=self.firscob,
                               nupdatepca=self.updatecob,
                               npca=self.ncob)

        p0 = np.zeros(self.ndim)
        sampler.run_mcmc(self.niterations, p0)

        return sampler


class MultinormalTest(Test):
    """
    Class implementing test on multinormal distribution.
    """

    def __init__(self, nsteps, ndim=2, cov=None, cobparams={}):
        """

        :param int nsteps: number of MCMC iterations.
        :param int ndim: target dimension
        :param np.array cov: covariance matrix. If None, random covariance is
         constructed.
        """
        target = Multinormal(ndim, cov)
        super(MultinormalTest, self).__init__(ndim, target, nsteps, cobparams)


class TargetDistribution(object):
    """
    Class for test target distributions.
    """


class Multinormal(TargetDistribution):
    def __init__(self, ndim=2, cov=None):

        self.ndim = ndim
        if cov is not None:
            assert cov.shape == (ndim, ndim), 'Dimensions of covariance ' \
                                              'matrix do no match.'
            self.cov = cov
        else:
            # If covariance is not given, initialise at random.
            self.cov = 0.5 - np.random.rand(self.ndim ** 2).reshape((self.ndim,
                                                                     self.ndim))
            self.cov = np.triu(self.cov)
            self.cov += self.cov.T - np.diag(self.cov.diagonal())
            self.cov = np.dot(self.cov, self.cov)

        self.mean = np.zeros(ndim)
        self.dist = scipy.stats.multivariate_normal(mean=self.mean,
                                                    cov=self.cov)

    def pdf(self, x):
        """
        Return value of pdf at point x
        :param np.array x: Position in parameter space.
        :return:
        """
        return self.dist.pdf(x)

    def logpdf(self, x):
        return self.dist.logpdf(x)


class Rosenbrock(TargetDistribution):
    """
    Class implementing the Rosenbrock density.
    """
    def __init__(self, ndim):
        self.ndim = ndim

    @staticmethod
    def pdf(x):
        if (np.abs(x[0]) > 10) or (np.abs(x[1]) > 10):
            return 0
        return np.exp(-0.05*((1 - x[0])**2 + 100*(x[1] - x[0]**2)**2))

    @staticmethod
    def logpdf(x):
        if (np.abs(x[0]) > 30) or (np.abs(x[1]) > 30):
            return -np.inf
        else:
            return -0.05*((1 - x[0])**2 + 100*(x[1] - x[0]*x[0])**2)
