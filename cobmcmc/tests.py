# coding=utf-8
import numpy as np
import scipy.stats
from .cobsampler import ChangeOfBasisSampler


class Test(object):
    """
    Super class implementing tests for CoBSampler. Sub-classes should specify
    target distribution.
    """

    def __init__(self, ndim, target, nsteps):
        self.ndim = ndim
        self.targetdist = target
        self.niterations = nsteps

    def run(self):
        # initialise sampler
        sampler = ChangeOfBasisSampler(self.ndim, self.targetdist.logpdf, (),
                                       {}, startpca=np.inf)
                                       # startpca=self.niterations/10,
                                       # nupdatepca=self.niterations/10)

        p0 = np.zeros(self.ndim)
        sampler.run_mcmc(self.niterations, p0)
        chain = sampler.chain

        return sampler


class MultinormalTest(Test):
    """
    Class implementing test on multinormal distribution.
    """

    def __init__(self, nsteps, ndim=2, cov=None):
        """

        :param int nsteps: number of MCMC iterations.
        :param int ndim: target dimension
        :param np.array cov: covariance matrix. If None, random covariance is
         constructed.
        """
        target = Multinormal(ndim, cov)
        super(MultinormalTest, self).__init__(ndim, target, nsteps)


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
