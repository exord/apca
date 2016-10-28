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

    def run(self, p0):
        # initialise sampler
        sampler = self.sampler(self.ndim, self.targetdist.logpdf, (),
                               {}, startpca=self.firscob,
                               nupdatepca=self.updatecob,
                               npca=self.ncob)

        # p0 = np.zeros(self.ndim)
        sampler.run_mcmc(self.niterations, p0)

        return sampler


class MultinormalTest(Test):
    """
    Class implementing test on multinormal distribution.
    """

    def __init__(self, nsteps, ndim=2, mean=None, cov=None, cobparams={}):
        """

        :param int nsteps: number of MCMC iterations.
        :param int ndim: target dimension
        :param np.array cov: covariance matrix. If None, random covariance is
         constructed.
        """
        target = Multinormal(ndim, mean, cov)
        super(MultinormalTest, self).__init__(ndim, target, nsteps, cobparams)


class RosenbrockTest(Test):
    """
    Class implementing test on Rosenbrock density.
    """
    def __init__(self, nsteps, a=1, b=100, cobparams={}):
        target = Rosenbrock(a, b, 2)
        super(RosenbrockTest, self).__init__(2, target, nsteps, cobparams)


class TargetDistribution(object):
    """
    Class for test target distributions.
    """


class Multinormal(TargetDistribution):
    def __init__(self, ndim=2, mean=None, cov=None):

        self.ndim = ndim

        if mean is None:
            mean = np.zeros(ndim)
        else:
            assert len(mean) == ndim, 'Dimensions of mean arry do no match ' \
                                      'of dimensions.'
        self.mean = mean

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
    def __init__(self, a=1, b=100, ndim=2):
        self.a = a
        self.b = b
        self.ndim = ndim

    def pdf(self, x):
        if (np.abs(x[0]) > 10) or (np.abs(x[1]) > 10):
            return 0
        return np.exp(-(self.a - x[0])**2 - self.b*(x[1] - x[0]**2)**2)

    def logpdf(self, x):
        if (np.abs(x[0]) > 30) or (np.abs(x[1]) > 30):
            return -np.inf
        else:
            return -(self.a - x[0])**2 - self.b*(x[1] - x[0]*x[0])**2

    def contour(self, k, n=1000):
        """
        :param float k: constant identifying contour.
        :param int n: number of points used to construct contour.
        """
        x = np.linspace(self.a - k, self.a + k, n)
        yplus = x**2 + np.sqrt( (k**2 - (x - self.a)**2)/self.b )
        yminus = x ** 2 - np.sqrt((k ** 2 - (x - self.a) ** 2) / self.b)

        xx = np.concatenate((x, x[::-1]))
        yy = np.concatenate((yminus, yplus[::-1]))
        return np.array([xx, yy])

    def rvs(self, size=1):
        """
        Draw samples from the Rosenbrock density.
        Uses the fact that p(x1,x2) = p(x2|x1)*p(x1) and that:
        1) p(x1) \propto N(a, 1)
        2) p(x2|x1) \propto N(x1**2, 1/sqrt(2*b))
        """
        # Draw samples from marginal p(x1)
        x1 = np.random.randn(size) + self.a

        # Draw samples from conditional, p(x2 | x1)
        sigma = 1./np.sqrt(2 * self.b)
        x2 = np.random.randn(size) * sigma + x1**2

        return np.array([x1, x2]).T
