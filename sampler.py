from math import log
import warnings

import numpy as np
import numpy.random
import scipy
from numpy.random import multivariate_normal as mvn


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


class MetropolisSampler(Sampler):
    """
    A class implementing a generic Metropolis-Hastings sampler.
    """

    def __init__(self, dim, lnprobfn, lnprobargs, lnprobkwargs,
                 proposalcov=None, verbose=False):

        # Super class call
        super(MetropolisSampler, self).__init__(dim, lnprobfn, lnprobargs,
                                                lnprobkwargs, verbose=verbose)

        # Start attributes that change at runtime.
        # number of accepted steps
        self.naccepted = np.zeros(0, dtype=int)
        self.last_lnprob = None

        # Proposal scale matrix
        if proposalcov is not None:
            self.proposalcov = proposalcov
        else:
            # randomly initialised
            x = np.random.rand(dim)
            kprop = x[:, None] + x[None, :]

            # Add power in diagonal until matrix becomes positive definite.
            # TO BE DONE
            self.proposalcov = kprop

        self.proposalcovrecord = np.array([self.proposalcov])

    def accept(self, lnprob1, proposal):
        # Set new lnprob as last lnprob
        self.last_lnprob = lnprob1

        # Add proposal step to chain
        self._chain[self.nlinks + 1] = proposal
        self._lnprob[self.nlinks + 1] = lnprob1

        # Add one to array with accepted steps
        self.naccepted[self.nlinks + 1] = 1

    def reject(self):
        # Repeat previous link in chain.
        self._chain[self.nlinks + 1] = self.chain[self.nlinks]
        self._lnprob[self.nlinks + 1] = self.last_lnprob

    def make_proposal(self):
        raise NotImplementedError('To be implemented at the sub-class level')

    def sample(self):
        # Compute ln probability for initial step
        lnprob0 = self.last_lnprob

        # Propose a new chain state
        proposal = self.make_proposal()

        # Compute log(prob) in proposal state.
        lnprob1 = self.get_lnprob(proposal)

        # Compute Metropolis ratio
        r = lnprob1 - lnprob0

        if r > 0:
            self.accept(lnprob1, proposal)
            return 1
        elif log(numpy.random.rand()) < r:
            self.accept(lnprob1, proposal)
            return 1
        else:
            self.reject()
            return 0


def compute_covariance(m):
    # Subtract mean from each parameter

    c = np.cov(m, rowvar=0)

    try:
        d = np.diag(c)
    except ValueError:  # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    # Product of covariances
    covprod = np.sqrt(np.multiply.outer(d, d))

    """
    mean_p = np.mean(m, axis=0)
    std_p = np.std(m, axis=0)
    return np.cov(m - mean_p, rowvar=0), mean_p, std_p
    """
    return c / covprod, m.mean(axis=0), np.sqrt(np.diag(covprod))


class AdaptivePCASampler(MetropolisSampler):
    """
    A class implementing the adaptive PCA sampling algorithm of Diaz+2014.

    :param int npca: number of chain steps to use to compute covariance
    matrix.

    :param int nupdatepca: interval between updates of covariance matrix.

    :param int startpca: first iteration on which PCA is used.
    """
    def __init__(self, dim, lnprobfn, lnprobfnargs, lnprobfnkwargs,
                 proposalcov=None, startpca=10000, npca=100, verbose=False,
                 **kwargs):

        super(AdaptivePCASampler, self).__init__(dim, lnprobfn, lnprobfnargs,
                                                 lnprobfnkwargs, proposalcov,
                                                 verbose)
        self.startpca = startpca
        self.npca = npca
        self.nupdatepca = kwargs.pop('nupdatepca', np.inf)

        if self.npca > self.startpca:
            warnings.warn('Npca > Startpca; changing Npca = startpca')
            self.npca = self.startpca

        # Initialise change of basis matrix
        self.cobmatrix = np.empty((dim, dim))
        self.meanp = None
        self.stdp = None
        #
        # Initialise arrays for bookkeeping
        #

        # Array to keep number of proposals per parameter and iteration
        self.proposalrecord = np.zeros((1, dim))

        # Array to keep number of acceptances per parameter and interation
        self.acceptancerecord = np.zeros((1, dim))

        # Bookkeeping of PCA proposals
        # self.pcaproposalrecord = np.empty((0, dim))
        # self.pcaacceptancerecord = np.empty((0, dim))

        # Arrays to keep number of proposals and acceptance between updates.
        self.nproposed_since_update = np.zeros(dim)
        self.naccepted_since_update = np.zeros(dim)

        # Initialise update interval per parameter
        self.update_interval = np.full(dim, 2)
        self.last_scalefactor = np.full(dim, np.nan)

    def bookkeeping(self, indexvector, accept):

        i = self.nlinks

        # For all iterations keep record of number of proposals
        self.proposalrecord[i] = self.proposalrecord[i-1] + indexvector

        # Record proposal covariance for this step (even if identical)
        self.proposalcovrecord[i] = self.proposalcov

        #  Increase number of proposed steps since update per parameter
        self.nproposed_since_update += indexvector

        if accept:
            # Keep record of number of acceptances
            self.acceptancerecord[i] = self.acceptancerecord[i-1] + indexvector

            # Increase number of accepted steps since update
            self.naccepted_since_update += indexvector

        else:
            # Keep record of number of acceptances (in this case, just copy)
            self.acceptancerecord[i] = self.acceptancerecord[i-1]

    def make_proposal(self, onebyone=True):
        """
        The proposal function for the Adaptive PCA sampler.

        :param bool onebyone: determines whether jumps are done on all
        parameters at the same time or one by one.
        :return:
        """
        if onebyone:
            # Jump one parameter at a time. Choose randomly
            jumpind = np.random.randint(0, self.dim)

            # Array indicating which parameter jumps
            self.jump = np.zeros(self.dim, dtype=int)
            self.jump[jumpind] += 1

            # Get scale from proposal covariance
            jumpscale = np.sqrt(np.diag(self.proposalcov))

            # Proposed jump (for single parameter, zero for the rest)
            proposedjump = self.jump * np.random.randn() * jumpscale

        else:
            # Jump all parameters
            self.jump = np.ones(self.dim, dtype=int)

            # Proposed jump
            proposedjump = mvn(mean=np.zeros(self.dim), cov=self.proposalcov)

        # Proposing without PCA
        if self.nlinks < self.startpca:
            # Starting point is last point in chain
            startingpoint = self.chain[self.nlinks]

        # Propsing with PCA
        else:
            # Make proposal in rotated parameter space.
            # Convert last chain link to principal component space
            x = (self.chain[self.nlinks] - self.meanp)/self.stdp
            startingpoint = np.dot(self.cobmatrix, x)

        newpoint = startingpoint + proposedjump

        if self.nlinks < self.startpca:
            return newpoint
        else:
            # Convert back to parameter space
            return np.dot(self.cobmatrix.T, newpoint) * self.stdp + self.meanp

    def pca(self):
        # Update proposal scales and PCA matrix.
        if self.nlinks == self.startpca:
            print('STARTING PRINCIPAL COMPONENT ANALYSIS')
        else:
            print('Updating covariance matrix for PCA')

        # Compute matrix of correlation coefficients on last npca steps
        corrmatrix, self.meanp, self.stdp = compute_covariance(
            self.chain[self.nlinks - self.npca:self.nlinks])

        # Compute eigenvectors of covariance matrix
        eigval, eigvec = scipy.linalg.eigh(corrmatrix)

        # Set new change of basis matrix
        self.cobmatrix = eigvec.T

        # Use eigenvalues as proposed parameter scale for pc's
        self.proposalcov = np.diag(np.abs(eigval))

        # Reset counters of updates
        self.nproposed_since_update[self.jump != 0] = 0
        self.naccepted_since_update[self.jump != 0] = 0

        return

    def update_proposal_scale(self, target_rate=0.25):
        """
        Update proposal scales using adapted version of the prescription by
        Ford (ApJ 642 505, 2006)

        :param float target_rate: target fraction of proposals to accept.
        """
        # Compute variance of estimated acceptance rate acceptance_rate.
        # This is: psi0*(1 - psi0)/number_steps_since_update
        # If the squared difference (acceptance_rate - psi0)**2 is larger
        # than a few times this value, then the scale must be updated
        update_stat = self.update_interval * target_rate * (1 - target_rate) / \
            self.nproposed_since_update

        # Compute acceptance rate
        acceptance_rate = np.where(self.nproposed_since_update > 0,
                                   self.naccepted_since_update /
                                   self.nproposed_since_update,
                                   target_rate)

        # condition for updating
        condupdate = ((acceptance_rate - target_rate) ** 2 > update_stat) * (self.nproposed_since_update > 100)

        # If no parameter needs updating, exit.
        if np.all(~condupdate):
            return

        rho = acceptance_rate/target_rate

        # Array with scale factor initialised to 1/100
        k = np.full(self.dim, 0.01)

        # Compute factoring scales for different ranges of ratio rho
        k[rho > 2] = rho[rho > 2]**2.0
        k[(rho > 1) * (rho <= 2)] = rho[(rho > 1) * (rho <= 2)]
        k[(rho > 0.5) * (rho <= 1)] = np.sqrt(rho[(rho > 0.5) * (rho <= 1)])
        k[(rho > 0.2) * (rho <= 0.5)] = rho[(rho > 0.2) * (rho <= 0.5)]**1.5
        k[(rho > 0.1) * (rho <= 0.2)] = rho[(rho > 0.1) * (rho <= 0.2)]**2

        # Set all parameters where updating is not necessary to one
        k[~condupdate] = 1.0

        # Increase mean update interval for parameters whose scale has
        # changed in opposite directions in successive updates.
        cond = (self.last_scalefactor - 1)*(k - 1) < 0.0
        self.update_interval[cond * ~np.isnan(self.last_scalefactor)] += 1

        # Update proposal scale
        newpropsalcov = self.proposalcov.copy()
        # New scale for covariance.
        newscale = np.diag(self.proposalcov) * k**2
        newscale = np.where(newscale != 0, newscale,
                            self.chain[self.nlinks] * 0.1)

        newpropsalcov[np.diag_indices_from(newpropsalcov)] = newscale

        self.last_scalefactor = k

        # Reset counters of updates
        self.nproposed_since_update[self.jump != 0] = 0
        self.naccepted_since_update[self.jump != 0] = 0

        if self.verbose:
            for j, upbool in enumerate(condupdate):
                if upbool:
                    print('Scale adjusted for parameter {}; old scale: {:.2e}; '
                          'new scale: {:.2e}'
                          ''.format(j,
                                    np.sqrt(np.diag(self.proposalcov)[j]),
                                    np.sqrt(np.diag(newpropsalcov)[j])))
        # Update proposal covariance
        self.proposalcov = newpropsalcov
        return

    def initialise_arrays(self, n):

        # Add n empty elements to chain.
        self._chain = np.concatenate((self._chain, np.empty((n, self.dim))),
                                     axis=0)

        self._lnprob = np.concatenate((self._lnprob, np.empty(n)), axis=0)

        # Add n elements to bookkeeping arrays
        self.naccepted = np.concatenate((self.naccepted, np.zeros(n, dtype=int))
                                        )

        self.proposalrecord = np.concatenate((self.proposalrecord,
                                              np.empty((n, self.dim), dtype=int)
                                              ))

        self.acceptancerecord = np.concatenate((self.acceptancerecord,
                                                np.empty((n, self.dim),
                                                         dtype=int)
                                                ))

        self.proposalcovrecord = np.concatenate((self.proposalcovrecord,
                                                 np.empty((n, self.dim,
                                                           self.dim))),
                                                axis=0)

    def run_mcmc(self, n, p0=None, lnprob0=None):
        """
        Run Markov Chain Monte Carlo algorithm for n iterations starting from
        point p0

        :param int n: number of iterations.

        :param array_like p0: initial position in paramter space.

        Optional arguments
        :param float lnprob0: value of target distribution in initial step.

        :return None:
        """

        # If sampler already started before, get number of links already
        # in chain.
        self.nlinks = self._chain.shape[0]

        self.initialise_arrays(n)

        # If fresh run, try to compute lnprob at starting point.
        if self.nlinks == 0:
            print('Fresh Markov Chain. Adding first link using starting point.')
            try:
                self.last_lnprob = self.get_lnprob(p0)
            except:
                raise AttributeError('First evaluation of target distribution'
                                     'produced and error.')

            # Initialise chain to starting point
            self._chain[self.nlinks] = p0
            self._lnprob[self.nlinks] = self.last_lnprob
            n -= 1

        else:
            self.nlinks -= 1

        for i in xrange(n):

            if self.verbose:
                if i % 500 == 0:
                    print('#### Iteration {} out of {}'.format(i, n))
                if i % 3000 == 0:
                    for j in range(self.chain.shape[1]):
                        print('Param {0}: {1}'.format(j,
                                                      self.chain[self.nlinks, j]
                                                      ))
                    print('## ln(prior*likelihood)'
                          ' = {}'.format(self.lnprob[self.nlinks]))

            # Update PCA matrix every nupdatepca iterations.
            if self.nlinks >= self.startpca and \
               (self.nlinks - self.startpca) % self.nupdatepca == 0:
                self.pca()

            # Sample
            acceptflag = self.sample()

            # Increase number of links in chain
            self.nlinks += 1
            self.bookkeeping(self.jump, accept=acceptflag)

            # Update proposal scales
            self.update_proposal_scale(0.25)

        return

__author__ = 'Rodrigo F. Diaz'
