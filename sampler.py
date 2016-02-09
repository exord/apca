import numpy as np
import numpy.random
from numpy.random import multivariate_normal as mvn
import scipy
from math import log


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


class MetropolisSampler(Sampler):
    """
    A class implementing a generic Metropolis-Hastings sampler.
    """

    def __init__(self, dim, lnprobfn, lnprobargs, lnprobkwargs,
                 verbose=False):

        # Super class call
        super(MetropolisSampler, self).__init__(dim, lnprobfn, lnprobargs,
                                                lnprobkwargs, verbose=verbose)

        # Start attributes that change at runtime.
        # number of accepted steps
        self.naccepted = 0
        self.last_lnprob = None

        # Array to keep number of proposals per parameter and iteration
        self.proposalrecord = np.empty((0, dim))
        self.acceptancerecord = np.empty((0, dim))

        # Array that contain moment of last proposal per parameter
        self.lastproposal = np.zeros(dim)
        # Array that contain moment of last acceptance per parameter
        self.lastacceptance = np.zeros(dim)
        # Array that contain moment of last proposal update per parameter
        self.lastupdate = np.zeros(dim)

        # Proposal scale matrix
        self.proposalcov = np.empty((0, dim, dim))

    def accept(self, lnprob1, proposal):
        # Set new lnprob as last lnprob
        self.last_lnprob = lnprob1

        # Add proposal step to chain
        self.chain[self.nlinks + 1] = proposal
        self._lnprob[self.nlinks + 1] = lnprob1
        self.naccepted += 1

    def reject(self):
        # Repeat previous link in chain.
        self.chain[self.nlinks + 1] = self.chain[self.nlinks]
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
            self.nlinks += 1
            return 1
        elif log(numpy.random.rand()) < r:
            self.accept(lnprob1, proposal)
            self.nlinks += 1
            return 1
        else:
            self.reject()
            self.nlinks += 1
            return 0


def compute_covariance(m):
    # Subtract mean from each parameter
    mean_p = np.mean(m, axis=0)
    return np.cov(m - mean_p), mean_p


class AdaptivePCASampler(MetropolisSampler):
    """
    A class implementing the adaptive PCA sampling algorithm of Diaz+2014.

    :param int npca: number of chain steps to use to compute covariance
    matrix.

    :param int nupdatepca: interval between updates of covariance matrix.

    :param int startpca: first iteration on which PCA is used.
    """
    def __init__(self, dim, lnprobfn, lnprobargs, lnprobkwargs,
                 startpca=10000, npca=100, verbose=False, **kwargs):

        super(AdaptivePCASampler, self).__init__(dim, lnprobfn, lnprobargs,
                                                 lnprobkwargs, verbose=verbose)

        self.startpca = startpca
        self.npca = npca
        self.nupdatepca = kwargs.pop('nupdatepca', np.inf)

        # Initialise change of basis matrix
        self.cobmatrix = np.empty((dim, dim))

        self.pcaproposalrecord = np.empty((0, dim))
        self.pcaacceptancerecord = np.empty((0, dim))

        # Initialise proposal scale for principal components
        self.pc_proposalscale = np.zeros((0, self.dim))

        # Initialise update interval per parameter to 100
        self.update_interval = np.full(dim, 100)
        self.last_scalefactor = np.full(dim, np.nan)

    def bookkeeping(self, indexvector, action):
        if action == 'proposal':
            # Keer record of number of proposals
            self.proposalrecord = np.vstack((self.proposalrecord,
                                             self.proposalrecord[-1] +
                                             indexvector))
            # Record moment of proposal
            self.lastproposal[indexvector != 0] = indexvector * self.nlinks

        elif action == 'acceptance':
            # Keep record of number of acceptances
            self.acceptancerecord = np.vstack((self.acceptancerecord,
                                               self.acceptancerecord[-1] +
                                               indexvector))
            # Record moment of acceptance
            self.lastacceptance[indexvector != 0] = indexvector * self.nlinks

        elif action == 'update':
            # Record moment of update
            self.lastupdate[indexvector != 0] = indexvector * self.nlinks

    def make_proposal(self, onebyone=True):
        """
        The proposal function for the Adaptive PCA sampler.

        :param bool onebyone: determines whether jumps are done on all
        parameters at the same time or one by one.
        :return:
        """
        # Proposing without PCA
        if self.nlinks < self.startpca:
            if onebyone:
                # Jump one parameter at a time. Choose randomly
                jumpind = np.random.randint(0, self.dim)

                # Array indicating which parameter jumps
                self.jump = np.zeros(self.dim, dtype=int)
                self.jump[jumpind] += 1

                # Keep records
                self.bookkeeping(self.jump, action='proposal')

                # Get scale from proposal covariance
                jumpscale = np.diag(self.proposalcov[-1])[jumpind]

                return self.chain[self.nlinks] + self.jump * np.random.randn() \
                    * jumpscale

            else:
                self.jump = np.ones(self.dim, dtype=int)
                self.bookkeeping(self.jump, action='proposal')

                # Jump all parameters using covariance matrix
                return mvn(mean=self.chain[self.nlinks],
                           cov=self.proposalcov[-1])

        # Proposing with PCA
        else:
            # Make proposal in rotated parameter space.
            # Convert last chain link to principal component space
            pclast = np.dot(self.cobmatrix, self.chain[self.nlinks])

            if onebyone:
                # Jump one parameter at a time. Choose randomly
                jumpind = np.random.randint(0, self.dim)

                # Array indicating which parameter jumps
                self.jump = np.zeros(self.dim, dtype=int)
                self.jump[jumpind] += 1

                # Keep records
                self.bookkeeping(self.jump, action='PCAproposal')

                # Get scale from proposal covariance
                jumpscale = self.pc_proposalscale[-1][jumpind]

                # New proposal in rotated space
                pcprop = pclast + self.jump * np.random.randn() * jumpscale

            else:
                self.jump = np.ones(self.dim)
                self.bookkeeping(self.jump, action='PCAproposal')

                # Jump all parameters at the same time.
                pcprop = pclast + np.random.rand(self.dim) * \
                    self.pc_proposalscale[-1]

            # Go back to regular parameter space
            return np.dot(self.cobmatrix.T, pcprop)

    def pca(self):
        # Update proposal scales and PCA matrix.
        if self.nlinks == self.startpca:
            print('STARTING PRINCIPAL COMPONENT ANALYSIS')
        else:
            print('Updating covariance matrix for PCA')

        # Compute covariance matrix on last npca steps
        covmatrix, meanp = compute_covariance(self.chain[self.nlinks -
                                                         self.npca:self.nlinks])

        # Compute eigenvectors of covariance matrix
        eigval, eigvec = scipy.linalg.eigh(covmatrix)

        # Set new change of basis matrix
        self.cobmatrix = eigvec.T

        # Use eigenvalues as proposed parameter scale for pc's
        self.pc_proposalscale = np.concatenate((self.pc_proposalscale,
                                                np.abs(eigval)), axis=0)
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
            (self.nlinks - self.lastupdate)

        # Compute acceptance rate
        naccepted_since_update = self.acceptancerecord[-1] - \
            self.acceptancerecord[self.lastupdate]
        nproposed_since_update = self.proposalrecord[-1] - \
            self.proposalrecord[self.lastupdate]
        acceptance_rate = naccepted_since_update / nproposed_since_update

        # condition for updating
        condupdate = (acceptance_rate - target_rate) ** 2 > update_stat

        # If no parameter needs updating, exit.
        if np.all(~condupdate):
            return

        rho = acceptance_rate/target_rate

        # Array with scale factor initialised to 1/100
        k = np.full(self.dim, 0.01)

        # Compute factoring scales for different ranges of ratio rho
        k[rho > 2] = rho**2.0
        k[1 < rho <= 2] = rho
        k[0.5 < rho <= 1] = np.sqrt(rho)
        k[0.2 < rho <= 0.5*rho] = rho**1.5
        k[0.1 < rho <= 0.2*rho] = rho**2.0

        # Set all parameters where updating is not necessary to one
        k[~condupdate] = 1.0

        # Increase mean update interval for parameters whose scale has
        # changed in opposite directions in successive updates.
        cond = (self.last_scalefactor - 1)*(k - 1) < 0.0
        self.update_interval[cond * ~np.isnan(self.last_scalefactor)] += 1

        # Update proposal scale
        newpropsalscale = self.proposalcov[-1].copy()
        np.fill_diagonal(newpropsalscale, np.diag(self.proposalcov) * k)
        self.proposalcov = np.vstack((self.proposalcov, newpropsalscale))

        self.last_scalefactor = k

        # Bookkeeping of updates.
        self.bookkeeping(condupdate.astype(int), action='update')

        """
        if self.verbose:
            print('Scale adjusted for parameter {0};\t\t acceptence rate: '
                  '{1:.2f}%; old scale {2:.2e}; new scale {3:.2e}; '
                  'Value = {4:.3f}'.format(par.label, acceptance_rate * 100,
                                           betamu, betamu * k, par.get_value())
                )
        else:
            print('Scale adjusted for parameter {0};\t\t acceptence rate: '
                  '{1:.2f}%; old scale {2:.6f}; new scale {3:.6f}; '
                  'Value = {4:.3f}'.format(par.label, acceptance_rate*100,
                                           betamu, betamu*k, par.get_value())
                  )
        """
        return

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

        # Add n empty elements to chain.
        self._chain = np.concatenate((self._chain, np.empty(n, self.dim)), axis=0)

        self._lnprob = np.concatenate((self._lnprob, np.empty(n)), axis=0)

        # If fresh run, try to compute lnprob at starting point.
        if self.nlinks == 0:
            try:
                self.last_lnprob = self.get_lnprob(p0)
            except:
                raise AttributeError('First evaluation of target distribution'
                                     'produced and error.')

        for i in xrange(n):

            # Sample
            acceptflag = self.sample()

            if acceptflag:
                self.bookkeeping(self.jump, action='acceptance')

            # Update proposal scales
            self.update_proposal_scale(0.25)

            # Update PCA matrix every nupdatepca iterations.
            if self.nlinks > self.startpca and \
               (self.nlinks - self.startpca) % self.nupdatepca == 0:
                self.pca()

        return

__author__ = 'Rodrigo F. Diaz'
