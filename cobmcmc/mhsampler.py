from math import log

import numpy as np
import numpy.random

from cobmcmc.sampler import Sampler

__author__ = 'Rodrigo F. Diaz'


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

    def post_proposal(self, proposal):
        return proposal

    def sample(self):
        # Compute ln probability for initial step
        lnprob0 = self.last_lnprob

        # Propose a new chain state
        proposal = self.make_proposal()

        # Here, add implementation of post-jump function to deal with
        # circular parameters such as omega or L0
        proposal = self.post_proposal(proposal)

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
