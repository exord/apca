from distutils.core import setup

setup(name='cobmcmc',
      description='Change-of-Basis, a flexible Metropolis-Hastings MCMC algorithm.',
      version='0.1.0',
      author='Rodrigo F. Diaz',
      author_email='rodrigo.diaz@unige.ch',
      url='',
      packages=['cobmcmc'],
      requires=['numpy', 'scipy']
      )
