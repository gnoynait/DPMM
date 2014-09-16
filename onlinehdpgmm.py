"""
online hdp with lazy update
part of code is adapted from Matt's online lda code
"""
import numpy as np
import scipy.special as sp
import os, sys, math, time
import utils
from corpus import document, corpus
from itertools import izip
import random

from scipy.special import digamma as _digamma, gammaln as _gammaln

meanchangethresh = 0.00001
random_seed = 999931111
np.random.seed(random_seed)
random.seed(random_seed)
min_adding_noise_point = 10
min_adding_noise_ratio = 1 
mu0 = 0.3
rhot_bound = 0.0

def digamma(x):
    return _digamma(x + np.finfo(np.float32).eps)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(sp.psi(alpha) - sp.psi(np.sum(alpha)))
    return(sp.psi(alpha) - sp.psi(np.sum(alpha, 1))[:, np.newaxis])

def expect_log_sticks(sticks):
    """
    For stick-breaking hdp, this returns the E[log(sticks)] 
    """
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks 


def _bound_state_log_lik(X, initial_bound, precs, means):
    """Update the bound with likelihood terms, for standard covariance types"""
    n_components, n_features = means.shape
    n_samples = X.shape[0]
    bound = np.empty((n_samples, n_components))
    bound[:] = initial_bound
    for k in range(n_components):
        d = X - means[k]
        bound[:, k] -= 0.5 * np.sum(d * d * precs[k], axis=1)
    return bound

class suff_stats:
    def __init__(self, T, Wt, Dt):
        self.m_batchsize = Dt
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_beta_ss = np.zeros((T, Wt))
        # Wt is the vector's dimension
        self.m_var_mean_ss = np.zeros((T, Wt))
        # TODO: prarameters for update precise
        self.
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)
        self.m_var_mean_ss.fill(0.0)

class online_hdp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, K, D, W, eta, alpha, gamma, kappa, tau, dim = 500, scale=1.0, adding_noise=False):
        """
        this follows the convention of the HDP paper
        gamma: first level concentration
        alpha: second level concentration
        eta: the topic Dirichlet
        T: top level truncation level
        K: second level truncation level
        W: size of vocab
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """

        self.m_W = W # size of vocab
        self.m_D = D # number of doc
        self.m_T = T # Top level truncation
        self.m_K = K # second level truncation
        self.m_alpha = alpha # second level concentration
        self.m_gamma = gamma # first level truncation

        ## each column is the top level topic's beta distribution para
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        #self.m_var_sticks[1] = self.m_gamma
        # make a uniform at beginning
        self.m_var_sticks[1] = range(T-1, 0, -1)

        self.m_varphi_ss = np.zeros(T)

        ## top level topic: topic * word
        self.m_lambda = np.random.gamma(1.0, 1.0, (T, W)) * D*100/(T*W)-eta
        ## topic dirichlet
        self.m_eta = eta
        ## topic * word
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)


        ## start
        self.m_dim = dim # the vector dimension
        self.m_means = np.random.uniform(0.000001, 10, (self.m_T - 1, self.m_dim))
        self.m_dof = np.ones(self.m_T - 1)
        self.m_scale = np.ones(self.m_T - 1)
        self.m_precs = np.ones((self.m_T, self.m_dim))
        self.bound_prec = 0.5 * self.m_T * (digamma(self.m_dof) - np.log(self.m_scale))

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0 
        self.m_status_up_to_date = True
        self.m_adding_noise = adding_noise
        self.m_num_docs_parsed = 0

        # Timestamps and normalizers for lazy updates
        self.m_timestamp = np.zeros(self.m_W, dtype=int)
        self.m_r = [0]
        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)

    def new_init(self, c):
        self.m_lambda = 1.0/self.m_W + 0.01 * np.random.gamma(1.0, 1.0, \
            (self.m_T, self.m_W))
        self.m_Elogbeta = dirichlet_expectation(self.m_eta + self.m_lambda)

        num_samples = min(c.num_docs, burn_in_samples)
        ids = random.sample(range(c.num_docs), num_samples)
        docs = [c.docs[id] for id in ids]

        unique_words = dict()
        word_list = []
        for doc in docs:
            for w in doc.words:
                if w not in unique_words:
                    unique_words[w] = len(unique_words)
                    word_list.append(w)
        Wt = len(word_list) # length of words in these documents

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks
        for doc in docs:
            old_lambda = self.m_lambda[:, word_list].copy()
            for iter in range(5):
                sstats = suff_stats(self.m_T, Wt, 1) 
                doc_score = self.doc_e_step(doc, sstats, Elogsticks_1st, \
                            word_list, unique_words, var_converge=0.0001, max_iter=5)

                self.m_lambda[:, word_list] = old_lambda + sstats.m_var_beta_ss / sstats.m_batchsize
                self.m_Elogbeta = dirichlet_expectation(self.m_lambda)

        self.m_lambda_sum = np.sum(self.m_lambda, axis=1)

    def process_documents(self, docs, var_converge, unseen_ids=[], ropt_o=True):


        # ...and do the lazy updates on the necessary columns of lambda
        rw = np.array([self.m_r[t] for t in self.m_timestamp[word_list]])
        self.m_lambda[:, word_list] *= np.exp(self.m_r[-1] - rw)
        self.m_Elogbeta[:, word_list] = \
            sp.psi(self.m_eta + self.m_lambda[:, word_list]) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])

        ss = suff_stats(self.m_T, Wt, len(docs)) 

        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0

        for i, cop in enumerate(cops):
            cop_score = self.doc_e_step(cop, ss, Elogsticks_1st, var_converge)

        #self.update_lambda(ss, word_list, opt_o)
        self.update_model(ss)
    
        return (score, count, unseen_score, unseen_count) 

    def optimal_ordering(self):
        """
        ordering the topics
        """
        idx = [i for i in reversed(np.argsort(self.m_lambda_sum))]
        self.m_varphi_ss = self.m_varphi_ss[idx]
        self.m_lambda = self.m_lambda[idx,:]
        self.m_lambda_sum = self.m_lambda_sum[idx]
        self.m_Elogbeta = self.m_Elogbeta[idx,:]

    def doc_e_step(self, cop, ss, Elogsticks_1st, \
                   word_list, unique_words, var_converge, \
                   max_iter=100):
        """
        e step for a single doc
        """

        ## Elogbeta_doc = self.m_Elogbeta[:, doc.words] 
        ## very similar to the hdp equations
        v = np.zeros((2, self.m_K-1))  
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        phi = np.ones((cop.shape[0], self.m_K)) * 1.0/self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        
        iter = 0
        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi 
            Eloggauss = expect_log_gauss(cop)
            if iter < 3:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 3:
                phi = np.dot(Eloggauss, var_phi.T)
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            #phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            phi_all = phi[:, np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1
            
        # update the suff_stat ss 
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, batchids] += np.dot(var_phi.T, phi.T * doc.counts)

        return(likelihood)

    def update_lambda(self, sstats, word_list, opt_o): 
        
        self.m_status_up_to_date = False
        if len(word_list) == self.m_W:
          self.m_status_up_to_date = True
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot

        # Update appropriate columns of lambda based on documents.
        self.m_lambda[:, word_list] = self.m_lambda[:, word_list] * (1-rhot) + \
            rhot * self.m_D * sstats.m_var_beta_ss / sstats.m_batchsize
        self.m_lambda_sum = (1-rhot) * self.m_lambda_sum+ \
            rhot * self.m_D * np.sum(sstats.m_var_beta_ss, axis=1) / sstats.m_batchsize

        self.m_updatect += 1
        self.m_timestamp[word_list] = self.m_updatect
        self.m_r.append(self.m_r[-1] + np.log(1-rhot))

        self.m_varphi_ss = (1.0-rhot) * self.m_varphi_ss + rhot * \
               sstats.m_var_sticks_ss * self.m_D / sstats.m_batchsize

        if opt_o:
            self.optimal_ordering();

        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

    def expect_log_gauss(self, X):
        ## TODO some const could be precomputed
        log_lik = np.zeros(X.shape[0], self.m_T)
        log_lik += -0.5 * self.m_T * np.log(2 * np.pi) - np.log(2 * np.pi * np.e)
        log_lik += 0.5 * self.m_T * (digamma(self.m_dof) - np.log(self.m_scale))
        for t in self.m_T:
            cons = 0.5 * self.m_dof[t] / self.m_scale[t]
            d = X - means[t]
            log_lik[:, t] -= cons * (np.sum(d * d) + self.m_dim)
        return log_lik
        
    def score_samples(self, X):
        """Return the likelihood of the data under the model.

        Compute the bound on log probability of X under the model
        and return the posterior distribution (responsibilities) of
        each mixture component for each element of X.

        This is done by computing the parameters for the mean-field of
        z for each observation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities: array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        z = np.zeros((X.shape[0], self.m_T))
        sd = digamma(self.gamma_.T[1] + self.gamma_.T[2])
        dgamma1 = digamma(self.gamma_.T[1]) - sd
        dgamma2 = np.zeros(self.n_components)
        dgamma2[0] = digamma(self.gamma_[0, 2]) - digamma(self.gamma_[0, 1] +
                                                          self.gamma_[0, 2])
        for j in range(1, self.n_components):
            dgamma2[j] = dgamma2[j - 1] + digamma(self.gamma_[j - 1, 2])
            dgamma2[j] -= sd[j - 1]
        dgamma = dgamma1 + dgamma2
        # Free memory and developers cognitive load:
        del dgamma1, dgamma2, sd

        p = _bound_state_log_lik(X, self._initial_bound + self.bound_prec_,
                                 self.precs_, self.means_,
                                 self.covariance_type)
        z = p + dgamma
        z = log_normalize(z, axis=-1)
        bound = np.sum(z * p, axis=-1)
        return bound, z

    def update_expectations(self):
        """
        Since we're doing lazy updates on lambda, at any given moment
        the current state of lambda may not be accurate. This function
        updates all of the elements of lambda and Elogbeta so that if (for
        example) we want to print out the topics we've learned we'll get the
        correct behavior.
        """
        for w in range(self.m_W):
            self.m_lambda[:, w] *= np.exp(self.m_r[-1] - 
                                          self.m_r[self.m_timestamp[w]])
        self.m_Elogbeta = sp.psi(self.m_eta + self.m_lambda) - \
            sp.psi(self.m_W*self.m_eta + self.m_lambda_sum[:, np.newaxis])
        self.m_timestamp[:] = self.m_updatect
        self.m_status_up_to_date = True

    def update_means(self, X, z):
        """Update the variational distributions for the means"""
        n_features = X.shape[1]
        for k in range(self.n_components):
            num = np.sum(z.T[k].reshape((-1, 1)) * X, axis=0)
            num *= self.precs_[k]
            den = 1. + self.precs_[k] * np.sum(z.T[k])
            self.means_[k] = num / den

    def update_precisions(self, X, z):
        """Update the variational distributions for the precisions"""
        n_features = X.shape[1]
        self.dof_ = 0.5 * n_features * np.sum(z, axis=0)
        for k in range(self.n_components):
            # could be more memory efficient ?
            sq_diff = np.sum((X - self.means_[k]) ** 2, axis=1)
            self.scale_[k] = 1.
            self.scale_[k] += 0.5 * np.sum(z.T[k] * (sq_diff + n_features))
            self.bound_prec_[k] = (
                0.5 * n_features * (
                    digamma(self.dof_[k]) - np.log(self.scale_[k])))
        self.precs_ = np.tile(self.dof_ / self.scale_, [n_features, 1]).T


    def save_topics(self, filename):
        if not self.m_status_up_to_date:
            self.update_expectations()
        f = file(filename, "w") 
        betas = self.m_lambda + self.m_eta
        for beta in betas:
            line = ' '.join([str(x) for x in beta])  
            f.write(line + '\n')
        f.close()

    def infer_only(self, docs, half_train_half_test=False, split_ratio=0.9, iterative_average=False):
        # be sure to run update_expectations()
        sticks = self.m_var_sticks[0]/(self.m_var_sticks[0]+self.m_var_sticks[1])
        alpha = np.zeros(self.m_T)
        left = 1.0
        for i in range(0, self.m_T-1):
            alpha[i] = sticks[i] * left
            left = left - alpha[i]
        alpha[self.m_T-1] = left      
        #alpha = alpha * self.m_gamma
        score = 0.0
        count = 0.0
        for doc in docs:
            if half_train_half_test:
                (s, c, gamma) = lda_e_step_half(doc, alpha, self.m_Elogbeta, split_ratio) 
                score += s
                count += c
            else:
                score += lda_e_step(doc, alpha, np.exp(self.m_Elogbeta))
                count += doc.total
        return (score, count)
