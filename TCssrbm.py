"""

This file extends the mu-ssRBM for tiled-convolutional training

"""
import cPickle, pickle
import numpy
numpy.seterr('warn') #SHOULD NOT BE IN LIBIMPORT
from PIL import Image
import theano
from theano import tensor
from theano.tensor import nnet,grad

import sys
from unshared_conv_diagonally import FilterActs
from unshared_conv_diagonally import WeightActs
from unshared_conv_diagonally import ImgActs
from Brodatz import Brodatz_op

#import scipy.io
import os
_temp_data_path_ = '.'#'/Tmp/luoheng'

if 1:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg
    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


floatX=theano.config.floatX
sharedX = lambda X, name : theano.shared(numpy.asarray(X, dtype=floatX),
        name=name)

def Toncv(image,filters,module_stride=1):
    op = FilterActs(module_stride)
    return op(image,filters)
    
def Tdeconv(filters, hidacts, irows, icols, module_stride=1):
    op = ImgActs(module_stride)
    return op(filters, hidacts, irows, icols)

def contrastive_cost(free_energy_fn, pos_v, neg_v):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: TensorType matrix MxN of M "positive phase" particles
    :param neg_v: TensorType matrix MxN of M "negative phase" particles

    :returns: TensorType scalar that's the sum of the difference of free energies

    :math: \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])

    """
    return (free_energy_fn(pos_v) - free_energy_fn(neg_v)).sum()


def contrastive_grad(free_energy_fn, pos_v, neg_v, wrt, other_cost=0, consider_constant=[]):
    """
    :param free_energy_fn: lambda (TensorType matrix MxN) ->  TensorType vector of M free energies
    :param pos_v: positive-phase sample of visible units
    :param neg_v: negative-phase sample of visible units
    :param wrt: TensorType variables with respect to which we want gradients (similar to the
        'wrt' argument to tensor.grad)
    :param other_cost: TensorType scalar (should be the sum over a minibatch, not mean)

    :returns: TensorType variables for the gradient on each of the 'wrt' arguments


    :math: Cost = other_cost + \sum_i free_energy(pos_v[i]) - free_energy(neg_v[i])
    :math: d Cost / dW for W in `wrt`


    This function is similar to tensor.grad - it returns the gradient[s] on a cost with respect
    to one or more parameters.  The difference between tensor.grad and this function is that
    the negative phase term (`neg_v`) is considered constant, i.e. d `Cost` / d `neg_v` = 0.
    This is desirable because `neg_v` might be the result of a sampling expression involving
    some of the parameters, but the contrastive divergence algorithm does not call for
    backpropagating through the sampling procedure.

    Warning - if other_cost depends on pos_v or neg_v and you *do* want to backpropagate from
    the `other_cost` through those terms, then this function is inappropriate.  In that case,
    you should call tensor.grad separately for the other_cost and add the gradient expressions
    you get from ``contrastive_grad(..., other_cost=0)``

    """
    cost=contrastive_cost(free_energy_fn, pos_v, neg_v)
    if other_cost:
        cost = cost + other_cost
    return theano.tensor.grad(cost,
            wrt=wrt,
            consider_constant=consider_constant+[neg_v])



def unnatural_sgd_updates(params, grads, stepsizes, tracking_coef=0.1, epsilon=1):
    grad_means = [theano.shared(numpy.zeros_like(p.get_value(borrow=True)))
            for p in params]
    grad_means_sqr = [theano.shared(numpy.ones_like(p.get_value(borrow=True)))
            for p in params]
    updates = dict()
    for g, gm, gms, p, s in zip(
            grads, grad_means, grad_means_sqr, params, stepsizes):
        updates[gm] = tracking_coef * g + (1-tracking_coef) * gm
        updates[gms] = tracking_coef * g*g + (1-tracking_coef) * gms

        var_g = gms - gm**2
        # natural grad doesn't want sqrt, but i found it worked worse
        updates[p] = p - s * gm / tensor.sqrt(var_g+epsilon)
    return updates


def safe_update(a, b):
    for k,v in dict(b).iteritems():
        if k in a:
            raise KeyError(k)
        a[k] = v
    return a
    
def most_square_shape(N):
    """rectangle (height, width) with area N that is closest to sqaure
    """
    for i in xrange(int(numpy.sqrt(N)),0, -1):
        if 0 == N % i:
            return (i, N/i)


def tile_conv_weights(w,flip=False, scale_each=False):
    """
    Return something that can be rendered as an image to visualize the filters.
    """
    #if w.shape[1] != 3:
    #    raise NotImplementedError('not rgb', w.shape)
    if w.shape[2] != w.shape[3]:
        raise NotImplementedError('not square', w.shape)

    if w.shape[1] == 1:
	wmin, wmax = w.min(), w.max()
    	if not scale_each:
            w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    	trows, tcols= most_square_shape(w.shape[0])
    	outrows = trows * w.shape[2] + trows-1
    	outcols = tcols * w.shape[3] + tcols-1
    	out = numpy.zeros((outrows, outcols), dtype='uint8')
    	#tr_stride= 1+w.shape[1]
    	for tr in range(trows):
            for tc in range(tcols):
            	# this is supposed to flip the filters back into the image
            	# coordinates as well as put the channels in the right place, but I
            	# don't know if it really does that
            	tmp = w[tr*tcols+tc,
			     0,
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            	if scale_each:
                    tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            	out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    	return out

    wmin, wmax = w.min(), w.max()
    if not scale_each:
        w = numpy.asarray(255 * (w - wmin) / (wmax - wmin + 1e-6), dtype='uint8')
    trows, tcols= most_square_shape(w.shape[0])
    outrows = trows * w.shape[2] + trows-1
    outcols = tcols * w.shape[3] + tcols-1
    out = numpy.zeros((outrows, outcols,3), dtype='uint8')

    tr_stride= 1+w.shape[1]
    for tr in range(trows):
        for tc in range(tcols):
            # this is supposed to flip the filters back into the image
            # coordinates as well as put the channels in the right place, but I
            # don't know if it really does that
            tmp = w[tr*tcols+tc].transpose(1,2,0)[
                             ::-1 if flip else 1,
                             ::-1 if flip else 1]
            if scale_each:
                tmp = numpy.asarray(255*(tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-6),
                        dtype='uint8')
            out[tr*(1+w.shape[2]):tr*(1+w.shape[2])+w.shape[2],
                    tc*(1+w.shape[3]):tc*(1+w.shape[3])+w.shape[3]] = tmp
    return out

class RBM(object):
    """
    Light-weight class that provides math related to inference in Spike & Slab RBM

    Attributes:
     - v_prec - the base conditional precisions of data units [shape (n_img_rows, n_img_cols,)]
     - v_shape - the input image shape  (ie. n_imgs, n_chnls, n_img_rows, n_img_cols)

     - n_conv_hs - the number of spike and slab hidden units
     - filters_hs_shape - the kernel filterbank shape for hs units
     - filters_h_shape -  the kernel filterbank shape for h units
     - filters_hs - a tensor with shape (n_conv_hs,n_chnls,n_ker_rows, n_ker_cols)
     - conv_bias_hs - a vector with shape (n_conv_hs, n_out_rows, n_out_cols)
     - subsample_hs - how to space the receptive fields (dx,dy)

     - n_global_hs - how many globally-connected spike and slab units
     - weights_hs - global weights
     - global_bias_hs -

     - _params a list of the attributes that are shared vars


    The technique of combining convolutional and global filters to account for border effects is
    borrowed from  (Alex Krizhevsky, TR?, October 2010).
    """
    def __init__(self, **kwargs):
        print 'init rbm'
	self.__dict__.update(kwargs)

    @classmethod
    def alloc(cls,
            conf,
            image_shape,  # input dimensionality
            filters_hs_shape,       
            filters_irange,
            v_prec,
            v_prec_lower_limit, #should be parameter of the training algo            
            seed = 8923402            
            ):
 	print 'alloc rbm'
        rng = numpy.random.RandomState(seed)

        self = cls()
       
	n_images, n_channels, n_img_rows, n_img_cols = image_shape
        n_filters_hs_modules, n_filters_hs_per_modules, fcolors, n_filters_hs_rows, n_filters_hs_cols = filters_hs_shape        
        assert fcolors == n_channels        
        self.v_shape = image_shape
        print 'v_shape'
	print self.v_shape
	self.filters_hs_shape = filters_hs_shape
        print 'self.filters_hs_shape'
        print self.filters_hs_shape
        self.out_conv_hs_shape = FilterActs.infer_shape_without_instance(self.v_shape,self.filters_hs_shape)        
        print 'self.out_conv_hs_shape'
        print self.out_conv_hs_shape
        conv_bias_hs_shape = self.out_conv_hs_shape[1:]
        self.conv_bias_hs_shape = conv_bias_hs_shape
        print 'self.conv_bias_hs_shape'
        print self.conv_bias_hs_shape
        self.v_prec = sharedX(numpy.zeros((n_channels, n_img_rows, n_img_cols))+v_prec, 'var_v_prec')
        self.v_prec_lower_limit = sharedX(v_prec_lower_limit, 'v_prec_lower_limit')
        #a = self.v_prec.broadcastable
        #b = self.v_prec_lower_limit.broadcastable
        #print a,b
        
        self.filters_hs = sharedX(rng.randn(*filters_hs_shape) * filters_irange , 'filters_hs')  
        #a = self.filters_hs.broadcastable
        #print a

        conv_bias_ival = rng.rand(*conv_bias_hs_shape)*2-1
        conv_bias_ival *= conf['conv_bias_irange']
        conv_bias_ival += conf['conv_bias0']
	self.conv_bias_hs = sharedX(conv_bias_ival, name='conv_bias_hs')
                
        conv_mu_ival = numpy.zeros(conv_bias_hs_shape,dtype=floatX) + conf['conv_mu0']
	self.conv_mu = sharedX(conv_mu_ival, 'conv_mu')
        
	if conf['alpha_logdomain']:
            conv_alpha_ival = numpy.zeros(conv_bias_hs_shape,dtype=floatX) + numpy.log(conf['conv_alpha0'])
	    self.conv_alpha = sharedX(conv_alpha_ival,'conv_alpha')
	else:
            self.conv_alpha = sharedX(
                    numpy.zeros(conv_bias_hs_shape)+conf['conv_alpha0'],
                    'conv_alpha')
 
        if conf['lambda_logdomain']:
            self.conv_lambda = sharedX(
                    numpy.zeros(self.filters_hs_shape)
                        + numpy.log(conf['lambda0']),
                    name='conv_lambda')
        else:
            self.conv_lambda = sharedX(
                    numpy.zeros(self.filters_hs_shape)
                        + (conf['lambda0']),
                    name='conv_lambda')

        negsample_mask = numpy.zeros((n_channels,n_img_rows,n_img_cols),dtype=floatX)
 	negsample_mask[:,n_filters_hs_rows:n_img_rows-n_filters_hs_rows+1,n_filters_hs_cols:n_img_cols-n_filters_hs_cols+1] = 1
	self.negsample_mask = sharedX(negsample_mask,'negsample_mask')                
        
        self.conf = conf
        self._params = [self.v_prec,
                self.filters_hs,
                self.conv_bias_hs,
                self.conv_mu, 
                self.conv_alpha,
                self.conv_lambda
                ]
        return self

    def get_conv_alpha(self):
        if self.conf['alpha_logdomain']:
            rval = tensor.exp(self.conv_alpha)
	    return rval
        else:
            return self.conv_alpha
    def get_conv_lambda(self):
        if self.conf["lambda_logdomain"]:
            L = tensor.exp(self.conv_lambda)
        else:
            L = self.conv_lambda
        return L
    def conv_problem_term(self, v):
        L = self.get_conv_lambda()
        W = self.filters_hs
        alpha = self.get_conv_alpha()
        vLv = self.convdot(v*v, L)        
        rval = vLv
        return rval
    def conv_problem_term_T(self, h):
        L = self.get_conv_lambda()
        #W = self.filters_hs
        #alpha = self.get_conv_alpha()
        hL = self.convdot_T(L, h)        
        rval = hL
        return rval
    def convdot(self, image, filters):
        return Toncv(image,filters)
        
    def convdot_T(self, filters, hidacts):
        n_images, n_channels, n_img_rows, n_img_cols = self.v_shape
        return Tdeconv(filters, hidacts, n_img_rows, n_img_cols)         

    #####################
    # spike-and-slab convolutional hidden units
    def mean_convhs_h_given_v(self, v):
        """Return the mean of binary-valued hidden units h, given v
        """
        alpha = self.get_conv_alpha()
        W = self.filters_hs
        vW = self.convdot(v, W)
	rval = nnet.sigmoid(
                tensor.add(
                    self.conv_bias_hs,
                    -0.5*self.conv_problem_term(v),
                    self.conv_mu * vW,
                    0.5 * (vW**2)/ alpha))
        return rval

    def mean_var_convhs_s_given_v(self, v):
        """
        Return mu (N,K,B) and sigma (N,K,K) for latent s variable.

        For efficiency, this method assumes all h variables are 1.

        """
        alpha = self.get_conv_alpha()
        vW = self.convdot(v, self.filters_hs)
        rval = self.conv_mu + vW/alpha        
        return rval, 1.0 / alpha

    #####################
    # visible units
    def mean_var_v_given_h_s(self, convhs_h, convhs_s):
        shF = self.convdot_T(self.filters_hs, convhs_h*convhs_s)
        #bbb = convhs_h*convhs_s
        #broadcastable_value = bbb.broadcastable
        #print broadcastable_value
        conv_hL = self.conv_problem_term_T(convhs_h)
        contrib = shF               
        sigma_sq = 1.0 / (self.v_prec + conv_hL)
        mu = contrib * sigma_sq
        
        return mu, sigma_sq


    def all_hidden_h_means_given_v(self, v):
        mean_convhs_h = self.mean_convhs_h_given_v(v)
        return mean_convhs_h

    #####################

    def gibbs_step_for_v(self, v, s_rng, return_locals=False):
        #positive phase

        # spike variable means
        mean_convhs_h = self.all_hidden_h_means_given_v(v)
        #broadcastable_value = mean_convhs_h.broadcastable
        #print broadcastable_value
        
        # slab variable means
        meanvar_convhs_s = self.mean_var_convhs_s_given_v(v)
        #smean, svar = meanvar_convhs_s
        #broadcastable_value = smean.broadcastable
        #print broadcastable_value
        #broadcastable_value = svar.broadcastable
        #print broadcastable_value
        
        # spike variable samples
        def sample_h(hmean,shp):
            return tensor.cast(s_rng.uniform(size=shp) < hmean, floatX)
        def sample_s(smeanvar, shp):
            smean, svar = smeanvar
            return s_rng.normal(size=shp)*tensor.sqrt(svar) + smean

        sample_convhs_h = sample_h(mean_convhs_h, self.out_conv_hs_shape)
        
        # slab variable samples
        sample_convhs_s = sample_s(meanvar_convhs_s, self.out_conv_hs_shape)
	
        #negative phase
        vv_mean, vv_var = self.mean_var_v_given_h_s(
                sample_convhs_h, sample_convhs_s,
                )
        vv_sample = s_rng.normal(size=self.v_shape) * tensor.sqrt(vv_var) + vv_mean
        vv_sample = theano.tensor.mul(vv_sample,self.negsample_mask)
        #broadcastable_value = vv_mean.broadcastable
        #print broadcastable_value
       
	if return_locals:
            return vv_sample, locals()
        else:
            return vv_sample

    def free_energy_given_v(self, v):
        # This is accurate up to a multiplicative constant
        # because I dropped some terms involving 2pi
        def pre_sigmoid(x):
            assert x.owner and x.owner.op == nnet.sigmoid
            return x.owner.inputs[0]

        pre_convhs_h = pre_sigmoid(self.mean_convhs_h_given_v(v))
        rval = tensor.add(
                -tensor.sum(nnet.softplus(pre_convhs_h),axis=[1,2,3,4]), #the shape of pre_convhs_h: 64 x 11 x 32 x 8 x 8
                0.5 * tensor.sum(self.v_prec * (v**2), axis=[1,2,3]), #shape: 64 x 1 x 98 x 98 
                )
        assert rval.ndim==1
        return rval

    def cd_updates(self, pos_v, neg_v, stepsizes, other_cost=None):
        grads = contrastive_grad(self.free_energy_given_v,
                pos_v, neg_v,
                wrt=self.params(),
                other_cost=other_cost
                ) 
        assert len(stepsizes)==len(grads)

        if self.conf['unnatural_grad']:
            sgd_updates = unnatural_sgd_updates
        else:
            #sgd_updates = pylearn.gd.sgd.sgd_updates
            pass
        #self.conv_bias_hs = self.conv_bias_hs+self.h_tiled_conv_mask
        rval = dict(
                sgd_updates(
                    self.params(),
                    grads,
                    stepsizes=stepsizes))
        if 0:
            #DEBUG STORE GRADS
            grad_shared_vars = [sharedX(0*p.value.copy(),'') for p in self.params()]
            self.grad_shared_vars = grad_shared_vars
            rval.update(dict(zip(grad_shared_vars, grads)))
       
	return rval

    def params(self):
        # return the list of *shared* learnable parameters
        # that are, in your judgement, typically learned in this model
        return list(self._params)

    def save_weights_to_files(self, identifier):
        # save 4 sets of weights:
        pass
    def save_weights_to_grey_files(self, identifier):
        # save 4 sets of weights:

        #filters_hs
        def arrange_for_show(filters_hs,filters_hs_shape):
	    n_filters_hs_modules, n_filters_hs_per_modules, fcolors, n_filters_hs_rows, n_filters_hs_cols  = filters_hs_shape            
            filters_fs_for_show = filters_hs.reshape(
                       (n_filters_hs_modules*n_filters_hs_per_modules, 
                       fcolors,
                       n_filters_hs_rows,
                       n_filters_hs_cols))
            fn = theano.function([],filters_fs_for_show)
            rval = fn()
            return rval
        filters_fs_for_show = arrange_for_show(self.filters_hs, self.filters_hs_shape)
        Image.fromarray(
                       tile_conv_weights(
                       filters_fs_for_show,flip=False), 'L').save(
                'filters_hs_%s.png'%identifier)
   
        if self.conf['lambda_logdomain']:
            raise NotImplementedError()
        else:
	    conv_lambda_for_show = arrange_for_show(self.conv_lambda, self.filters_hs_shape) 	    
	    Image.fromarray(
                            tile_conv_weights(
                            conv_lambda_for_show,flip=False), 'L').save(
                    'conv_lambda_%s.png'%identifier)
     
    def dump_to_file(self, filename):
        try:
            cPickle.dump(self, open(filename, 'wb'))
        except cPickle.PicklingError:
            pickle.dump(self, open(filename, 'wb'))


class Gibbs(object): # if there's a Sampler interface - this should support it
    @classmethod
    def alloc(cls, rbm, batchsize, rng):
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        self = cls()
        seed=int(rng.randint(2**30))
        self.rbm = rbm
	if batchsize==rbm.v_shape[0]:
	    self.particles = sharedX(
            rng.randn(*rbm.v_shape),
            name='particles')
	else:
	    self.particles = sharedX(
            rng.randn(batchsize,1,98,98),
            name='particles')
        self.s_rng = RandomStreams(seed)
        return self

def HMC(rbm, batchsize, rng): # if there's a Sampler interface - this should support it
    if not hasattr(rng, 'randn'):
        rng = numpy.random.RandomState(rng)
    seed=int(rng.randint(2**30))
    particles = sharedX(
            rng.randn(*rbm.v_shape),
            name='particles')
    return pylearn.sampling.hmc.HMC_sampler(
            particles,
            rbm.free_energy_given_v,
            seed=seed)


class Trainer(object): # updates of this object implement training
    @classmethod
    def alloc(cls, rbm, visible_batch,
            lrdict,
            conf,
            rng=234,
            iteration_value=0,
            ):

        batchsize = rbm.v_shape[0]
        sampler = Gibbs.alloc(rbm, batchsize, rng=rng)
	print 'alloc trainer'
        error = 0.0
        return cls(
                rbm=rbm,
                batchsize=batchsize,
                visible_batch=visible_batch,
                sampler=sampler,
                iteration=sharedX(iteration_value, 'iter'),
                learn_rates = [lrdict[p] for p in rbm.params()],
                conf=conf,
                annealing_coef=sharedX(1.0, 'annealing_coef'),
                conv_h_means = sharedX(numpy.zeros(rbm.conv_bias_hs_shape)+0.5,'conv_h_means'),
                recons_error = sharedX(error,'reconstruction_error')
                )

    def __init__(self, **kwargs):
        print 'init trainer'
	self.__dict__.update(kwargs)

    def updates(self):
        
        print 'start trainer.updates'
	conf = self.conf
        ups = {}
        add_updates = lambda b: safe_update(ups,b)

        annealing_coef = 1.0 - self.iteration / float(conf['train_iters'])
        ups[self.iteration] = self.iteration + 1 #
        ups[self.annealing_coef] = annealing_coef

        conv_h = self.rbm.all_hidden_h_means_given_v(
                self.visible_batch)
        
        
        new_conv_h_means = 0.1 * conv_h.mean(axis=0) + .9*self.conv_h_means
        ups[self.conv_h_means] = new_conv_h_means
        #ups[self.global_h_means] = new_global_h_means


        sparsity_cost = 0
        self.sparsity_cost = sparsity_cost
        # SML updates PCD
        add_updates(
                self.rbm.cd_updates(
                    pos_v=self.visible_batch,
                    neg_v=self.sampler.particles,
                    stepsizes=[annealing_coef*lr for lr in self.learn_rates],
                    other_cost=sparsity_cost))
        
        if conf['chain_reset_prob']:
            # advance the 'negative-phase' chain
            resets = self.sampler.s_rng.uniform(size=(conf['batchsize'],))<conf['chain_reset_prob']
            old_particles = tensor.switch(resets.dimshuffle(0,'x','x','x'),
                    self.visible_batch,   # reset the chain
                    self.sampler.particles,  #continue chain
                    )
        else:
            old_particles = self.sampler.particles
        new_particles  = self.rbm.gibbs_step_for_v(old_particles, self.sampler.s_rng)
        #broadcastable_value = new_particles.broadcastable
        #print broadcastable_value
        #reconstructions= self.rbm.gibbs_step_for_v(self.visible_batch, self.sampler.s_rng)
	#recons_error   = tensor.sum((self.visible_batch-reconstructions)**2)
	recons_error = 0
        ups[self.recons_error] = recons_error
	#return {self.particles: new_particles}
        ups[self.sampler.particles] = tensor.clip(new_particles,
                conf['particles_min'],
                conf['particles_max'])
        
        # make sure that the new v_precision doesn't top below its floor
        new_v_prec = ups[self.rbm.v_prec]
        ups[self.rbm.v_prec] = tensor.switch(
                new_v_prec<self.rbm.v_prec_lower_limit,
                self.rbm.v_prec_lower_limit,
                new_v_prec)
        """
        # make sure that the interior region of global weights matrix is properly masked
        if self.conf['zero_out_interior_weights']:
            ups[self.rbm.weights_hs] = self.rbm.weights_mask * ups[self.rbm.weights_hs]
        """
        if self.conf['alpha_min'] < self.conf['alpha_max']:
            if self.conf['alpha_logdomain']:
                ups[self.rbm.conv_alpha] = tensor.clip(
                        ups[self.rbm.conv_alpha],
                        numpy.log(self.conf['alpha_min']).astype(floatX),
                        numpy.log(self.conf['alpha_max']).astype(floatX))
                #ups[self.rbm.global_alpha] = tensor.clip(
                #        ups[self.rbm.global_alpha],
                #        numpy.log(self.conf['alpha_min']).astype(floatX),
                #        numpy.log(self.conf['alpha_max']).astype(floatX))
            else:
                ups[self.rbm.conv_alpha] = tensor.clip(
                        ups[self.rbm.conv_alpha],
                        self.conf['alpha_min'],
                        self.conf['alpha_max'])
                #ups[self.rbm.global_alpha] = tensor.clip(
                #        ups[self.rbm.global_alpha],
                #        self.conf['alpha_min'],
                #        self.conf['alpha_max'])
        if self.conf['lambda_min'] < self.conf['lambda_max']:
            if self.conf['lambda_logdomain']:
                ups[self.rbm.conv_lambda] = tensor.clip(ups[self.rbm.conv_lambda],
                        numpy.log(self.conf['lambda_min']).astype(floatX),
                        numpy.log(self.conf['lambda_max']).astype(floatX))
                #ups[self.rbm.global_lambda] = tensor.clip(ups[self.rbm.global_lambda],
                #        numpy.log(self.conf['lambda_min']).astype(floatX),
                #        numpy.log(self.conf['lambda_max']).astype(floatX))
            else:
                ups[self.rbm.conv_lambda] = tensor.clip(ups[self.rbm.conv_lambda],
                        self.conf['lambda_min'],
                        self.conf['lambda_max'])
                #ups[self.rbm.global_lambda] = tensor.clip(ups[self.rbm.global_lambda],
                #        self.conf['lambda_min'],
                #        self.conf['lambda_max'])
        #ups[self.rbm.conv_bias_hs] = self.rbm.conv_bias_hs.get_value(borrow=True)+self.rbm.h_tiled_conv_mask        
        return ups

    def save_weights_to_files(self, pattern='iter_%05i'):
        #pattern = pattern%self.iteration.get_value()

        # save particles
        #Image.fromarray(tile_conv_weights(self.sampler.particles.get_value(borrow=True),
        #    flip=False),
        #        'RGB').save('particles_%s.png'%pattern)
        #self.rbm.save_weights_to_files(pattern)
        pass

    def save_weights_to_grey_files(self, pattern='iter_%05i'):
        pattern = pattern%self.iteration.get_value()

        # save particles
        """
        particles_for_show = self.sampler.particles.dimshuffle(3,0,1,2)
        fn = theano.function([],particles_for_show)
        particles_for_show_value = fn()
        Image.fromarray(tile_conv_weights(particles_for_show_value,
            flip=False),'L').save('particles_%s.png'%pattern)
        self.rbm.save_weights_to_grey_files(pattern)
        """
        Image.fromarray(tile_conv_weights(self.sampler.particles.get_value(borrow=True),
            flip=False),'L').save('particles_%s.png'%pattern)
        self.rbm.save_weights_to_grey_files(pattern)
    def print_status(self):
        def print_minmax(msg, x):
            assert numpy.all(numpy.isfinite(x))
            print msg, x.min(), x.max()

        print 'iter:', self.iteration.get_value()
        print_minmax('filters_hs ', self.rbm.filters_hs.get_value(borrow=True))
        print_minmax('conv_bias_hs', self.rbm.conv_bias_hs.get_value(borrow=True))
        #print_minmax('weights_hs ', self.rbm.weights_hs.get_value(borrow=True))
        #print_minmax('global_bias_hs', self.rbm.global_bias_hs.get_value(borrow=True))
        print_minmax('conv_mu', self.rbm.conv_mu.get_value(borrow=True))
        #print_minmax('global_mu', self.rbm.global_mu.get_value(borrow=True))
        if self.conf['alpha_logdomain']:
            print_minmax('conv_alpha',
                    numpy.exp(self.rbm.conv_alpha.get_value(borrow=True)))
            #print_minmax('global_alpha',
            #        numpy.exp(self.rbm.global_alpha.get_value(borrow=True)))
        else:
            print_minmax('conv_alpha', self.rbm.conv_alpha.get_value(borrow=True))
            #print_minmax('global_alpha', self.rbm.global_alpha.get_value(borrow=True))
        if self.conf['lambda_logdomain']:
            print_minmax('conv_lambda',
                    numpy.exp(self.rbm.conv_lambda.get_value(borrow=True)))
            #print_minmax('global_lambda',
            #        numpy.exp(self.rbm.global_lambda.get_value(borrow=True)))
        else:
            print_minmax('conv_lambda', self.rbm.conv_lambda.get_value(borrow=True))
            #print_minmax('global_lambda', self.rbm.global_lambda.get_value(borrow=True))
        print_minmax('v_prec', self.rbm.v_prec.get_value(borrow=True))
        print_minmax('particles', self.sampler.particles.get_value())
        print_minmax('conv_h_means', self.conv_h_means.get_value())
        #print self.conv_h_means.get_value()[0,0:11,0:11]
	#print self.rbm.conv_bias_hs.get_value(borrow=True)[0,0,0:3,0:3]
        #print self.rbm.h_tiled_conv_mask.get_value(borrow=True)[0,32,0:3,0:3]
	#print_minmax('global_h_means', self.global_h_means.get_value())
        print 'lr annealing coef:', self.annealing_coef.get_value()
	print 'reconstruction error:', self.recons_error.get_value()

def main_inpaint(filename, algo='Gibbs', rng=777888):
    rbm = cPickle.load(open(filename))
    sampler = Gibbs.alloc(rbm, rbm.conf['batchsize'], rng)

    mat = scipy.io.loadmat('../Brodatz')
    #mat = scipy.io.loadmat('../Brodatz_D103_smallSTD')
    batchdata = mat['batchdata']
    batchdata = numpy.cast['float32'](batchdata)
    batchdata = batchdata[0:10,:,:,:]
    batchdata[:,:,11:88,11:88] = 0
    shared_batchdata = sharedX(batchdata,name='data')

    border_mask = numpy.zeros((10,1,98,98),dtype=floatX)
    border_mask[:,:,11:88,11:88]=1

    sampler.particles = shared_batchdata
    new_particles = rbm.gibbs_step_for_v(sampler.particles, sampler.s_rng)
    new_particles = tensor.mul(new_particles,border_mask)
    new_particles = tensor.add(new_particles,batchdata)
    fn = theano.function([], [],
                updates={sampler.particles: new_particles})
    particles = sampler.particles


    for i in xrange(200):
        print i
        if i % 10 == 0:
            savename = '%s_inpaint_%04i.png'%(filename,i)
            print 'saving'
            Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True),
                    flip=False),
                'L').save(savename)
        fn()

def main_sample(filename, algo='Gibbs', rng=777888, burn_in=500, save_interval=500, n_files=1):
    rbm = cPickle.load(open(filename))
    if algo == 'Gibbs':
        sampler = Gibbs.alloc(rbm, rbm.conf['batchsize'], rng)
        new_particles  = rbm.gibbs_step_for_v(sampler.particles, sampler.s_rng)
        new_particles = tensor.clip(new_particles,
                rbm.conf['particles_min'],
                rbm.conf['particles_max'])
        fn = theano.function([], [],
                updates={sampler.particles: new_particles})
        particles = sampler.particles
    elif algo == 'HMC':
        print "WARNING THIS PROBABLY DOESNT WORK"
        # still need to figure out how to get the clipping into
        # the iterations of mcmc
        sampler = HMC(rbm, rbm.conf['batchsize'], rng)
        ups = sampler.updates()
        ups[sampler.positions] = tensor.clip(ups[sampler.positions],
                rbm.conf['particles_min'],
                rbm.conf['particles_max'])
        fn = theano.function([], [], updates=ups)
        particles = sampler.positions

    for i in xrange(burn_in):
	print i
	if i % 10 == 0:
            savename = '%s_sample_burn_%04i.png'%(filename,i)
	    print 'saving'
	    Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True),
                    flip=False),
                'L').save(savename)	
        fn()

    for n in xrange(n_files):
        for i in xrange(save_interval):
            fn()
        savename = '%s_sample_%04i.png'%(filename,n)
        print 'saving', savename
        Image.fromarray(
                tile_conv_weights(
                    particles.get_value(borrow=True),
                    flip=False),
                'L').save(savename)
                
def main_print_status(filename, algo='Gibbs', rng=777888, burn_in=500, save_interval=500, n_files=1):
    def print_minmax(msg, x):
        assert numpy.all(numpy.isfinite(x))
        print msg, x.min(), x.max()
    rbm = cPickle.load(open(filename))
    if algo == 'Gibbs':
        sampler = Gibbs.alloc(rbm, rbm.conf['batchsize'], rng)
        new_particles  = rbm.gibbs_step_for_v(sampler.particles, sampler.s_rng)
        #new_particles = tensor.clip(new_particles,
        #        rbm.conf['particles_min'],
        #        rbm.conf['particles_max'])
        fn = theano.function([], [],
                updates={sampler.particles: new_particles})
        particles = sampler.particles
    elif algo == 'HMC':
        print "WARNING THIS PROBABLY DOESNT WORK"
     
    for i in xrange(burn_in):
	fn()
        print_minmax('particles', particles.get_value(borrow=True))              
                
                
def main0(rval_doc):
    if 'conf' not in rval_doc:
        raise NotImplementedError()

    conf = rval_doc['conf']
    batchsize = conf['batchsize']

    batch_idx = tensor.iscalar()
    batch_range = batch_idx * conf['batchsize'] + numpy.arange(conf['batchsize'])
    
    
       
    if conf['dataset']=='Brodatz':
        n_examples = conf['batchsize']   #64
        n_img_rows = 98
        n_img_cols = 98
        n_img_channels=1
  	batch_x = Brodatz_op(batch_range,
  	                     '../Brodatz/D6.gif',   # download from http://www.ux.uis.no/~tranden/brodatz.html
  	                     patch_shape=(n_img_channels,
  	                                 n_img_rows,
  	                                 n_img_cols), 
  	                     noise_concelling=100., 
  	                     seed=3322, 
  	                     batchdata_size=n_examples
  	                     )	
    else:
        raise ValueError('dataset', conf['dataset'])
     
       
    rbm = RBM.alloc(
            conf,
            image_shape=(        
                n_examples,
                n_img_channels,
                n_img_rows,
                n_img_cols
                ),
            filters_hs_shape=(
                conf['filters_hs_size'],  
                conf['n_filters_hs'],
                n_img_channels,
                conf['filters_hs_size'],
                conf['filters_hs_size']
                ),            #fmodules(stride) x filters_per_modules x fcolors(channels) x frows x fcols
            filters_irange=conf['filters_irange'],
            v_prec=conf['v_prec_init'],
            v_prec_lower_limit=conf['v_prec_init'],            
            )

    rbm.save_weights_to_grey_files('iter_0000')

    base_lr = conf['base_lr_per_example']/batchsize
    conv_lr_coef = conf['conv_lr_coef']

    trainer = Trainer.alloc(
            rbm,
            visible_batch=batch_x,
            lrdict={
                # higher learning rate ok with CD1
                rbm.v_prec: sharedX(base_lr, 'prec_lr'),
                rbm.filters_hs: sharedX(conv_lr_coef*base_lr, 'filters_hs_lr'),
                rbm.conv_bias_hs: sharedX(base_lr, 'conv_bias_hs_lr'),
                rbm.conv_mu: sharedX(base_lr, 'conv_mu_lr'),
                rbm.conv_alpha: sharedX(base_lr, 'conv_alpha_lr'),
                rbm.conv_lambda: sharedX(conv_lr_coef*base_lr, 'conv_lambda_lr'),
                },
            conf = conf
            )

    ntrain_batches = 1 #because we randomly generate minibatches
    print 'start building function'
    training_updates = trainer.updates() #
    train_fn = theano.function(inputs=[batch_idx],
            outputs=[],
	    #mode='FAST_COMPILE',
            #mode='DEBUG_MODE',
	    updates=training_updates	    
	    )  #

    print 'training...'
    
    iter = 0
    while trainer.annealing_coef.get_value()>=0: #
        dummy = train_fn(iter%ntrain_batches) #
        #trainer.print_status()
	if iter % 100 == 0:
            rbm.dump_to_file(os.path.join(_temp_data_path_,'rbm_%06i.pkl'%iter))
        if iter <= 1000 and not (iter % 100): #
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        elif not (iter % 1000):
            trainer.print_status()
            trainer.save_weights_to_grey_files()
        iter += 1


def main_train():
    print 'start main_train'
    main0(dict(
        conf=dict(
            dataset='Brodatz',
            chain_reset_prob=.02,#approx CD-50
            unnatural_grad=True,
            alpha_logdomain=True,
            conv_alpha0=10.,
            global_alpha0=10.,
            alpha_min=1.,
            alpha_max=100.,
            lambda_min=0,
            lambda_max=10,
            lambda0=0.001,
            lambda_logdomain=False,
            conv_bias0=0., 
            conv_bias_irange=1.,#conv_bias0 +- this
            conv_mu0 = 1.0,
            train_iters=600000,
            base_lr_per_example=0.0003,
            conv_lr_coef=1.0,
            batchsize=64,
            n_filters_hs=32,
            v_prec_init=10, # this should increase with n_filters_hs?
            filters_hs_size=11,
            filters_irange=.1,
            zero_out_interior_weights=False,
            #sparsity_weight_conv=0,#numpy.float32(500),
            #sparsity_weight_global=0.,
            particles_min=-50,
            particles_max=50,
            #problem_term_vWWv_weight = 0.,
            #problem_term_vIv_weight = 0.,
            n_tiled_conv_offset_diagonally = 1,
            )))
    

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        sys.exit(main_train())
    if sys.argv[1] == 'sampling':
	sys.exit(main_sample(sys.argv[2]))
    #if sys.argv[1] == 'inpaint':
    #    sys.exit(main_inpaint(sys.argv[2]))
    if sys.argv[1] == 'print_status':
        sys.exit(main_print_status(sys.argv[2]))