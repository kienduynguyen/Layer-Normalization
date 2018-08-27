
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.autograd import Function, Variable, gradcheck
from math import sqrt


class layer_norm(Function):

	@staticmethod
	def forward(ctx, input, gain=None, bias=None):
		ctx.save_for_backward(input, gain, bias)
		mean = input.mean(-1, keepdim=True)
		var = input.var(-1, unbiased=False, keepdim=True)
		input_normalized = (input - mean) / torch.sqrt(var + 1e-9)

		if gain is not None and bias is not None:
			output = input_normalized * gain + bias
		elif not (gain is None and bias is None):
			raise RuntimeError("gain and bias of LayerNorm should be both None or not None!")
		else:
			output = input_normalized

		return output

	@staticmethod
	def backward(ctx, grad_output):
		input, gain, bias = ctx.saved_variables
		mean = input.mean(-1, keepdim=True)
		var = input.var(-1, unbiased=False, keepdim=True)
		input_normalized = (input - mean) / torch.sqrt(var + 1e-9)
		grad_input = grad_gain = grad_bias = None

		N = input.size(-1)
		input_mu = input - mean
		std_inv = 1. / torch.sqrt(var + 1e-9)

		if ctx.needs_input_grad[0]:
			if gain is not None:
				grad_input_normalized = (grad_output * gain)
			else:
				grad_input_normalized = grad_output
			grad_var = (-0.5) * (grad_input_normalized * input_mu).sum(dim=-1, keepdim=True) * (std_inv ** 3)
			grad_mean = (-1.0) * (grad_input_normalized * std_inv).sum(dim=-1, keepdim=True) \
				- 2.0 * grad_var * input_mu.mean(dim=-1, keepdim=True)
			grad_input = grad_input_normalized * std_inv + (2. / N) * grad_var *  input_mu + (1. / N) * grad_mean
		if gain is not None and ctx.needs_input_grad[1]:
			grad_gain = (grad_output * input_normalized).sum(dim=0)
		if bias is not None and ctx.needs_input_grad[2]:
			grad_bias = grad_output.sum(dim=0)

		return grad_input, grad_gain, grad_bias


class LayerNorm(nn.Module):
	"""
	Layer Normalization layer's implementation which follows paper "https://arxiv.org/abs/1607.06450".
	Notes: This implement serves for the (N x C) tensor only where C is the number of features.

	"""
	def __init__(self, num_features, affine=True):
		super(LayerNorm, self).__init__()
		self.affine = affine
		if self.affine:
			self.weight = Parameter(torch.Tensor(num_features))
			self.bias = Parameter(torch.Tensor(num_features))
		else:
			self.register_parameter("weight", None)
			self.register_parameter("bias", None)
		self.reset_parameters()

	def reset_parameters(self):
		if self.affine:
			self.weight.data.uniform_()
			self.bias.data.zero_()

	def forward(self, input):
		return layer_norm.apply(input, self.weight, self.bias)

	def __repr__(self):
		return ("{name}(num_features={num_features}, affine={affine})"
			.format(name=self.__class__.__name__, **self.__dict__))


if __name__ == "__main__":
	torch.manual_seed(123)
	input = (Variable(torch.randn(30, 20).double(), requires_grad=True),
		Variable(torch.randn(20).double(), requires_grad=True),
		Variable(torch.randn(20).double(), requires_grad=True))

	test = gradcheck(layer_norm.apply, input)
	print("test:", test)
