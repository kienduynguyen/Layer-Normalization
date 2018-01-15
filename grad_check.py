
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import sys
import collections

from torch.autograd import Variable


def varible_to_tensor(variable):
	if isinstance(variable, Variable):
		return variable.data
	elif isinstance(variable, collections.Iterable):
		return tuple(element.data for element in variable)
	else:
		raise TypeError("Input is expected to be a Variable or a sequence of Variables")


def make_contiguous_tuple(x, is_tuple=True):
	if isinstance(x, Variable):
		return (x.contiguous(),) if is_tuple else x.contiguous()
	elif isinstance(x, collections.Iterable):
		return tuple(make_contiguous_tuple(element, is_tuple=False) for element in x if element is not None)
	else:
		raise TypeError("Input is expected to be a Variable or a sequence of Variables")


def create_jacobian_matrix(num_in, num_out):
	return torch.zeros(num_in, num_out)


def zero_grad(x):
	if isinstance(x, Variable):
		if x.grad is not None:
			x.grad.detach_()
			x.grad.data.zero_()
	elif isinstance(x, collections.Iterable):
		for element in x:
			zero_grad(element)
	else:
		raise TypeError("Input is expected to be a Variable or a sequence of Variables")


def numerical_grad(fn, input, eps=1e-3):
	input = make_contiguous_tuple(input)
	output = make_contiguous_tuple(fn(*input))
	input_tensor = varible_to_tensor(input)

	def _single_output_grad(idx, num_out):
		jacobians = []
		output_right = torch.DoubleTensor(num_out)
		output_left = torch.DoubleTensor(num_out)

		for in_tensor in input_tensor:
			jacobian = create_jacobian_matrix(in_tensor.numel(), num_out)
			flat_tensor = in_tensor.view(-1)

			for i in range(flat_tensor.numel()):
				temp = flat_tensor[i]
				flat_tensor[i] = temp - eps
				temp_output = make_contiguous_tuple(fn(*input))[idx]
				output_left.copy_(temp_output.data, broadcast=False)

				flat_tensor[i] = temp + eps
				temp_output = make_contiguous_tuple(fn(*input))[idx]
				output_right.copy_(temp_output.data, broadcast=False)
				flat_tensor[i] = temp

				output_right.add_(-1, output_left).div_(2 * eps)
				jacobian[i] = output_right

			jacobians.append(jacobian)

		return jacobians

	return [_single_output_grad(idx, out_tensor.numel()) for idx, out_tensor in enumerate(output) 
			if out_tensor.requires_grad]


def analytical_grad(fn, input):
	input = make_contiguous_tuple(input)
	output = make_contiguous_tuple(fn(*input))

	def _single_output_grad(single_output):
		jacobians = []
		grad_single_output = single_output.data.clone().zero_()
		flat_grad_single_output = grad_single_output.view(-1)

		for in_tensor in input:
			jacobian = create_jacobian_matrix(in_tensor.numel(), single_output.numel())
			jacobians.append(jacobian)

		for i in range(single_output.numel()):
			zero_grad(input)
			flat_grad_single_output.zero_()
			flat_grad_single_output[i] = 1
			single_output.backward(grad_single_output, create_graph=True)

			for j, in_tensor in enumerate(input):
				jacobians[j][:,i].copy_(in_tensor.grad.data, broadcast=False)

		return jacobians

	return [_single_output_grad(out_tensor) for out_tensor in output if out_tensor.requires_grad]


def grad_check(fn, input, eps=1e-6, atol=1e-5, rtol=1e-3, raise_except=True):

	def _raise_except(msg):
		if raise_exception:
			raise RuntimeError(msg)
		return False

	def _check_size(numerical_matrix, analytical_matrix):
		if isinstance(numerical_matrix, collections.Sequence) and isinstance(analytical_matrix, collections.Sequence):
			if len(numerical_matrix) != len(analytical_matrix):
				return _raise_except("Length of numerical and analytical jacobian matrices not match!")

			for (numerical, analytical) in zip(numerical_matrix, analytical_matrix):
				_check_size(numerical, analytical)
		elif torch.is_tensor(numerical_matrix) and torch.is_tensor(analytical_matrix):
			if tuple(numerical_matrix.size()) != tuple(analytical_matrix.size()):
				return _raise_except("Size of numerical and analytical jacobian matrices not match!")
		else:
			return _raise_except("Type of numerical and analytical jacobian matrices not match!")

	def _check_error(numerical_matrix, analytical_matrix):
		if isinstance(numerical_matrix, collections.Sequence):
			for (numerical, analytical) in zip(numerical_matrix, analytical_matrix):
				_check_error(numerical, analytical)
		elif torch.is_tensor(numerical_matrix) and torch.is_tensor(analytical_matrix):
			if not ((numerical_matrix - analytical_matrix).abs() <= (atol + rtol * analytical_matrix.abs())).all():
				return _raise_except("Error of numerical and analytical jacobian matrices is too big!")

	numerical_jacobians = numerical_grad(fn, input)
	analytical_jacobians = analytical_grad(fn, input)

	_check_size(numerical_jacobians, analytical_jacobians)
	_check_error(numerical_jacobians, analytical_jacobians)
	
	return True


if __name__ == "__main__":
	from LayerNorm import layer_norm

	input = (Variable(torch.randn(30, 20).double(), requires_grad=True),
		Variable(torch.randn(20).double(), requires_grad=True),
		Variable(torch.randn(20).double(), requires_grad=True))

	test = grad_check(layer_norm.apply, input, eps=1e-6, atol=1e-4)
	print("test:", test)