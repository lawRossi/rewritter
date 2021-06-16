"""
Assorted utilities for working with neural networks in AllenNLP.
"""
from typing import  List, Optional, Sequence, TypeVar
import torch


T = TypeVar('T')


def combine_tensors(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of tensors using element-wise operations and concatenation, specified by a
    ``combination`` string.  The string refers to (1-indexed) positions in the input tensor list,
    and looks like ``"1,2,1+2,3-1"``.

    We allow the following kinds of combinations: ``x``, ``x*y``, ``x+y``, ``x-y``, and ``x/y``,
    where ``x`` and ``y`` are positive integers less than or equal to ``len(tensors)``.  Each of
    the binary operations is performed elementwise.  You can give as many combinations as you want
    in the ``combination`` string.  For example, for the input string ``"1,2,1*2"``, the result
    would be ``[1;2;1*2]``, as you would expect, where ``[;]`` is concatenation along the last
    dimension.

    If you have a fixed, known way to combine tensors that you use in a model, you should probably
    just use something like ``torch.cat([x_tensor, y_tensor, x_tensor * y_tensor])``.  This
    function adds some complexity that is only necessary if you want the specific combination used
    to be `configurable`.

    If you want to do any element-wise operations, the tensors involved in each element-wise
    operation must have the same shape.

    This function also accepts ``x`` and ``y`` in place of ``1`` and ``2`` in the combination
    string.
    """
    combination = combination.replace('x', '1').replace('y', '2')
    to_concatenate = [_get_combination(piece, tensors) for piece in combination.split(',')]
    return torch.cat(to_concatenate, dim=-1)


def _rindex(sequence: Sequence[T], obj: T) -> int:
    """
    Return zero-based index in the sequence of the last item whose value is equal to obj.  Raises a
    ValueError if there is no such item.

    Parameters
    ----------
    sequence : ``Sequence[T]``
    obj : ``T``

    Returns
    -------
    zero-based index associated to the position of the last item equal to obj
    """
    for i in range(len(sequence) - 1, -1, -1):
        if sequence[i] == obj:
            return i

    raise ValueError(f"Unable to find {obj} in sequence {sequence}.")


def _get_combination(combination: str, tensors: List[torch.Tensor]) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return tensors[index]
    else:
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            return first_tensor * second_tensor
        elif operation == '/':
            return first_tensor / second_tensor
        elif operation == '+':
            return first_tensor + second_tensor
        elif operation == '-':
            return first_tensor - second_tensor


def combine_tensors_and_multiply(combination: str,
                                 tensors: List[torch.Tensor],
                                 weights: torch.nn.Parameter) -> torch.Tensor:
    """
    Like :func:`combine_tensors`, but does a weighted (linear) multiplication while combining.
    This is a separate function from ``combine_tensors`` because we try to avoid instantiating
    large intermediate tensors during the combination, which is possible because we know that we're
    going to be multiplying by a weight vector in the end.

    Parameters
    ----------
    combination : ``str``
        Same as in :func:`combine_tensors`
    tensors : ``List[torch.Tensor]``
        A list of tensors to combine, where the integers in the ``combination`` are (1-indexed)
        positions in this list of tensors.  These tensors are all expected to have either three or
        four dimensions, with the final dimension being an embedding.  If there are four
        dimensions, one of them must have length 1.
    weights : ``torch.nn.Parameter``
        A vector of weights to use for the combinations.  This should have shape (combined_dim,),
        as calculated by :func:`get_combined_dim`.
    """
    combination = combination.replace('x', '1').replace('y', '2')
    pieces = combination.split(',')
    tensor_dims = [tensor.size(-1) for tensor in tensors]
    combination_dims = [_get_combination_dim(piece, tensor_dims) for piece in pieces]
    dims_so_far = 0
    to_sum = []
    for piece, combination_dim in zip(pieces, combination_dims):
        weight = weights[dims_so_far:(dims_so_far + combination_dim)]
        dims_so_far += combination_dim
        to_sum.append(_get_combination_and_multiply(piece, tensors, weight))
    result = to_sum[0]
    for result_piece in to_sum[1:]:
        result = result + result_piece
    return result


def _get_combination_and_multiply(combination: str,
                                  tensors: List[torch.Tensor],
                                  weight: torch.nn.Parameter) -> torch.Tensor:
    if combination.isdigit():
        index = int(combination) - 1
        return torch.matmul(tensors[index], weight)
    else:
        first_tensor = _get_combination(combination[0], tensors)
        second_tensor = _get_combination(combination[2], tensors)
        operation = combination[1]
        if operation == '*':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '/':
            if first_tensor.dim() > 4 or second_tensor.dim() > 4:
                raise ValueError("Tensors with dim > 4 not currently supported")
            desired_dim = max(first_tensor.dim(), second_tensor.dim()) - 1
            if first_tensor.dim() == 4:
                expanded_dim = _rindex(first_tensor.size(), 1)
                first_tensor = first_tensor.squeeze(expanded_dim)
            if second_tensor.dim() == 4:
                expanded_dim = _rindex(second_tensor.size(), 1)
                second_tensor = second_tensor.squeeze(expanded_dim)
            intermediate = first_tensor * weight
            result = torch.matmul(intermediate, second_tensor.pow(-1).transpose(-1, -2))
            if result.dim() == desired_dim + 1:
                result = result.squeeze(-1)
            return result
        elif operation == '+':
            return torch.matmul(first_tensor, weight) + torch.matmul(second_tensor, weight)
        elif operation == '-':
            return torch.matmul(first_tensor, weight) - torch.matmul(second_tensor, weight)


def get_combined_dim(combination: str, tensor_dims: List[int]) -> int:
    """
    For use with :func:`combine_tensors`.  This function computes the resultant dimension when
    calling ``combine_tensors(combination, tensors)``, when the tensor dimension is known.  This is
    necessary for knowing the sizes of weight matrices when building models that use
    ``combine_tensors``.

    Parameters
    ----------
    combination : ``str``
        A comma-separated list of combination pieces, like ``"1,2,1*2"``, specified identically to
        ``combination`` in :func:`combine_tensors`.
    tensor_dims : ``List[int]``
        A list of tensor dimensions, where each dimension is from the `last axis` of the tensors
        that will be input to :func:`combine_tensors`.
    """
    combination = combination.replace('x', '1').replace('y', '2')
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        return first_tensor_dim
