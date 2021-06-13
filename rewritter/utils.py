"""
Assorted utilities for working with neural networks in AllenNLP.
"""
from typing import  List, Optional, Sequence, TypeVar
import logging
import torch
from allennlp.common.checks import ConfigurationError


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
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
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
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
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
        else:
            raise ConfigurationError("Invalid operation: " + operation)


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
    if len(tensors) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
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
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
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
        else:
            raise ConfigurationError("Invalid operation: " + operation)


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
    if len(tensor_dims) > 9:
        raise ConfigurationError("Double-digit tensor lists not currently supported")
    combination = combination.replace('x', '1').replace('y', '2')
    return sum([_get_combination_dim(piece, tensor_dims) for piece in combination.split(',')])


def _get_combination_dim(combination: str, tensor_dims: List[int]) -> int:
    if combination.isdigit():
        index = int(combination) - 1
        return tensor_dims[index]
    else:
        if len(combination) != 3:
            raise ConfigurationError("Invalid combination: " + combination)
        first_tensor_dim = _get_combination_dim(combination[0], tensor_dims)
        second_tensor_dim = _get_combination_dim(combination[2], tensor_dims)
        operation = combination[1]
        if first_tensor_dim != second_tensor_dim:
            raise ConfigurationError("Tensor dims must match for operation \"{}\"".format(operation))
        return first_tensor_dim


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.
    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def get_device_of(tensor: torch.Tensor) -> int:
    """
    Returns the device of the tensor.
    """
    if not tensor.is_cuda:
        return -1
    else:
        return tensor.get_device()


def flatten_and_batch_shift_indices(indices: torch.Tensor,
                                    sequence_length: int) -> torch.Tensor:
    """
    This is a subroutine for :func:`~batched_index_select`. The given ``indices`` of size
    ``(batch_size, d_1, ..., d_n)`` indexes into dimension 2 of a target tensor, which has size
    ``(batch_size, sequence_length, embedding_size)``. This function returns a vector that
    correctly indexes into the flattened target. The sequence length of the target must be
    provided to compute the appropriate offsets.

    .. code-block:: python

        indices = torch.ones([2,3], dtype=torch.long)
        # Sequence length of the target tensor.
        sequence_length = 10
        shifted_indices = flatten_and_batch_shift_indices(indices, sequence_length)
        # Indices into the second element in the batch are correctly shifted
        # to take into account that the target tensor will be flattened before
        # the indices are applied.
        assert shifted_indices == [1, 1, 1, 11, 11, 11]

    Parameters
    ----------
    indices : ``torch.LongTensor``, required.
    sequence_length : ``int``, required.
        The length of the sequence the indices index into.
        This must be the second dimension of the tensor.

    Returns
    -------
    offset_indices : ``torch.LongTensor``
    """
    # Shape: (batch_size)
    if torch.max(indices) >= sequence_length or torch.min(indices) < 0:
        raise ConfigurationError(f"All elements in indices should be in range (0, {sequence_length - 1})")
    offsets = get_range_vector(indices.size(0), get_device_of(indices)) * sequence_length
    for _ in range(len(indices.size()) - 1):
        offsets = offsets.unsqueeze(1)

    # Shape: (batch_size, d_1, ..., d_n)
    offset_indices = indices + offsets

    # Shape: (batch_size * d_1 * ... * d_n)
    offset_indices = offset_indices.view(-1)
    return offset_indices


def batched_index_select(target: torch.Tensor,
                         indices: torch.LongTensor,
                         flattened_indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
    """
    The given ``indices`` of size ``(batch_size, d_1, ..., d_n)`` indexes into the sequence
    dimension (dimension 2) of the target, which has size ``(batch_size, sequence_length,
    embedding_size)``.

    This function returns selected values in the target with respect to the provided indices, which
    have size ``(batch_size, d_1, ..., d_n, embedding_size)``. This can use the optionally
    precomputed :func:`~flattened_indices` with size ``(batch_size * d_1 * ... * d_n)`` if given.

    An example use case of this function is looking up the start and end indices of spans in a
    sequence tensor. This is used in the
    :class:`~allennlp.models.coreference_resolution.CoreferenceResolver`. Model to select
    contextual word representations corresponding to the start and end indices of mentions. The key
    reason this can't be done with basic torch functions is that we want to be able to use look-up
    tensors with an arbitrary number of dimensions (for example, in the coref model, we don't know
    a-priori how many spans we are looking up).

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A 3 dimensional tensor of shape (batch_size, sequence_length, embedding_size).
        This is the tensor to be indexed.
    indices : ``torch.LongTensor``
        A tensor of shape (batch_size, ...), where each element is an index into the
        ``sequence_length`` dimension of the ``target`` tensor.
    flattened_indices : Optional[torch.Tensor], optional (default = None)
        An optional tensor representing the result of calling :func:~`flatten_and_batch_shift_indices`
        on ``indices``. This is helpful in the case that the indices can be flattened once and
        cached for many batch lookups.

    Returns
    -------
    selected_targets : ``torch.Tensor``
        A tensor with shape [indices.size(), target.size(-1)] representing the embedded indices
        extracted from the batch flattened target tensor.
    """
    if flattened_indices is None:
        # Shape: (batch_size * d_1 * ... * d_n)
        flattened_indices = flatten_and_batch_shift_indices(indices, target.size(1))

    # Shape: (batch_size * sequence_length, embedding_size)
    flattened_target = target.view(-1, target.size(-1))

    # Shape: (batch_size * d_1 * ... * d_n, embedding_size)
    flattened_selected = flattened_target.index_select(0, flattened_indices)
    selected_shape = list(indices.size()) + [target.size(-1)]
    # Shape: (batch_size, d_1, ..., d_n, embedding_size)
    selected_targets = flattened_selected.view(*selected_shape)
    return selected_targets


def flattened_index_select(target: torch.Tensor,
                           indices: torch.LongTensor) -> torch.Tensor:
    """
    The given ``indices`` of size ``(set_size, subset_size)`` specifies subsets of the ``target``
    that each of the set_size rows should select. The `target` has size
    ``(batch_size, sequence_length, embedding_size)``, and the resulting selected tensor has size
    ``(batch_size, set_size, subset_size, embedding_size)``.

    Parameters
    ----------
    target : ``torch.Tensor``, required.
        A Tensor of shape (batch_size, sequence_length, embedding_size).
    indices : ``torch.LongTensor``, required.
        A LongTensor of shape (set_size, subset_size). All indices must be < sequence_length
        as this tensor is an index into the sequence_length dimension of the target.

    Returns
    -------
    selected : ``torch.Tensor``, required.
        A Tensor of shape (batch_size, set_size, subset_size, embedding_size).
    """
    if indices.dim() != 2:
        raise ConfigurationError("Indices passed to flattened_index_select had shape {} but "
                                 "only 2 dimensional inputs are supported.".format(indices.size()))
    # Shape: (batch_size, set_size * subset_size, embedding_size)
    flattened_selected = target.index_select(1, indices.view(-1))

    # Shape: (batch_size, set_size, subset_size, embedding_size)
    selected = flattened_selected.view(target.size(0), indices.size(0), indices.size(1), -1)
    return selected


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)
