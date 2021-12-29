import math
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import init


__all__ = ["triple_uniform_", "Dense", "FeatureWeighting"]


def triple_uniform_(n_triple: int = 40):
    def init_func_(
        tensor: Tensor,
        gain: float = 1.0,
    ):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        t = torch.nn.init._no_grad_uniform_(tensor, -a, a)
        t[:, -n_triple:] = 0.0

        return t

    return init_func_


class Dense(nn.Linear):
    """
    Applies a linear transformation to the incoming data, and if activation is not None,
    apply activation function after linear transformation.

    Attributes
    ----------
    in_features : int
        size of each input sample.
    out_features : int
        size of each output sample
    bias : bool, default=True
        If set to False, the layer will not learn an additive bias.
    activation : collable or None, default=None
        activation function after calculating the linear layer.
    weight_init : collable, default=torch.nn.init.xavier_uniform_
    bias_init : collable, default=torch.nn.init.constant_
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation=None,
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.constant_,
    ) -> None:
        self.activation = activation
        self.weight_init = weight_init
        self.bias_init = bias_init
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self) -> None:
        """
        Reinitialize model weight and bias values.
        """
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias, val=0.0)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute layer output.

        Parameters
        ----------
        inputs : torch.Tensor
            batch of input values.

        Returns
        -------
        y : torch.Tensor
            layer output.
        """
        # compute linear layer y = xW^T + b
        y = super().forward(inputs)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class FeatureWeighting(nn.Module):
    """
    Multiplies a different weighting to the each feature value of incoming data.

    Attributes
    ----------
    in_features : int
        size of each input sample.
    weight_init : collable, default=torch.nn.init.ones_
    """

    def __init__(self, in_features: int, weight_init=nn.init.ones_) -> None:
        super().__init__()
        self.weighting = nn.parameter.Parameter(torch.empty(in_features))
        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reinitialize model weight and bias values.
        """
        self.weight_init(self.weighting)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Compute layer output.

        Parameters
        ----------
        inputs : torch.Tensor
            batch of input values.

        Returns
        -------
        y : torch.Tensor
            layer output.
        """
        y = inputs * self.weighting
        return y
