import torch
from torch import Tensor
import numpy as np


class StochasticInterpolantMatcher:
    """
    Class to compute the vector fields in the stochastic interpolant.
    """

    def __init__(self):
        """
        x_t = I(t,x_0,x_1) + gamma(t) * z
        I(t,x_0,x_1) = t * x_1 + (1 - t) * x_0
        gamma(t) = sqrt(t(1 - t)) * sigma
        """
        pass

    def gamma_(self, t, sigma, stable=False):
        """
        Compute gamma(t)
        """
        if isinstance(t, Tensor):
            if stable:
                return sigma * torch.sqrt(t * (1 - t)).clip(min=0.1)
            else:
                return sigma * torch.sqrt(t * (1 - t))
        else:
            if stable:
                return sigma * np.sqrt(t * (1 - t)).clip(min=0.1)
            else:
                return sigma * np.sqrt(t * (1 - t))

    def gamma_deriv(self, t, sigma):
        """
        Compute the derivative of gamma(t) over t
        """
        return (0.5 - t) * sigma ** 2 / self.gamma_(t, sigma, stable=True)

    def velocity(self, vel, den, t, sigma):
        """
        Compute the velocity b(t,x_t) = v(t,x_t) + gamma_deriv(t) * denoiser(t,x_t)
        """
        return vel + self.gamma_deriv(t, sigma) * den

    def drift(self, vel, den, t, sigma):
        """
        Compute the drift term: b_F(t,x_t) = b(t,x_t) + eps(t) * score(t,x_t)
        """
        ### eps = 0.5*\sigma^2, which coincides with dx_t = (x_1-x_t)/(1-t) dt + \sigma dB_t
        eps = 0.5 * sigma**2
        return self.velocity(vel, den, t, sigma) + eps * self.denoiser_to_score(den, t, sigma)

    def diffusion(self, sigma):
        """
        Compute the diffusion term: sigma
        """
        return sigma

    def expectation_on_x0(self, xt, vel, den, t, sigma):
        return xt - t * vel - self.gamma_(t, sigma, stable=False) * den

    def expectation_on_x1(self, xt, vel, den, t, sigma):
        return self.expectation_on_x0(xt, vel, den, t, sigma) + vel

    def expectation_on_vel(self, xt, x0, x1, t, sigma):
        return x1 - x0

    def expectation_on_den(self, xt, x0, x1, t, sigma):
        return (xt - t * x1 - (1 - t) * x0) / self.gamma_(t, sigma, stable=True)

    def likelihood(self, xt, x0, x1, t, sigma):
        """
        Compute the likelihood q_t(x_t|x_0, x_1)
        """
        ndim = x0.numel()
        mu = t * x1 + (1 - t) * x0
        gamma_t = self.gamma_(t, sigma, stable=True)
        nll = torch.norm(xt - mu) ** 2 / (2 * gamma_t**2) \
              + 0.5 * ndim * np.log(2 * np.pi) \
              + ndim * np.log(gamma_t)
        nll /= ndim  # scaling for numerical stability
        return torch.exp(-nll)

    def conditional_score(self, xt, x0, x1, t, sigma):
        """
        Compute the conditional score \nabla log q_t(x_t|x_0, x_1)
        """
        mu = t * x1 + (1 - t) * x0
        cscore = -(xt - mu) / self.gamma_(t, sigma, stable=True)**2
        return cscore

    def denoiser_to_score(self, z, t, sigma):
        score = -z / self.gamma_(t, sigma, stable=True)
        return score

    def sample_xt(self, x0, x1, t, sigma):
        z = torch.randn_like(x0)
        xt = t.reshape(-1, 1) * (x1 - x0) + x0 + self.gamma_(t, sigma, stable=False).reshape(-1, 1) * z
        return xt, z

    def sample_conditional_flow(self, x0, x1, t, sigma):
        """
        Compute the sample xt along the geodesic from x0 to x1
        and the conditional vector field ut(xt|z).

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        t  : Tensor, shape (bs,)
            represents the noised time

        Returns
        -------
        xt : Tensor, shape (bs, *dim)
            represents the samples drawn along the geodesic
        vel : conditional velocity field, d{I(t,x_0,x_1)}/dt=x_1-x_0
        den : denoiser, z
        """
        xt, z = self.sample_xt(x0, x1, t, sigma)
        return xt, x1 - x0, z


INTERP_MATCHER = StochasticInterpolantMatcher()


if __name__ == "__main__":
    sigma = 0.2
    n = 10
    x0 = torch.randn(n, 3)
    x1 = torch.randn(n, 3)
    t = np.random.rand()
    t_diff = torch.ones(n, 1) * t
    xt, vel_gt, den_gt = INTERP_MATCHER.sample_conditional_flow(x0, x1, t_diff, sigma)
    drf_gt = INTERP_MATCHER.drift(vel_gt, den_gt, t_diff, sigma)
    drf_bm = (x1 - xt) / (1 - t_diff)
    assert torch.allclose(drf_gt, drf_bm, rtol=1e-6, atol=1e-6)
