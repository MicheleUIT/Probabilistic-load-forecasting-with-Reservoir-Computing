import torch
import pyro

import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample
# from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
# from pyro.optim import Adam
# from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS, HMC


class BayesianModel(PyroModule):
    def __init__(self, torch_model, config, device):
        super().__init__()

        self.device = device
        self.config = config
        self.model = torch_model

        self.prior_dist = []

        for prior in self.config.prior:
            if prior["name"] == "Gauss":
                self.prior_dist.append(dist.Normal(torch.tensor(prior["params"][0], device=self.device),
                                            torch.tensor(prior["params"][1], device=self.device)))
            elif prior["name"] == "Uniform":
                self.prior_dist.append(dist.Uniform(prior["params"][0], prior["params"][1]))
            else:
                raise ValueError("Undefined prior distribution.")
        
        self.torch2pyro()


    def torch2pyro(self):
        pyro.nn.module.to_pyro_module_(self.model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(self.prior_dist[0].expand(value.shape).to_event(value.dim())))


    def forward(self, x, y=None):
        mean = self.model(x).squeeze(-1)

        sigma = pyro.sample("sigma", self.prior_dist[1]).to(self.device)

        with pyro.plate("data", x.shape[0], device=self.device):
            # should I change also the likelihood?
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean


    def render_model(self, model_args):
        return pyro.render_model(self, model_args, render_distributions=True)
    


# when defining the guide, put constraints on variances