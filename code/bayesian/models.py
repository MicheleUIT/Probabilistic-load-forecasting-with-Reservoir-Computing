import torch
import pyro

import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive, MCMC, NUTS, HMC


class BayesianModel(PyroModule):
    def __init__(self, torch_model, config, device):
        super().__init__()

        self.device = device
        self.config = config
        
        pyro.nn.module.to_pyro_module_(torch_model)
        self.model = torch_model
        
        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(dist.Normal(torch.tensor(0., device=self.device), torch.tensor(1., device=self.device))
                                                                .expand(value.shape).to_event(value.dim())))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(0., 10.)).to(self.device)
        mean = self.model(x).squeeze(-1)
        with pyro.plate("data", x.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean

    def render_model(self, model_args):
        pyro.render_model(self, model_args, render_distributions=True)
    


