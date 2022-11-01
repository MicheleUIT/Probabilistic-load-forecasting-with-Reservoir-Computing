# import torch
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

        self.distributions = self._get_priors()
        
        self._torch2pyro()


    def forward(self, x, y=None):
        mean = self.model(x).squeeze(-1)

        sigma = pyro.sample("sigma", self.distributions[-2]).to(self.device)

        with pyro.plate("data", x.shape[0], device=self.device):
            obs = pyro.sample("obs", self.distributions[-1](mean, sigma), obs=y).to(self.device)
        return mean

    
    def _torch2pyro(self):
        pyro.nn.module.to_pyro_module_(self.model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(self.distributions[0].expand(value.shape).to_event(value.dim())))


    def _get_priors(self):
        distributions = []
        distr_list = self.config.distributions
        param_list = self.config.parameters

        for i in range(len(distr_list)):
            if distr_list[i] == "gauss":
                try:
                    p = param_list[i]
                except:
                    distributions.append(dist.Normal)
                else:
                    distributions.append(dist.Normal(p[0],p[1]))
                
            elif distr_list[i] == "unif":
                try:
                    p = param_list[i]
                except:
                    distributions.append(dist.Uniform)
                else:
                    distributions.append(dist.Uniform(p[0],p[1]))
            else:
                raise ValueError(f"{distr_list[i]['name']} prior distribution not defined.")
        
        return distributions


    def render_model(self, model_args):
        return pyro.render_model(self, model_args, render_distributions=True, filename="model.png")
    


# when defining the guide, put constraints on variances

class BayesianGuide(PyroModule):
    def __init__(self, torch_model, config, device):
        super().__init__()

        self.device = device
        self.config = config
        self.model = torch_model

        self.distributions = self.get_priors()
        
        self.torch2pyro()


    def forward(self, x, y=None):
        mean = self.model(x).squeeze(-1)

        sigma = pyro.sample("sigma", self.distributions[-2]).to(self.device)

        return mean

    
    def torch2pyro(self):
        pyro.nn.module.to_pyro_module_(self.model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(self.distributions[0].expand(value.shape).to_event(value.dim())))


    def render_model(self, model_args):
        return pyro.render_model(self, model_args, render_distributions=True, filename="guide.png")