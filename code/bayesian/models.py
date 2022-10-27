import torch
import pyro
import yaml

import pyro.distributions as dist

from munch import munchify
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

        self.get_priors()
        
        self.torch2pyro()


    # def torch2pyro(self):
    #     pyro.nn.module.to_pyro_module_(self.model)

    #     for m in self.model.modules():
    #         for name, value in list(m.named_parameters(recurse=False)):
    #             setattr(m, name, PyroSample(self.prior_dist[0].expand(value.shape).to_event(value.dim())))
    
    def torch2pyro(self):
        pyro.nn.module.to_pyro_module_(self.model)

        for m in self.model.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                # come passare i parametri?
                # to(self.device) basta per spostare PyroSample sulla gpu?
                setattr(m, name, PyroSample(self.prior_dist[0]().expand(value.shape).to_event(value.dim()).to(self.device)))


    # def get_priors(self):
    #     config_priors = munchify(yaml.safe_load(open("config_priors.yaml"))[f"{self.config.pyro_model}"])
    #     self.prior_dist = []
    #     for prior in config_priors.distributions:
    #         if prior == "gauss":
    #             self.prior_dist.append(dist.Normal(torch.tensor(prior["params"][0], device=self.device),
    #                                         torch.tensor(prior["params"][1], device=self.device)))
    #         elif prior["name"] == "Uniform":
    #             self.prior_dist.append(dist.Uniform(prior["params"][0], prior["params"][1]))
    #         else:
    #             raise ValueError(f"{prior['name']} prior distribution not defined.")

    def get_priors(self):
        config_priors = munchify(yaml.safe_load(open("config_priors.yaml"))[f"{self.config.pyro_model}"])
        self.prior_dist = []
        for prior in config_priors.distributions:
            if prior == "gauss":
                 # posso passare le distribuzioni come funzioni?
                 # come passo i parametri in maniera flessibile in torch2pyro?
                 # potrebbero essere di numero variabile...
                self.prior_dist.append(dist.Normal)
            elif prior == "unif":
                self.prior_dist.append(dist.Uniform)
            else:
                raise ValueError(f"{prior['name']} prior distribution not defined.")


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