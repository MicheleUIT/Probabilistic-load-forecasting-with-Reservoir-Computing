import torch
import pyro

import pyro.distributions as dist

from pyro.nn import PyroModule, PyroSample


class TorchModel(torch.nn.Module):
    """
    Deterministic model

    :param list widths: List of layers' widths.
    :param string activation: String specifying the activation function to use.
    """
    def __init__(self, widths, activation, dropout=False, p=0.0, quantiles=None):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        if activation == "tanh":
            a = torch.nn.Tanh()
        elif activation == "relu":
            a = torch.nn.ReLU()
        else:
            raise ValueError(f"{activation} not defined.")
        
        dp = torch.nn.Dropout(p)

        for i in range(len(widths)-2):
            self.layers.append(torch.nn.Linear(widths[i], widths[i+1]))
            self.layers.append(a)
            if dropout:
                self.layers.append(dp)
        
        output = widths[-1] if quantiles==None else len(quantiles)
        self.layers.append(torch.nn.Linear(widths[-2], output))

    def forward(self, x):
        for f in self.layers:
            x = f(x)
        return x


class BayesianModel(PyroModule):
    """
    Probabilistic model built from a TorchModel

    :param torch_model: TorchModel object to transform into probabilistic model.
    :param config: Configuration settings.
    :param string device: String specifying the device to use, 'cpu' or 'cuda'.
    """
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

        # with pyro.plate("data", size=x.shape[0], subsample_size=50, device=self.device) as ind:
        with pyro.plate("data", device=self.device):
            obs = pyro.sample("obs", self.distributions[-1](mean, sigma),
                            #   obs=y.index_select(0, ind) if y != None else y).to(self.device)
                              obs=y).to(self.device)
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
                    distributions.append(dist.Normal(torch.tensor(float(p[0]), device=self.device),
                                                    torch.tensor(float(p[1]), device=self.device)))
                
            elif distr_list[i] == "unif":
                try:
                    p = param_list[i]
                except:
                    distributions.append(dist.Uniform)
                else:
                    distributions.append(dist.Uniform(torch.tensor(float(p[0]), device=self.device),
                                                    torch.tensor(float(p[1]), device=self.device)))
                    
            elif distr_list[i] == "lapl":
                try:
                    p = param_list[i]
                except:
                    distributions.append(dist.Laplace)
                else:
                    distributions.append(dist.Laplace(torch.tensor(float(p[0]), device=self.device),
                                                    torch.tensor(float(p[1]), device=self.device))) # >0
                    
            elif distr_list[i] == "bern":
                try:
                    p = param_list[i]
                except:
                    raise ValueError(f"Missing parameter for distribution {i} ({distr_list[i]}).")
                else:
                    distributions.append(dist.RelaxedBernoulli(torch.tensor(float(p[1]), device=self.device), # >0
                                                               torch.tensor(float(p[0]), device=self.device)))
                    
            else:
                raise ValueError(f"{distr_list[i]['name']} prior distribution not defined.")
        
        return distributions


    def render_model(self, model_args, filename=None):
        return pyro.render_model(self, model_args, render_distributions=True, filename=filename)


class BayesianLinear_m(PyroModule):
    """
    Linear Bayesian model with no activations for SSVS built "manually"
    """
    def __init__(self, in_features, out_features, device, sigma=1.):
        super().__init__()

        mu = torch.tensor(0., device=device)
        sigma = torch.tensor(sigma, device=device)

        self.weights = PyroSample(dist.Normal(mu, sigma).expand((in_features, out_features)).to_event(2))
        self.bias = PyroSample(dist.Normal(mu, sigma).expand((out_features,)).to_event(1))

    def forward(self, x):
        return x @ self.weights + self.bias


class BayesianLinear_t(PyroModule):
    """
    Linear Bayesian model with no activations for SSVS using torch.nn.Linear
    """
    def __init__(self, in_features, out_features, device, sigma=1.):
        super().__init__()

        mu = torch.tensor(0., device=device)
        sigma = torch.tensor(sigma, device=device)

        self.linear = torch.nn.Sequential(torch.nn.Linear(in_features, out_features)).to(device)
        pyro.nn.module.to_pyro_module_(self.linear)

        setattr(self.linear[0].weight, 'weight', PyroSample(dist.Normal(mu, torch.tensor(1., device=device)).expand([in_features, out_features]).to_event(2)))
        setattr(self.linear[0].bias, 'bias', PyroSample(dist.Normal(mu, sigma).expand([out_features]).to_event(1)))

    def forward(self, x):
        return self.linear(x)


class HorseshoeSSVS(PyroModule):
    """
    SSVS implemented with an horseshoe continuous RV

    :param type: string to choose linear layer type
    """
    def __init__(self, in_features, out_features, type, device):
        super().__init__()

        self.device = device

        # TODO: what is the difference between these two linear layers?
        BL = BayesianLinear_m if type=='m' else BayesianLinear_t

        self.linear1 = BL(in_features, out_features, self.device)
        self.linear2 = BL(in_features, out_features, self.device, 0.001)

    def forward(self, x, y=None):
        tau = pyro.sample("tau", dist.HalfCauchy(1.)).to(self.device)
        lamb = pyro.sample("lamb", dist.HalfCauchy(1.)).to(self.device)
        sig = lamb*tau
        gamma = pyro.sample("gamma", dist.Normal(0, sig)).to(self.device)

        sigma = pyro.sample("sigma", dist.Uniform(0., 10.)).to(self.device)
        mean = (self.linear1(x)*gamma + self.linear2(x)*gamma).squeeze(-1)

        with pyro.plate("data", x.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)
        return mean
    

### TODO: Define a full custom guide


# # should I define a full custom guide?
# # when defining the guide, put constraints on variances
# class BayesianGuide(PyroModule):
#     def __init__(self, torch_model, config, device):
#         super().__init__()

#         self.device = device
#         self.config = config
#         self.model = torch_model

#         self.distributions = self.get_priors()
        
#         self.torch2pyro()


#     def forward(self, x, y=None):
#         mean = self.model(x).squeeze(-1)

#         sigma = pyro.sample("sigma", self.distributions[-2]).to(self.device)

#         return mean

    
#     def torch2pyro(self):
#         pyro.nn.module.to_pyro_module_(self.model)

#         for m in self.model.modules():
#             for name, value in list(m.named_parameters(recurse=False)):
#                 setattr(m, name, PyroSample(self.distributions[0].expand(value.shape).to_event(value.dim())))


#     def render_model(self, model_args):
#         return pyro.render_model(self, model_args, render_distributions=True, filename="guide.png")