import matplotlib.pyplot as plt
import numpy as np
import torch
import gpytorch
import pandas as pd
from scipy.stats import differential_entropy, norm
from itertools import product


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPToolModel:

    def __init__(self):
        self.sample_size = 100
        self.n = 1
        self.limits = []
        self.file_name = None

    # Expected improvement, maximizing y
    def calculate_acf(self, pred_mean, pred_std, y_max):
        improve = y_max - pred_mean
        z_score = np.divide(improve, pred_std + 1e-9)
        acf = np.multiply(improve, norm.cdf(z_score)) + np.multiply(pred_std, norm.pdf(z_score))
        return max(acf, 0)

    def train_GP(self, train_x, train_y, train_iter):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        likelihood.noise = 10 ** -8
        model = GPModel(train_x, train_y, likelihood)
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(train_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return model, likelihood

    def predict_GP(self, GP_model, x):
        GP_model.eval()
        with torch.no_grad():
            f_preds = GP_model(x)
        return f_preds

    def get_indices(self, index, dimensions):
        indices = []
        for dim_size in reversed(dimensions):
            index, remainder = divmod(index, dim_size)
            indices.insert(0, remainder)
        return tuple(indices)

    def runGPBO(self):
        x = pd.read_csv(self.file_name, dtype=np.float64, header=None).to_numpy()
        x_sample_space = np.empty(self.sample_size).T
        for i in range(self.n):
            _ = np.linspace(self.limits[i][0], self.limits[i][1], self.sample_size)
            x_sample_space = np.vstack((x_sample_space, _))
        x_sample_space = x_sample_space[1:]
        dimensions = np.array([self.sample_size] * self.n)

        iterations = 1
        for i in range(iterations):
            inx = torch.from_numpy(x[:, :-1])
            iny = torch.from_numpy(x[:, -1])
            self.GP_model, self.GP_likelihood = self.train_GP(inx, iny, train_iter=1000)

            acq_func = np.zeros(tuple([self.sample_size] * self.n))
            mean = np.zeros(tuple([self.sample_size] * self.n))
            variance = np.zeros(tuple([self.sample_size] * self.n))
            mesh = {}

            self.GP_model.eval()
            self.GP_likelihood.eval()

            for index, point in enumerate(product(*x_sample_space)):
                indices = self.get_indices(index, dimensions)
                mesh[indices] = point
                input_value = torch.from_numpy(np.asarray(point))[None, :]

                model_preds = self.predict_GP(self.GP_model, input_value)

                acq_func[indices] = self.calculate_acf(model_preds.mean.numpy(), torch.sqrt(model_preds.variance).numpy(),
                                                  np.min(x[-1]))
                mean[indices] = model_preds.mean.numpy()[0]
                variance[indices] = 1.96 * np.sqrt(model_preds.variance.numpy())

            location = np.unravel_index(acq_func.argmax(), acq_func.shape)
            te = mesh[location]

            return te

    def query(self, p):
        p = np.array(p)
        input_value = torch.from_numpy(np.asarray(p))[None, :]

        model_preds = self.predict_GP(self.GP_model, input_value)
        print("The mean is {} and the variance is {} at the point {}".format(model_preds.mean.numpy(), 1.96 * np.sqrt(
            model_preds.variance.numpy()), p))