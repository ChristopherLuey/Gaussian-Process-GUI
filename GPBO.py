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
        self.x = None

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
        self.x = pd.read_csv(self.file_name, dtype=np.float64, header=None).to_numpy()
        self.x_sample_space = np.empty(self.sample_size).T
        for i in range(self.n):
            _ = np.linspace(self.limits[i][0], self.limits[i][1], self.sample_size)
            self.x_sample_space = np.vstack((self.x_sample_space, _))
        self.x_sample_space = self.x_sample_space[1:]
        dimensions = np.array([self.sample_size] * self.n)

        iterations = 1
        for i in range(iterations):
            inx = torch.from_numpy(self.x[:, :-1])
            iny = torch.from_numpy(self.x[:, -1])
            self.GP_model, self.GP_likelihood = self.train_GP(inx, iny, train_iter=1000)

            self.acq_func = np.zeros(tuple([self.sample_size] * self.n))
            self.mean = np.zeros(tuple([self.sample_size] * self.n))
            self.variance = np.zeros(tuple([self.sample_size] * self.n))
            mesh = {}

            self.GP_model.eval()
            self.GP_likelihood.eval()

            for index, point in enumerate(product(*self.x_sample_space)):
                indices = self.get_indices(index, dimensions)
                mesh[indices] = point
                input_value = torch.from_numpy(np.asarray(point))[None, :]

                model_preds = self.predict_GP(self.GP_model, input_value)

                self.acq_func[indices] = self.calculate_acf(model_preds.mean.numpy(), torch.sqrt(model_preds.variance).numpy(),
                                                  np.min(self.x[-1]))
                self.mean[indices] = model_preds.mean.numpy()[0]
                self.variance[indices] = 1.96 * np.sqrt(model_preds.variance.numpy())

            location = np.unravel_index(self.acq_func.argmax(), self.acq_func.shape)
            te = mesh[location]

            return te

    def query(self, p):
        p = np.array(p)
        input_value = torch.from_numpy(np.asarray(p))[None, :]

        model_preds = self.predict_GP(self.GP_model, input_value)
        print("The mean is {} and the variance is {} at the point {}".format(model_preds.mean.numpy(), 1.96 * np.sqrt(
            model_preds.variance.numpy()), p))

    def graph(self,axis):

        if self.x is not None:
            fig, ax = plt.subplots()
            # ax.plot(x,y, label="1")
            ax.scatter(self.x.T[axis-1], self.x.T[-1], label="3")
            ax.plot(self.x_sample_space[axis-1].T, self.mean[axis-1], label="4")
            ax.fill_between(self.x_sample_space[axis-1].T, (self.mean[axis-1] - self.variance[axis-1]), (self.mean[axis-1] + self.variance[axis-1]), alpha=0.3)
            plt.plot(self.x_sample_space[axis-1].T, self.acq_func[axis-1], label="2")
            plt.show()
            return fig
        return None
