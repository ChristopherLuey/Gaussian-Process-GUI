import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import gpytorch
import pandas as pd
from scipy.stats import differential_entropy, norm
from itertools import product


# graphing different colors for samples
# Save all of the images based on iteration
# labeling
# Size of the graph different, larger on top

# Resoultion input

# stochastic
# homoscedastic means that the mean
# heteroscedastic different uncertainties

# Uncertainty inputs assigning differe
# Optimization:
# Search space


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
        self.graphed_axis = 0
        self.fig1, self.fig2 = None, None
        self.te = None

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

    def read_file(self):
        self.x = pd.read_csv(self.file_name, dtype=np.float64, header=None).to_numpy()

    def runGPBO(self):
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
            self.te = mesh[location]

            return self.te

    def query(self, p):
        p = np.array(p)
        input_value = torch.from_numpy(np.asarray(p))[None, :]

        model_preds = self.predict_GP(self.GP_model, input_value)
        print("The mean is {} and the variance is {} at the point {}".format(model_preds.mean.numpy(), 1.96 * np.sqrt(
            model_preds.variance.numpy()), p))

        return model_preds.mean.numpy(), 1.96 * np.sqrt(
            model_preds.variance.numpy())

    def graph(self,axis, xlabel, ylabel):
        # if self.graphed_axis == axis:
        #     return self.fig1

        if self.x is not None and self.n == 1:
            self.fig1, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [2, 1]})

            self.ax2.set_xlabel(xlabel)
            self.ax1.set_ylabel(ylabel)

            self.ax1.scatter(self.x.T[axis-1], self.x.T[-1], label="3")
            self.ax1.plot(self.x_sample_space[axis-1].T, self.mean, label="4")
            self.ax1.fill_between(self.x_sample_space[axis-1].T, (self.mean - self.variance), (self.mean + self.variance), alpha=0.3)
            #self.fig2, self.ax2 = plt.subplots()
            self.ax2.plot(self.x_sample_space[axis-1].T, self.acq_func, label="2")
            #plt.show()
            print("success")
            self.graphed_axis = axis
            return self.fig1
        return None

    def new_data(self, checked, y):
        self.sample_size += 1
        print(type(self.te))
        self.x = np.vstack((self.x, np.array([*self.te, y])))
        if checked:
            np.savetxt(self.file_name, self.x, delimiter=",")
