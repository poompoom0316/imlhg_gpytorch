import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from scipy.special import gamma
import numpy as np
from sklearn.metrics import r2_score


def correction_factor(new):
    snew = np.pi**0.5/(2**(0.5*new)*gamma((new+1)/2))
    return snew


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y,  likelihood, kernel=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class IMLHGModel():
    def __init__(self, training_iter=50, order=2, use_cuda=False):
        """
        function to estimate Improved Most Likely Heteroscedastic Gaussian Process
        :param training_iter: number of iteration in training
        :param order:
        :param use_cuda: whether use cuda in estimation
        :type training_iter: int
        :type order: int
        :type use_cuda: bool
        """
        print("hogehoge")
        self.training_iter = training_iter
        self.order = order
        self.use_cuda = use_cuda

    def fit(self, train_x, train_y, kernel_mean=None, kernel_noise=None):
        """

        :param train_x: torch.tensor
        :param train_y: torch.tensor
        :param kernel_mean: gpytorch.kernels
        :param kernel_noise: gpytorch.kernels
        :return:
        """
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood, kernel=kernel_mean)
        training_iter = self.training_iter
        order = self.order

        if self.use_cuda:
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            model = model.cuda()
            likelihood = likelihood.cuda()

        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # Step 1: train for mean functions
        # # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            optimizer.step()

        # # estimate training residuals
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(train_x))
            mean = observed_pred.mean
            residuals = ((train_y-mean).abs()**order).detach()

        # Step 2: train for residual values
        likelihood_noise = gpytorch.likelihoods.GaussianLikelihood()
        model_noise = ExactGPModel(train_x, residuals, likelihood_noise, kernel=kernel_noise)

        if self.use_cuda:
            likelihood_noise = likelihood_noise.cuda()
            model_noise = model_noise.cuda()

        optimizer2 = torch.optim.Adam(model_noise.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        model_noise.train()
        likelihood_noise.train()
        # # "Loss" for GPs - the marginal log likelihood
        mll_noise = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_noise, model_noise)

        for i in range(self.training_iter*4):
            # Zero gradients from previous iteration
            optimizer2.zero_grad()
            # Output from model
            output_noise = model_noise(train_x)
            # Calc loss and backprop gradients
            loss2 = -mll_noise(output_noise, residuals)
            loss2.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss2.item(),
                model_noise.covar_module.base_kernel.lengthscale.item(),
                model_noise.likelihood.noise.item()
            ))
            optimizer2.step()

        # Step3. construct Fixed_noise model
        # # estimate error mean
        model_noise.eval()
        likelihood_noise.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood_noise(model_noise(train_x))
            mean_noise = observed_pred.mean

            gi = (torch.tensor(correction_factor(order)) * mean_noise).detach()
            gi[gi<0] = 0
            gi2 = gi**(1/order)

        likelihood_fixed = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=gi2)

        self.model = model
        self.likelihood = likelihood
        self.model_noise = model_noise
        self.likelihood_noise = likelihood_noise
        self.likelihood_fixed = likelihood_fixed

    def predict_mean_std(self, test_x):
        self.model.eval()
        self.likelihood_fixed.eval()

        self.model_noise.eval()
        self.likelihood_noise.eval()

        if self.use_cuda:
            test_x = test_x.cuda()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood_fixed(self.model(test_x))
            mean_test_y = observed_pred.mean.detach()
            std_pred = observed_pred.variance

            observed_noise = self.likelihood_noise(self.model_noise(test_x))
            mean_noise = observed_noise.mean
            mean_noise[mean_noise < 0] = 0
            std_test_y = (std_pred + mean_noise ** (1 / self.order)).detach()

        return mean_test_y, std_test_y

    def predict(self, test_x):
        mean_testy, _ = self.predict_mean_std(test_x)
        return mean_testy

    def predict_confidence_interval(self, test_x):
        mean_testy, std_test_y = self.predict_mean_std(test_x)
        lower, upper = mean_testy - 1.96 * std_test_y, mean_testy + 1.96 * std_test_y
        return lower, upper

