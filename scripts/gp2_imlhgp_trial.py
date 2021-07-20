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
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel_module():
    def __init__(self, training_iter=50, order=2, use_cuda=False):
        print("hogehoge")
        self.training_iter = training_iter
        self.order = order
        self.use_cuda = use_cuda

    def fit(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)
        training_iter = 50
        order = 2

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
            lower, upper = observed_pred.confidence_region()
            residuals = ((train_y-mean).abs()**order).detach()

        # Step 2: train for residual values
        likelihood_res = gpytorch.likelihoods.GaussianLikelihood()
        model_res = ExactGPModel(train_x, residuals, likelihood_res)

        if self.use_cuda:
            likelihood_res = likelihood_res.cuda()
            model_res = model_res.cuda()

        optimizer2 = torch.optim.Adam(model_res.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        model_res.train()
        likelihood_res.train()
        # # "Loss" for GPs - the marginal log likelihood
        mll_res = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_res, model_res)

        for i in range(self.training_iter*4):
            # Zero gradients from previous iteration
            optimizer2.zero_grad()
            # Output from model
            output_res = model_res(train_x)
            # Calc loss and backprop gradients
            loss2 = -mll_res(output_res, residuals)
            loss2.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss2.item(),
                model_res.covar_module.base_kernel.lengthscale.item(),
                model_res.likelihood.noise.item()
            ))
            optimizer2.step()

        # Step3. construct Fixed_noise model
        # # estimate error mean
        model_res.eval()
        likelihood_res.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood_res(model_res(train_x))
            mean_res = observed_pred.mean

            gi = (torch.tensor(correction_factor(order)) * mean_res).detach()
            gi[gi<0] = 0
            gi2 = gi**(1/order)

        likelihood_fixed = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=gi2)

        self.model = model
        self.likelihood = likelihood
        self.model_res = model_res
        self.likelihood_res = likelihood_res
        self.likelihood_fixed = likelihood_fixed

    def predict(self, test_x):
        self.model.eval()
        self.likelihood_fixed.eval()

        self.model_res.eval()
        self.likelihood_res.eval()

        if self.use_cuda:
            test_x = test_x.cuda()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood_fixed(self.model(test_x))
            mean_test = observed_pred.mean.detach()
            variance_pred = observed_pred.variance

            observed_res = self.likelihood_res(self.model_res(test_x))
            mean_res = observed_res.mean
            mean_res[mean_res < 0] = 0
            variance_test = (variance_pred + mean_res ** (1 / self.order)).detach()

        return mean_test, variance_test


def main():
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 400)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2
    test_x = torch.linspace(0, 1, 51).cuda()
    test_y = torch.sin(test_x * (2 * math.pi))

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()

    egpm_module = ExactGPModel_module()
    egpm_module.fit(train_x, train_y)
    mean_test, variance_test = egpm_module.predict(test_x)
    lower, upper = mean_test - 1.96*variance_test, mean_test + 1.96*variance_test

    rmse = r2_score(test_y.detach().cpu().numpy(), mean_test.detach().cpu().numpy())

    print(f"RMSE was {rmse:.3f}")

    # Plot training data as black stars
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().cpu().numpy(), mean_test.detach().cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().cpu(), lower.detach().cpu(), upper.detach().cpu(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])


def main2():
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 400)
    # True function is sin(2*pi*x) with Gaussian noise
    train_variance = 0.5+train_x
    train_noise = torch.normal(mean=torch.zeros(400), std=train_variance)
    train_y = torch.sin(train_x * (2 * math.pi)) + train_noise
    test_x = torch.linspace(0, 1, 401)
    test_noise = torch.normal(mean=torch.zeros(401), std=0.5+test_x)
    test_y_mean = torch.sin(test_x * (2 * math.pi))
    test_y = test_y_mean + test_noise

    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()

    egpm_module = ExactGPModel_module()
    egpm_module.fit(train_x, train_y)
    mean_test, variance_test = egpm_module.predict(test_x)
    lower, upper = mean_test - 1.96*variance_test, mean_test + 1.96*variance_test

    rmse = r2_score(test_y_mean.detach().cpu().numpy(), mean_test.detach().cpu().numpy())

    print(f"R2_score was {rmse:.3f}")

    # Plot training data as black stars
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(test_x.detach().cpu().numpy(), test_y.detach().cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().cpu().numpy(), mean_test.detach().cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().cpu(), lower.detach().cpu(), upper.detach().cpu(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

    # plt.plot(test_y.detach().cpu().numpy(), mean_test.detach().cpu().numpy(), "o")


def main3():
    # Training data is 11 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 400)
    # True function is sin(2*pi*x) with Gaussian noise
    train_variance = 0.2+0.3*torch.exp(-30*(train_x-0.5)**2)
    train_noise = torch.normal(mean=torch.zeros(400), std=train_variance)
    train_y = (1 + torch.sin(4*train_x))**1.1 + train_noise
    test_x = torch.linspace(0, 1, 401)
    test_variance = 0.2 + 0.3 * torch.exp(-30 * (test_x - 0.5) ** 2)
    test_noise = torch.normal(mean=torch.zeros(401), std=test_variance)
    test_y_mean = (1 + torch.sin(4*test_x))**1.1
    test_y = test_y_mean + test_noise

    # train_x = train_x.cuda()
    # train_y = train_y.cuda()
    # test_x = test_x.cuda()

    egpm_module = ExactGPModel_module(use_cuda=True)
    egpm_module.fit(train_x, train_y)
    mean_test, variance_test = egpm_module.predict(test_x)
    lower, upper = mean_test - 1.96*variance_test, mean_test + 1.96*variance_test

    rmse = r2_score(test_y_mean.detach().cpu().numpy(), mean_test.detach().cpu().numpy())

    print(f"R2_score was {rmse:.3f}")

    # Plot training data as black stars
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(test_x.detach().cpu().numpy(), test_y.detach().cpu().numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.detach().cpu().numpy(), mean_test.detach().cpu().numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.detach().cpu(), lower.detach().cpu(), upper.detach().cpu(), alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

    # plt.plot(test_y.detach().cpu().numpy(), mean_test.detach().cpu().numpy(), "o")



if __name__ == '__main__':
    main()
    main2()
