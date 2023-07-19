import tqdm
import numpy as np
from scipy.optimize import minimize
import torch


def optimize_input_quasi_newton(data_point, kept_feature_idx, predict_fn, proximity_weight=0.01, device='cpu'):
    """
    idea from: http://www.bnikolic.co.uk/blog/pytorch/python/2021/02/28/pytorch-scipyminim.html

    Uses quasi-Newton optimization (Sequential Least Squares Programming) to find optimal input alteration for model
    according to:
    loss = predict_fn(y) + gamma * mean squared distance between optimized and original point (excluding fixed values)
    :param data_point:          numpy model input to optimize
    :param kept_feature_idx:    index of feature in data_point to keep, or None for not constraining any feature
                                Can also contain a list of indices to keep
    :param predict_fn:          function of pytorch model to optimize loss for
    :param proximity_weight:    float weight loss factor for proximity to the optimized input
    :return:                    numpy optimized data_point
    """
    data_point = torch.autograd.Variable(torch.from_numpy(data_point.astype('float32')), requires_grad=True).to(device)
    proximity_weight = torch.Tensor([proximity_weight]).to(device)

    def val_and_grad(x):
        pred_loss = predict_fn(x)
        prox_loss = proximity_weight * torch.linalg.vector_norm(data_point - x)
        loss = pred_loss + prox_loss
        loss.backward()
        grad = x.grad
        return loss, grad

    def func(x):
        """scipy needs flattened numpy array with float64, tensorflow tensors with float32"""
        return [vv.cpu().detach().numpy().astype(np.float64).flatten() for vv in
                val_and_grad(torch.tensor(x.reshape([1, -1]), dtype=torch.float32, requires_grad=True))]

    kept_feature_idx = np.where(kept_feature_idx)[0]
    if len(kept_feature_idx) == 0:
        if type(kept_feature_idx) == int:
            constraints = {'type': 'eq', 'fun': lambda x: x[kept_feature_idx] - data_point[:, kept_feature_idx]}
        else:
            from functools import partial
            constraints = []
            for kept_idx in kept_feature_idx:
                constraints.append(
                    {'type': 'eq', 'fun': partial(lambda x, idx: x[idx] - data_point[:, idx], idx=kept_idx)})
    else:
        constraints = ()

    res = minimize(fun=func,
                   x0=data_point.detach().cpu(),
                   jac=True,
                   method='SLSQP',
                   constraints=constraints)
    opt_input = res.x.astype(np.float32).reshape([1, -1])

    return opt_input


def dynamic_synth_data(sample, maskMatrix, model):
    """
    Dynamically generate background "deletion" data for each synthetic data sample by minimizing model output.
    :param sample:                  np.ndarray sample to explain, shape (1, n_features)
    :param maskMatrix:              np.ndarray matrix with features to remove in SHAP sampling process
                                    1 := keep, 0 := optimize/remove
    :param model:                   ml-model to optimize loss for
    :return:                        np.ndarray with synthetic samples, shape maskMatrix.shape

    Example:
    # integrate into SHAP in shap.explainers.kernel @ KernelExplainer.explain(), right before calling self.run()
    if self.dynamic_background:
        from xai.automated_background import dynamic_synth_data
        self.synth_data, self.fnull = dynamic_synth_data(sample=instance.x,
                                                        maskMatrix=self.maskMatrix,
                                                        model=self.full_model)
        self.expected_value = self.fnull
    """
    assert sample.shape[0] == 1, \
        f"Dynamic background implementation can't explain more then one sample at once, but input had shape {sample.shape}"
    assert maskMatrix.shape[1] == sample.shape[1], \
        f"Dynamic background implementation requires sampling of all features (omitted in SHAP when baseline[i] == sample[i]):\n" \
        f"shapes were maskMatrix: {maskMatrix.shape} and sample: {sample.shape}\n" \
        f"Use of np.inf vector as SHAP baseline is recommended"

    # optimize all permutations with 1 kept variable, then aggregate results

    x_hat = []  # contains optimized feature (row) for each leave-one-out combo of varying features (column)
    # Sequential Least Squares Programming
    for kept_idx in tqdm.tqdm(range(sample.shape[1])):
        x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                                 kept_feature_idx=kept_idx,
                                                 predict_fn=model))
    x_hat.append(optimize_input_quasi_newton(data_point=sample,
                                             kept_feature_idx=None,
                                             predict_fn=model))
    x_hat = np.concatenate(x_hat)

    # Find x_tilde by adding x_hat entries for each feature to keep
    def sum_sample(row):
        S = x_hat[:-1][row == True]
        return ((S.sum(axis=0) + x_hat[-1]) / (S.shape[0] + 1)).reshape([1, -1])

    x_tilde_Sc = []
    for mask in maskMatrix:
        x_tilde_Sc.append(sum_sample(mask))
    x_tilde_Sc = np.concatenate(x_tilde_Sc)
    x_tilde = sample.repeat(maskMatrix.shape[0], axis=0) * maskMatrix + x_tilde_Sc * (1 - maskMatrix)

    fnull = model(torch.tensor(x_hat[-1])).unsqueeze(0).detach().numpy()
    return x_tilde, fnull
