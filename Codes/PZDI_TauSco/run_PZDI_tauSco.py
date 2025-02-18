import matplotlib.pyplot as plt
import os
import time
import math
import jax.numpy as jnp
import numpy as np
import jaxopt
import matplotlib.font_manager as fm
from jax import jit
from stokes2 import surface, generate_stokes_spectrum, generate_true_stokes_spectrum, plot_bfield, plot_bfield_var, read_obs, coef2fld, plot_predictive_distribution, plot_predictive_distribution_phase_subset
from jax.config import config
from jax import jacfwd
from functools import partial

config.update("jax_enable_x64", True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = (16, 12)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size': 24})
font = fm.FontProperties(family='arial')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def generate_spherical_harmonics(lmax, coef_min=-1.0, coef_max=1.0, ibet=0, igam=0):
    """
    Generate a set of spherical harmonic coefficients from a uniform distribution over the interval [coef_min, coef_max].
    The coefficients are scaled by a factor dependent on their degree `l`. The total number of coefficients generated
    depends on `lmax`, `ibet`, and `igam`.

    Parameters:
    ----------
    lmax : int
        Maximum degree of spherical harmonics. Determines the truncation of the expansion.
    coef_min : float, optional
        Minimum value of the uniform distribution for generating coefficients (default is -1.0).
    coef_max : float, optional
        Maximum value of the uniform distribution for generating coefficients (default is 1.0).
    ibet : int, optional
        If set to 1, we add beta coefficients in the spherhical hamonic expansion (independent radial and horizontal poloidal field) (default is 0).
    igam : int, optional
        If set to 1, we add gamma coefficients in the spherical harmonic expansion (toroidal field) (default is 0).

    Returns:
    -------
    np.ndarray
        Array of a randomly generated set of spherical harmonic coefficients, scaled by their degree `l`.
    """
    coef_min = coef_min
    coef_max = coef_max
    nh = lmax * (lmax + 2)
    nf = 1 + ibet + igam
    SHcoef = np.random.uniform(coef_min, coef_max, nh * nf)
    reduce_highl = 1.0
    i = 0
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            scale = 1.0 / (l ** reduce_highl)
            SHcoef[i] = SHcoef[i] * scale
            if ibet == 1:
                SHcoef[i + nh] = SHcoef[i + nh] * scale
            if igam == 1:
                SHcoef[i + nh * (1 + ibet)] = SHcoef[i + nh * (1 + ibet)] * scale
            i = i + 1
    return SHcoef


def standardZDI_objective(coef, area, lat, lon, lmax, ntot,
                                       star_vsini, star_incl, star_limbd,
                                       line_width, line_wave, line_lande,
                                       line_depth,  obs_ntimes, obs_noise, weakfield,
                                       ibet, igam, obs_v,nv,obs_phases, eta=None, ls_weights=None):
    """
    Computes the objective value for a standard ZDI (Zeeman Doppler Imaging) solver.

    This function evaluates the fit quality between the observed Stokes V profiles
    and the modeled profiles generated from the input spherical harmonic coefficients (coef), in terms of a (weighted) least squares objective.
    If `eta` is specified, a regularization term is included.

    Parameters:
    ----------
    coef : np.ndarray
        Array of coefficients for the spherical harmonic expansion of the magnetic field.
    ls_weights : np.ndarray, optional
        Weights for the least-squares residuals, if provided (default is None).
    *Other parameters are defined in the main script.*

    Returns:
    -------
    objective_value : float
        The computed objective value. This is the sum of:
        - The weighted or unweighted least-squares residuals between observed and modeled profiles.
        - Regularization terms, if `eta` is specified.
    """

    # generate synthetic Stokes I and Stokes V profiles along with the magnetic field map
    prfI, prfV, Bfield, obs_phases, obs_v = generate_stokes_spectrum(coef, area=area, lat=lat, lon=lon, lmax=lmax, ntot=ntot,
                                       star_vsini=star_vsini, star_incl=star_incl, star_limbd=star_limbd,
                                       line_width=line_width, line_wave=line_wave, line_lande=line_lande,
                                       line_depth=line_depth,  obs_ntimes=obs_ntimes, obs_noise=0.0, weakfield=weak_field,
                                       ibet=ibet, igam=igam, obs_v=obs_v,nv=nv, obs_phases=obs_phases)

    # calculate least squares objective
    obs_length = obs_ntimes * len(obs_v)
    prfV_vec = jnp.reshape(prfV, obs_length)
    residual_vec = prfV_vec - obs
    residual_vec = jnp.square(residual_vec)

    # add weights if applicable
    if ls_weights is not None:
        residual_vec = ls_weights @ residual_vec
        objective_value = jnp.sum(residual_vec)
    else:
        objective_value = jnp.sum(residual_vec)

    # add regularization if applicable
    if eta is not None:
        i=0
        nh = lmax * (lmax + 2)
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                objective_value += eta*(coef[i]**2)*(l**l_factor)
                if ibet == 1:
                    objective_value += eta*(l**l_factor)*(coef[i + nh]**2)
                if igam == 1:
                    objective_value += eta*(l**l_factor)*(coef[i + nh * (1 + ibet)]**2)
                i=i+1
    return objective_value


@jit
def evaluate_forward_function(coef):
    """
    Evaluates the forward model for generating the synthetic Stokes V profiles.

    This function uses the provided coefficients to compute the synthetic Stokes profiles
    (I and V), the magnetic field map, and associated observables.

    Parameters:
    ----------
    coef : np.ndarray
        Array of coefficients for the spherical harmonic expansion of the magnetic field.

    Returns:
    -------
    np.ndarray
        A reshaped array containing the synthetic Stokes V profiles (prfV) in a flattened
        format of shape (1, obs_length), where obs_length is the number of observations.
    """

    prfI_mph, prfV_mph, Bfield_mph, obs_phases_mph, obs_v_mph = generate_stokes_spectrum(coef, area=area, lat=lat, lon=lon, lmax=lmax, ntot=ntot,
                                                                                         star_vsini=star_vsini, star_incl=star_incl, star_limbd=star_limbd,
                                                                                         line_width=line_width, line_wave=line_wave, line_lande=line_lande,
                                                                                         line_depth=line_depth,  obs_ntimes=obs_ntimes, obs_noise=0.0, weakfield=weak_field,
                                                                                         ibet=ibet, igam=igam, obs_v=obs_v,nv=nv, obs_phases=obs_phases)
    obs_length = obs_ntimes * len(obs_v_mph)
    function_eval = np.reshape(prfV_mph, (1, obs_length))
    return function_eval


def prior_cov_matrix(eta, params=3):
    """
    Computes the prior covariance matrix, which is diagonal with elements 1/(eta*l^l_factor), where l_factor is a global parameter.

    Parameters:
    ----------
    eta : float
        Scaling parameter for the covariance matrix, where each variance term is proportional to 1 / eta.
    params : int, optional
        Number of parameters in the model (default is 3).

    Returns:
    -------
    np.ndarray
        A covariance matrix of shape `(params, params)` that defines the prior distribution over the model parameters.
    """
    covariance_matrix = np.eye(params)
    i = 0
    alpha_const = 1 / eta
    nh = lmax * (lmax + 2)
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            alpha_squared = alpha_const / (l ** l_factor)
            covariance_matrix[i][i] = alpha_squared
            if ibet == 1:
                covariance_matrix[i+nh][i+nh] = alpha_squared
            if igam == 1:
                covariance_matrix[i + nh * (1 + ibet)][i + nh * (1 + ibet)] = alpha_squared
            i = i + 1
    return covariance_matrix


def compute_posterior_params(data_matrix, obs_vector, likelihood_precision, prior_cov_mat, prior_mean_vec):
    """
    Computes the mean vector and covariance matrix of the Gaussian posterior distribution.

    This function calculates the posterior parameters of a Bayesian linear regression model,
    given a Gaussian likelihood and a Gaussian prior. It assumes the model:

        likelihood:    p(y | A, z) = N(Az, Λ⁻¹)
        prior:         p(z) = N(μ₀, Ω⁻¹)
        posterior:     p(z | y, A) = N(μ, Σ)

    Parameters:
    ----------
    data_matrix : np.ndarray
        Design matrix (A) of shape (N, D), where N is the number of observations, and D is the number of features.
    obs_vector : np.ndarray
        Observation vector (y) of shape (N, 1) or (N,).
    likelihood_precision : np.ndarray
        Precision matrix (Λ) of the Gaussian likelihood, containing the measurement noise.
    prior_cov_mat : np.ndarray
        Covariance matrix (Ω⁻¹) of the Gaussian prior on the parameters, shape (D, D).
    prior_mean_vec : np.ndarray
        Mean vector (μ₀) of the Gaussian prior on the parameters, shape (D, 1) or (D,).

    Returns:
    -------
    post_mean : np.ndarray
            Mean vector (μ) of the posterior distribution, shape (D, 1).
    post_cov : np.ndarray
            Covariance matrix (Σ) of the posterior distribution, shape (D, D).
    """
    prior_cov_inv = np.linalg.inv(prior_cov_mat)  # compute the inverse of the prior covariance matrix (Ω)
    data_matrix_t = np.transpose(data_matrix)     # transpose of design matrix
    post_cov = np.linalg.inv(prior_cov_inv + (data_matrix_t @ likelihood_precision @ data_matrix))  # computes the posterior covariance matrix -> Σ = (Ω + AᵀΛA)⁻¹
    post_mean = post_cov @ (prior_cov_inv @ prior_mean_vec + np.expand_dims((data_matrix_t @ likelihood_precision @ obs_vector), axis=1))  # compute the posterior mean vector -> μ = Σ (Ωμ₀ + AᵀΛy)
    return post_mean, post_cov


def compute_predictive_params(data_matrix, likelihood_covariance, posterior_mean_vec, posterior_covariance):
    """
    Computes the parameters of the Gaussian predictive distribution based on the posterior distribution.

    This function calculates:
        - The mean vector and covariance matrix of the predictive distribution.
        - The covariance matrix of the posterior predictive distribution.

    The predictive distribution describes the uncertainty in new observations given the posterior over the parameters,
    assuming the model:

        posterior: p(z | y, A) = N(μ, Σ).
        predictive: p(y* | A*, z) = N(μ_pred, Σ_pred)

    Parameters:
    ----------
    data_matrix : np.ndarray
        Design matrix for new observations (A*), shape (N*, D), where N* is the number of new data points, and D is the number of features.
    likelihood_covariance : np.ndarray
        Covariance matrix of observation noise ((Λ*)⁻¹), shape (N*, N*).
    posterior_mean_vec : np.ndarray
        Mean vector (μ) of the posterior distribution, shape (D, 1) or (D,).
    posterior_covariance : np.ndarray
        Covariance matrix (Σ) of the posterior distribution, shape (D, D).

    Returns:
    -------
    pred_mean : np.ndarray
            Mean vector of the predictive distribution (μ_pred), shape (N*, 1).
    pred_cov : np.ndarray
            Covariance matrix of the predictive distribution (Σ_pred), shape (N*, N*).
    postpred_cov : np.ndarray
            Covariance matrix of the posterior predictive distribution, shape (N*, N*).
    """
    pred_mean = data_matrix @ posterior_mean_vec  # compute mean of the predictive distribution -> μ_pred = A* @ μ
    postpred_cov = data_matrix @ posterior_covariance @ data_matrix.T  # compute covariance matrix of posterior predictive distribution -> μ_postpred = A* @ Σ @ A*.T
    pred_cov = postpred_cov + likelihood_covariance  # compute covariance matrix of predictive distribution -> Σ_pred = A* @ Σ @ A*.T + (Λ*)⁻¹
    return pred_mean, pred_cov, postpred_cov


def compute_neg_log_marginal_likelihood(eta_sqrt, prior_cov_mat_, data_matrix, observation_vector, likelihood_covariance):
    """

    Computes negative marginal log likelihood as a function of sqrt(eta).
    This is the objective function for evidence maximization (empirical Bayes)

    The computation assumes the model:

        likelihood:    p(y | A, z) = N(Az, Λ⁻¹)
        prior:         p(z) = N(μ₀, Ω⁻¹)

    with resulting marginal likelihood:
                       p(y) = N(y; 0, Λ⁻¹ + AΩ⁻¹A^T)

    Parameters:
    ----------
    eta_sqrt: float
        Square root of the hyperparameter eta
    prior_cov_mat_: np.ndarray
        Prior covariance matrix using eta = 1
    data_matrix: np.ndarray
        Design matrix (A) of shape (N, D), where N is the number of observations, and D is the number of features.
    observation_vector: np.ndarray
        Vector containing observations y
    observational_noise: np.ndarray
        Covariance matrix of observation noise

    Returns:
    -------
    -log_results: float
        negative logarithm of quantity proportional to the marginal likelihood
    """

    alpha_mat = (1/eta_sqrt**2) * prior_cov_mat_  # define prior covariance matrix with parameter eta defined by its square root to ensure positive value of eta
    marg_var = likelihood_covariance + data_matrix @ alpha_mat @ data_matrix.T
    sgn, log_const = jnp.linalg.slogdet(marg_var) # not considering (2*np.pi)**dimensions, since it does not impact the maximizing argument
    log_result = -0.5 * (observation_vector.T @ jnp.linalg.inv(marg_var) @ observation_vector) - 0.5*log_const - 0.5*jnp.log(sgn)  # note that marginal_mean is 0
    return -log_result

def compute_log_marginal_likelihood(prior_covariance_matrix, data_matrix, likelihood_covariance, observation_vector):
    """

    Returns logarithm of quantity proportional to the marginal likelihood.
    The implementation is the same as in 'compute_neg_log_marginal_likelihood', but here using a prior covariance matrix defined in terms of eta.
    """
    marg_var = likelihood_covariance + data_matrix @ prior_covariance_matrix @ data_matrix.T
    sgn, log_const = jnp.linalg.slogdet(marg_var)
    log_result = -0.5 * (observation_vector.T @ jnp.linalg.inv(marg_var) @ observation_vector) - 0.5*log_const - 0.5*jnp.log(sgn)
    return log_result


def find_Bfield_distribution(mean_vector, covariance_matrix):
    """
    Finds the magnetic field (Bfield) distribution given the Gaussian posterior
    over the spherical harmonic coefficients.

    This function computes the mean and standard deviation of the resulting Bfield
    distribution at each grid point, based on the provided posterior mean and
    covariance matrix of the coefficients.

    Parameters:
    ----------
    mean_vector : np.ndarray
        Mean vector of the posterior distribution over the spherical harmonic coefficients.
    covariance_matrix : np.ndarray
        Covariance matrix of the posterior distribution over the spherical harmonic coefficients.

    Returns:
    -------
    mean_Bfield : np.ndarray
            Mean magnetic field values at each grid point (shape: 3 x ntot).
    std_Bfield : np.ndarray
            Standard deviation of the magnetic field at each grid point (shape: 3 x ntot).
    """

    # helper functions to compute the radial, meridional and azimuthal magnetic field components
    def coef2fld_jax0(SHcoef):
        theta = 0.5 * math.pi - lat
        phi = lon
        B_field = coef2fld(theta, phi, SHcoef, lmax=lmax, ibet=ibet, igam=igam)
        return B_field[0, :]

    def coef2fld_jax1(SHcoef):
        theta = 0.5 * math.pi - lat
        phi = lon
        B_field = coef2fld(theta, phi, SHcoef, lmax=lmax, ibet=ibet, igam=igam)
        return B_field[1, :]

    def coef2fld_jax2(SHcoef):
        theta = 0.5 * math.pi - lat
        phi = lon
        B_field = coef2fld(theta, phi, SHcoef, lmax=lmax, ibet=ibet, igam=igam)
        return B_field[2, :]

    # compute the gradients of the magnetic field components with respect to the coefficients and evaluate at the posterior mean (linear model coefficients)
    grad_Bfield0, grad_Bfield1, grad_Bfield2 = jacfwd(coef2fld_jax0), jacfwd(coef2fld_jax1), jacfwd(coef2fld_jax2)
    Bc0, Bc1, Bc2 = np.squeeze(grad_Bfield0(mean_vector), axis=2), np.squeeze(grad_Bfield1(mean_vector), axis=2), np.squeeze(grad_Bfield2(mean_vector), axis=2)

    # compute the mean magnetic field components
    mean_Bfield_0, mean_Bfield_1, mean_Bfield_2 = (Bc0@mean_vector).T, (Bc1@mean_vector).T, (Bc2@mean_vector).T
    mean_Bfield = np.vstack((mean_Bfield_0, mean_Bfield_1, mean_Bfield_2))

    # compute the standard deviation for each component at each grid point
    std0, std1, std2 = np.zeros((1, ntot)), np.zeros((1, ntot)), np.zeros((1, ntot))

    for i in range(ntot):
        var0_, var1_, var2_, = Bc0[i,:] @ covariance_matrix @ (Bc0[i,:].T), Bc1[i,:] @ covariance_matrix @ (Bc1[i,:].T), Bc2[i,:] @ covariance_matrix @ (Bc2[i,:].T)
        std0[0, i], std1[0, i], std2[0, i] = np.sqrt(var0_), np.sqrt(var1_), np.sqrt(var2_)

    std_Bfield = np.vstack((std0, std1, std2))

    # return the mean and standard deviation of the magnetic field distribution
    return mean_Bfield, std_Bfield


def SH_energy(SHcoef, lmax, ibet, igam):
    """
    Computes the relative magnetic energy from spherical harmonic coefficients.

    This function calculates the energy distribution of a given set of spherical harmonic (SH) coefficients
    in terms of their poloidal and toroidal components, relative energies for each degree `l`, and the
    modes |m| < l / 2 (here referred to as axisymmetric).

    Parameters:
    ----------
    SHcoef : np.ndarray
        Array of spherical harmonic coefficients.
    lmax : int
        Maximum degree of spherical harmonics.
    ibet : int
        Indicates if beta modes are included (1 to include, 0 otherwise).
    igam : int
        Indicates if gamma modes are included (1 to include, 0 otherwise).

    Returns:
    -------
    El : np.ndarray
            Relative energy as a function of degree `l`.
    Ept : np.ndarray
            Relative energy of poloidal (index 0) and toroidal (index 1) modes.
    El_pol : np.ndarray
            Relative energy of poloidal modes for each degree `l`.
    El_tor : np.ndarray
            Relative energy of toroidal modes for each degree `l`.
    axisymetric : float
            Total relative energy of the axisymmetric modes (|m| < l / 2).
    """

    # calculate relative energy of all harmonic modes
    nh = lmax * (lmax + 2)
    SHenerg = SHcoef ** 2
    SHenerg = SHenerg / np.sum(SHenerg)

    # compute magnetic energy distributions
    i = 0
    El = np.zeros(lmax)
    El_pol = np.zeros(lmax)
    El_tor = np.zeros(lmax)
    Ept = np.zeros(2)
    axisymetric = 0.

    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            El[l - 1] = El[l - 1] + SHenerg[i]
            El_pol[l - 1] = El_pol[l - 1] + SHenerg[i]
            Ept[0] = Ept[0] + SHenerg[i]
            if np.abs(m)<l/2:
                axisymetric += SHenerg[i]
            if ibet == 1:
                El[l - 1] = El[l - 1] + SHenerg[i + nh]
                El_pol[l - 1] = El_pol[l - 1] + SHenerg[i + nh]
                Ept[0] = Ept[0] + SHenerg[i + nh]
                if np.abs(m)<l/2:
                    axisymetric += SHenerg[i+nh]
            if igam == 1:
                El[l - 1] = El[l - 1] + SHenerg[i + (ibet + 1) * nh]
                El_tor[l - 1] = El_tor[l - 1] + SHenerg[i + (ibet + 1) * nh]
                Ept[1] = Ept[1] + SHenerg[i + (ibet + 1) * nh]
                if np.abs(m)<l/2:
                    axisymetric += SHenerg[i + (ibet + 1) * nh]
            i = i + 1
    return (El, Ept, El_pol, El_tor, axisymetric)


def SH_energy_numerical(theta, phi, area, SHcoef, lmax, ibet, igam):
    """
    Computes the magnetic energy distribution numerically for each harmonic component.

    This function computes the magnetic energy numerically (surface integral of B^2) for each harmonic
    component of the spherical harmonics expansion. The energy distribution of a given set of spherical harmonic (SH) coefficients
    is calculated in terms of their poloidal and toroidal components, relative energies for each degree `l`, and the
    modes |m| < l / 2 (here referred to as axisymmetric)

    Parameters:
    ----------
    theta : np.ndarray
        Array of colatitudes for the grid points.
    phi : np.ndarray
        Array of longitudes for the grid points.
    area : np.ndarray
        Surface area corresponding to each grid point.
    SHcoef : np.ndarray
        Array of spherical harmonic coefficients.
    lmax : int
        Maximum degree of spherical harmonics.
    ibet : int
        Indicates if beta modes are included (1 to include, 0 otherwise).
    igam : int
        Indicates if gamma modes are included (1 to include, 0 otherwise).

    Returns:
    -------
    El : np.ndarray
            Relative energy as a function of degree `l`.
    Ept : np.ndarray
            Relative energy of poloidal (index 0) and toroidal (index 1) modes.
    El_pol : np.ndarray
            Relative energy of poloidal modes for each degree `l`.
    El_tor : np.ndarray
            Relative energy of toroidal modes for each degree `l`.
    axisymetric : float
            Total relative energy of the axisymmetric modes (|m| < l / 2).
    """

    nh = lmax * (lmax + 2)
    SHcoef1 = np.zeros(len(SHcoef))
    SHenerg = np.zeros(len(SHcoef))

    # calculate the energy for each spherical harmonic mode
    i = 0
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            SHcoef1[i] = SHcoef[i]
            Bfield = coef2fld(theta, phi, SHcoef1, lmax, ibet=ibet, igam=igam)
            SHenerg[i] = np.sum(area * np.sum(Bfield ** 2, axis=0))
            SHcoef1[i] = 0.0
            i = i + 1

    if ibet == 1:
        i = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                SHcoef1[i + nh] = SHcoef[i + nh]
                Bfield = coef2fld(theta, phi, SHcoef1, lmax, ibet=ibet, igam=igam)
                SHenerg[i + nh] = np.sum(area * np.sum(Bfield ** 2, axis=0))
                SHcoef1[i + nh] = 0.0
                i = i + 1

    if igam == 1:
        i = 0
        for l in range(1, lmax + 1):
            for m in range(-l, l + 1):
                SHcoef1[i + (ibet + 1) * nh] = SHcoef[i + (ibet + 1) * nh]
                Bfield = coef2fld(theta, phi, SHcoef1, lmax, ibet=ibet, igam=igam)
                SHenerg[i + (ibet + 1) * nh] = np.sum(area * np.sum(Bfield ** 2, axis=0))
                SHcoef1[i + (ibet + 1) * nh] = 0.0
                i = i + 1

    # relative energy of all harmonic modes
    SHenerg = SHenerg / np.sum(SHenerg)

    # compute magnetic energy distributions
    i = 0
    El = np.zeros(lmax)
    El_pol = np.zeros(lmax)
    El_tor = np.zeros(lmax)
    Ept = np.zeros(2)
    axisymetric = 0.
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            El[l - 1] = El[l - 1] + SHenerg[i]
            El_pol[l - 1] = El_pol[l - 1] + SHenerg[i]
            Ept[0] = Ept[0] + SHenerg[i]
            if np.abs(m)<l/2:
                axisymetric += SHenerg[i]
            if ibet == 1:
                El[l - 1] = El[l - 1] + SHenerg[i + nh]
                El_pol[l - 1] = El_pol[l - 1] + SHenerg[i + nh]
                Ept[0] = Ept[0] + SHenerg[i + nh]
                if np.abs(m)<l/2:
                    axisymetric += SHenerg[i+nh]
            if igam == 1:
                El[l - 1] = El[l - 1] + SHenerg[i + (ibet + 1) * nh]
                El_tor[l - 1] = El_tor[l - 1] + SHenerg[i + (ibet + 1) * nh]
                Ept[1] = Ept[1] + SHenerg[i + (ibet + 1) * nh]
                if np.abs(m)<l/2:
                    axisymetric += SHenerg[i + (ibet + 1) * nh]
            i = i + 1

    return (El, Ept, El_pol, El_tor, axisymetric)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=1.5)
    plt.setp(bp['whiskers'], color=color, linewidth=1.5)
    plt.setp(bp['caps'], color=color, linewidth=1.5)
    plt.setp(bp['medians'], color=color, linewidth=1.5)
    plt.setp(bp['fliers'], markeredgecolor=color, linewidth=1.5)


def plotRelativeMagneticEnergy(El_array_, El_tor_array_, El_pol_array_, Ept_array_, paxi_array_, filepaths_):
    """
    Plots the distribution of the relative magnetic energy over spherical harmonic modes as violin plots.
    The distribution is obtained numerically from the posterior distribution over the spherical harmonic coefficients.

    This function generates three plots:
        1. Comparison of poloidal and toroidal energy components across harmonic degrees l.
        2. Total energy distribution as a function of harmonic degree.
        3. Energy distribution for poloidal and axisymmetric (|m| < l / 2) components.

    Violin plots are used to visualize the magnetic energy distribution with median and quantile annotations.
    Each plot is saved to a file specified in the `filepaths_` parameter.

    Parameters:
    ----------
    El_array_ : np.ndarray
        Array of total relative energies across harmonic degrees l.
    El_tor_array_ : np.ndarray
        Array of toroidal energy components across harmonic degrees l.
    El_pol_array_ : np.ndarray
        Array of poloidal energy components across harmonic degrees l.
    Ept_array_ : np.ndarray
        Array containing poloidal and toroidal energy components as fractions of the total energy.
    paxi_array_ : np.ndarray
        Array of axisymmetric (|m| < l / 2) energy contributions as fractions of the total energy.
    filepaths_ : list
        List of file paths where the plots will be saved. The list should contain three paths,
        one for each plot.

    Returns:
    -------
    None
    """

    # Plot 1: Distribution of relative magnetic energy over poloidal (red) and toroidal (blue) field components
    fig, ax = plt.subplots()
    data_a = El_tor_array_.tolist()
    data_b = El_pol_array_.tolist()
    ticks = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    posA = np.array(range(len(data_a))) * 2.0 - 0.4
    posB = np.array(range(len(data_b))) * 2.0 + 0.4
    bpl = ax.violinplot(data_a, positions=posA, widths=0.6, showextrema=False, showmeans=False, showmedians=True, quantiles=[[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95]])
    bpr = ax.violinplot(data_b, positions=posB, widths=0.6, showextrema=False, showmeans=False, showmedians=True, quantiles=[[0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95], [0.05, 0.95]])

    for pc in bpl['bodies']:
        pc.set_facecolor('blue')

    for pc in bpr['bodies']:
        pc.set_facecolor('red')

    bpl['cmedians'].set_color('blue')
    bpr['cmedians'].set_color('red')

    bpl['cquantiles'].set_color('blue')
    bpr['cquantiles'].set_color('red')

    plt.plot([], c='blue', label='Toroidal')
    plt.plot([], c='red', label='Poloidal')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)

    plt.tight_layout()
    plt.ylabel("$(E_p, E_t) / E_{tot}$")
    plt.xlabel('l')
    plt.savefig(filepaths_[0], transparent=True, bbox_inches='tight')

    # Plot 2: Total relative magnetic energy distribution across harmonic degrees l
    fig, ax = plt.subplots()
    data_c = El_array_.tolist()
    positions_ = np.arange(1, len(ticks) + 1)
    bp = ax.violinplot(data_c, positions=positions_, showextrema=False, showmeans=False, showmedians=True, quantiles=[[0.05, 0.95]]*len(data_c))

    for pc in bp['bodies']:
        pc.set_facecolor('red')
    bp['cmedians'].set_color('red')
    bp['cquantiles'].set_color('red')

    plt.tight_layout()
    plt.xlabel('l')
    ax.set_xticks(positions_)
    ax.set_xticklabels([f'{i + 1}' for i in range(len(ticks))])
    plt.ylabel("$(E_p + E_t) / E_{tot}$")
    plt.savefig(filepaths_[1], transparent=True, bbox_inches='tight')

    # Plot 3: Relative energy distribution for poloidal and axisymmetric (|m| < l / 2) components
    fig, ax = plt.subplots()
    data_d = Ept_array_[0].tolist()
    data_e = paxi_array_.tolist()

    print(f'EPT mean: {np.mean(data_d)}')
    print(f'EPT 0.50 quantile: {np.quantile(data_d, 0.50)}')
    print(f'EPT 0.05 quantile: {np.quantile(data_d, 0.05)}')
    print(f'EPT 0.95 quantile: {np.quantile(data_d, 0.95)}')
    print(f'AXI mean: {np.mean(data_e)}')
    print(f'AXI 0.50 quantile: {np.quantile(data_e, 0.50)}')
    print(f'AXI 0.05 quantile: {np.quantile(data_e, 0.05)}')
    print(f'AXI 0.95 quantile: {np.quantile(data_e, 0.95)}')

    bp2 = ax.violinplot(data_d, positions=[1], showmeans=False, showmedians=True, showextrema=False, quantiles=[[0.05, 0.95]]) #showmedians=True
    bp3 = ax.violinplot(data_e, positions=[2], showmeans=False, showmedians=True, showextrema=False, quantiles=[[0.05, 0.95]])

    bp2['bodies'][0].set_facecolor('red')
    bp2['cmedians'].set_color('red')
    bp2['cquantiles'].set_color('red')
    bp3['bodies'][0].set_facecolor('red')
    bp3['cmedians'].set_color('red')
    bp3['cquantiles'].set_color('red')
    plt.tight_layout()
    ticks = ['Poloidal', 'Axisymmetric']
    plt.xticks([1,2], ticks)
    plt.ylabel("$(E_p, E_{a}) / E_{tot}$")
    plt.savefig(filepaths_[2], transparent=True, bbox_inches='tight')


weak_field = True                                    # use weakfield assumption to model the line profile response
standardZDI = False                                  # run standard ZDI inversion
probabilisticZDI = True                              # run probabilistic ZDI inversion (must be set to True to run the cases below)
probabilisticZDI_Case1 = True                        # run probabilistic ZDI inversion with fixed hyperparameter eta, referred to as Case 1
probabilisticZDI_Case2 = False                       # run probabilistic ZDI inversion using statistical model with mixture prior over eta, referred to as Case 2
probabilisticZDI_Case3 = False                       # run probabilistic ZDI inversion with two-component mixture prior over the exponent of angular degree l, referred to as Case 3
probabilisticZDI_empiricalBayes = False              # run evidence maximization (empirical Bayes) for hyperparameter selection of eta
plotPredictiveDistribution_Case1 = False             # plot predictive and posterior predictive Stokes V profiles and uncertainty for Case 1
plotPredictiveDistribution_Case2 = False             # plot predictive and posterior predictive Stokes V profiles and uncertainty for Case 2, as well as predictive mean for eta = 16 and eta = 421
plotPredictiveDistribution_Case3 = False             # plot predictive and posterior predictive Stokes V profiles and uncertainty for Case 3
plotSamples_Case2 = False                            # draw and plot samples of magnetic field and uncertainty maps for Case 2
plotSamples_Case3 = False                            # draw and plot samples of magnetic field and uncertainty maps for Case 3
plotMagneticEnergyDistribution_Case1 = False         # plot numerical distribution of magnetic energy spectrum for Case 1
plotMagneticEnergyDistribution_Case2 = False         # plot numerical distribution of magnetic energy spectrum for Case 2
plotMagneticEnergyDistribution_Case3 = False         # plot numerical distribution of magnetic energy spectrum for Case 3

# create output folder
folder_name = 'output'
os.makedirs(folder_name, exist_ok=True)

# parameters for harmonic expansion of magnetic field and surface discretization
ntot= 10173                # number of surface elements
lmax=10                    # maximum angular degree
ibet=1                     # independent radial and horizontal poloidal field
igam=1                     # toroidal field
sh_scale=1                 # randomly generated spherical harmonic coefficients are picked from [-sh_scale,+sh_scale]

random_seed = 100
seed_init = 200
np.random.seed(random_seed)

# set default prior parameters
eta = 100
l_factor = 2

# set stellar parameters for Tau Sco
star_vsini=6.              # projected rotational velocity
star_incl=70.              # stellar inclination
star_limbd=0.3             # linear limb darkening parameter
star_vrad=-0.48            # radial velocity shift

# load observational data
file_obs = 'data/tau_sco/tauSco_obsIV.txt'
obs_ntimes,nv,obs_phases,obs_v,errV,obsI,obsV=read_obs(file_obs)
obsV_noise = np.repeat(errV, np.shape(obsV)[1], axis=0)
obs_length = obs_ntimes * len(obs_v)
prfI_vec_, prfV_vec_ = np.reshape(obsI, obs_length), np.reshape(obsV, obs_length)
obs, obs_vstep = prfV_vec_, 1.0
obs_v=obs_v-star_vrad

# line parameters for tau Sco
line_type=0              # use weakfield line profile model
line_wave=5000.0         # line wavelength in Angstroems
line_lande=1.2           # line Lande factor
line_depth = 0.115       # strength of Gaussian function
line_width = 14.3        # width of Gaussian function (kms^-1)

# discretize surface grid
nlon, xyz, area, lat, lon = surface(ntot)

# generate a random set of spherical harmonic coefficients
SHcoef = generate_spherical_harmonics(lmax=lmax, coef_min=-1.0*sh_scale, coef_max=1.0*sh_scale, ibet=ibet, igam=igam)

# generate initial set of spherical harmonic coefficients for standard ZDI optimizer
np.random.seed(seed_init)
SHcoef_init = generate_spherical_harmonics(lmax=lmax, coef_min=-1.0*sh_scale, coef_max=1.0*sh_scale, ibet=ibet, igam=igam)
num_params = len(SHcoef_init)

print(f'Problem Size: degree = {lmax}, {num_params} parameters')

# construct likelihood covariance matrix and precision matrix, and set prior mean
obs_noise_matrix = np.diag(obsV_noise**2)
obs_noise_precision = np.linalg.inv(obs_noise_matrix)
prior_mean = np.zeros((num_params, 1))

if standardZDI: # generate point estimate of weak field solution using standard ZDI

    # set least-squares (ls) weights to solve weighted ls problem
    ls_weights = obs_noise_precision

    # define objective function
    standardZDI_objective_wrapper_function = partial(standardZDI_objective, eta=eta, area=area, lat=lat, lon=lon, lmax=lmax, ntot=ntot,
                          star_vsini=star_vsini, star_incl=star_incl, star_limbd=star_limbd,
                          line_width=line_width, line_wave=line_wave, line_lande=line_lande,
                          line_depth=line_depth, obs_ntimes=obs_ntimes, obs_noise=0.0, weakfield=weak_field,
                          ibet=ibet, igam=igam, obs_v=obs_v, nv=nv, obs_phases=obs_phases, ls_weights=ls_weights)

    # solve optimization problem numerically
    start = time.time()
    solver = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=standardZDI_objective_wrapper_function, jit=True, tol=1e-15, maxiter=5000)
    SHcoef_lsq_params, SHcoef_solver_state = solver.run(SHcoef_init)
    end = time.time()
    SHcoef_c = np.array(SHcoef_lsq_params)
    print(f'Result: {SHcoef_c}. Convergence: {SHcoef_solver_state}')
    print(f'Elapsed time (standard ZDI inversion): {end-start}')

if probabilisticZDI:     # generate probabilistic weak-field solution

    # generate data matrix A using automatic differentiation
    fwd_grad = jacfwd(evaluate_forward_function)
    A = np.squeeze(fwd_grad(SHcoef), axis=0)

    if probabilisticZDI_Case1:  # generate magnetic field and uncertainty maps using fixed hyperparameter eta

        # set prior hyperparameters and calculate parameters of the Bayesian posterior distribution over spherical harmonic coefficients
        eta, l_factor = 100, 2
        S0 = prior_cov_matrix(eta, params=num_params)
        mN, SN = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

        # calculate the posterior magnetic field and uncertainty maps
        mean_Bfield, mean_std = find_Bfield_distribution(mN, SN)
        average_b = np.mean(np.abs(mean_Bfield), axis=1)
        average_std = np.mean(mean_std, axis=1)
        quantile_bfield = np.quantile(np.abs(mean_Bfield), 0.95, axis=1)
        quantile_std = np.quantile(mean_std, 0.95, axis=1)
        median_bfield = np.quantile(np.abs(mean_Bfield), 0.5, axis=1)
        print(f'L2: Average bfield: {average_b}')
        print(f'L2: Average std: {average_std}')
        print(f'L2: Bfield 95th percentile: {quantile_bfield}')
        print(f'L2: Std 95th percentile: {quantile_std}')
        print(f'L2: Representative error: {average_std / quantile_bfield}')
        print(f'L2: Conservative error: {quantile_std / quantile_bfield}')
        print(f'L2: Average representative error: {average_std / average_b}')
        print(f'L2: Average conservative error: {quantile_std / average_b}')

        # generate magnetic field and Stokes V spectra at the mean of the posterior distribution
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        # plot mean magnetic field and uncertainty maps for Case 1
        plot_bfield_var(lat, lon, mean_Bfield, mean_std, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case1.pdf',
                        max_bfield=True)

    if probabilisticZDI_Case2:  # generate magnetic field and uncertainty maps using an eta-dependent mixture prior with 1000 components

        # define the set of prior hyperparameters of all components
        num_comp = 1000
        eta_array = np.linspace(16, 421, num_comp).tolist()
        l_factor = 2

        # calculate the posterior magnetic field and uncertainty maps
        bfield_means = np.zeros(((num_comp, 3, ntot)))
        bfield_stds = np.zeros(((num_comp, 3, ntot)))
        for j in range(num_comp):
            # calculate posterior distribution over spherical-harmonic coefficients for current component
            eta = eta_array[j]
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_eta, SN_eta = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

            # calculate parameters of posterior magnetic field for current component
            mean_Bfield_eta, mean_std_eta = find_Bfield_distribution(mN_eta, SN_eta)
            if j == 0:
                mean_Bfield_eta1, mean_std_eta1 = mean_Bfield_eta, mean_std_eta
            bfield_means[j, :, :], bfield_stds[j, :, :] = mean_Bfield_eta, mean_std_eta

        # calculate parameters of posterior magnetic field of mixture distribution
        mean_Bfield = np.mean(bfield_means, axis=0)
        m_squared = bfield_means ** 2 + bfield_stds ** 2
        mean_std = np.sqrt(np.mean(m_squared, axis=0) - mean_Bfield ** 2)
        average_b = np.mean(np.abs(mean_Bfield), axis=1)
        average_std = np.mean(mean_std, axis=1)
        quantile_bfield = np.quantile(np.abs(mean_Bfield), 0.95, axis=1)
        quantile_std = np.quantile(mean_std, 0.95, axis=1)
        print(f'Comp: Average bfield: {average_b}')
        print(f'Comp: Average std: {average_std}')
        print(f'Comp: Bfield 95th percentile: {quantile_bfield}')
        print(f'Comp: Std 95th percentile: {quantile_std}')
        print(f'Comp: Representative error: {average_std / quantile_bfield}')
        print(f'Comp: Conservative error: {quantile_std / quantile_bfield}')
        print(f'Comp: Average representative error: {average_std / average_b}')
        print(f'Comp: Average conservative error: {quantile_std / average_b}')

        # generate magnetic field and Stokes V spectra at the mean of the posterior distribution
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN_eta, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        # plot mean magnetic field and uncertainty maps for Case 2, and for independent fits using eta=16 and eta=421
        plot_bfield_var(lat, lon, mean_Bfield_eta1, mean_std_eta1, prfI_, prfV_, star_incl, obs_v, obs_ntimes,
                        obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case2_eta16.pdf')
        plot_bfield_var(lat, lon, mean_Bfield_eta, mean_std_eta, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case2_eta421.pdf')
        plot_bfield_var(lat, lon, mean_Bfield, mean_std, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case2.pdf', max_bfield=True)

    if probabilisticZDI_Case3: # generate magnetic field and uncertainty maps using a mixture prior with 2 components, corresponding to distinct exponents of l (1 or 2)
        num_comp = 2
        bfield_means = np.zeros(((num_comp, 3, ntot)))
        bfield_stds = np.zeros(((num_comp, 3, ntot)))

        # calculate the posterior magnetic field parameters for component 1
        l_factor = 2
        eta = 100
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_l2, SN_l2 = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
        mean_Bfield_l2, mean_std_l2 = find_Bfield_distribution(mN_l2, SN_l2)
        bfield_means[0, :, :], bfield_stds[0, :, :] = mean_Bfield_l2, mean_std_l2

        # calculate the posterior magnetic field parameters for component 2
        l_factor = 1
        eta = 275.47
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_l1, SN_l1 = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
        mean_Bfield_l1, mean_std_l1 = find_Bfield_distribution(mN_l1, SN_l1)
        bfield_means[1, :, :], bfield_stds[1, :, :] = mean_Bfield_l1, mean_std_l1

        # calculate parameters of mixture posterior magnetic field
        mean_Bfield = np.mean(bfield_means, axis=0)
        m_squared = bfield_means ** 2 + bfield_stds ** 2
        mean_std = np.sqrt(np.mean(m_squared, axis=0) - mean_Bfield ** 2)

        average_b = np.mean(np.abs(mean_Bfield), axis=1)
        average_std = np.mean(mean_std, axis=1)
        quantile_bfield = np.quantile(np.abs(mean_Bfield), 0.95, axis=1)
        quantile_std = np.quantile(mean_std, 0.95, axis=1)
        print(f'L1L2: Average bfield: {average_b}')
        print(f'L1L2: Average std: {average_std}')
        print(f'L1L2: Bfield 95th percentile: {quantile_bfield}')
        print(f'L1L2: Std 95th percentile: {quantile_std}')
        print(f'L1L2: Representative error: {average_std / quantile_bfield}')
        print(f'L1L2: Conservative error: {quantile_std / quantile_bfield}')
        print(f'L1L2: Average representative error: {average_std / average_b}')
        print(f'L1L2: Average conservative error: {quantile_std / average_b}')

        # generate magnetic field and Stokes V spectra at the mean of the posterior distribution
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN_l2, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        # plot mean magnetic field and uncertainty maps for Case 3
        plot_bfield_var(lat, lon, mean_Bfield_l2, mean_std_l2, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case3_comp1.pdf')
        plot_bfield_var(lat, lon, mean_Bfield_l1, mean_std_l1, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case3_comp2.pdf')
        plot_bfield_var(lat, lon, mean_Bfield, mean_std, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                        obs_phases, filename='output\magnetic_field_map_case3.pdf', max_bfield=True)


    if probabilisticZDI_empiricalBayes: # maximize the marginal likelihood to select the hyperparameter eta
        # set covariance matrix with eta=1 (to ensure we optimise for positive eta)
        eta_indep_prior_cov_matrix = prior_cov_matrix(1.0, params=num_params)

        # set up objective function to maximize marginal likelihood
        log_marg_lik_wrapper_function = partial(compute_neg_log_marginal_likelihood, prior_cov_mat_=eta_indep_prior_cov_matrix, data_matrix=A,
                                       observation_vector=jnp.asarray(obs), likelihood_covariance=obs_noise_matrix)

        # run numerical solver
        solver = jaxopt.ScipyMinimize(method='L-BFGS-B', fun=log_marg_lik_wrapper_function, jit=True, tol=1e-7, maxiter=5000)
        eta_init = 1000  # initial value of eta for numerical solver
        max_marg_lik_param, max_marg_lik_state = solver.run(jnp.sqrt(eta_init))
        sqrt_eta_hat = np.array(max_marg_lik_param)
        eta_hat = sqrt_eta_hat ** 2
        print(f'Result: {eta_hat}')
        print(f'Convergence: {max_marg_lik_state}')


    if plotPredictiveDistribution_Case1:  # plot prediction uncertainty for Case 1

        # set prior hyperparameters
        l_factor = 2
        eta = 100

        # calculate parameters of posterior distribution
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_post, SN_post = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
        marginal_likelihood_ = compute_log_marginal_likelihood(prior_covariance_matrix=S0, data_matrix=A, likelihood_covariance=obs_noise_matrix, observation_vector=jnp.asarray(obs))

        # calculate parameters of the predictive and posterior predictive distribution
        mN_pred, SN_pred, SN_postpred = compute_predictive_params(data_matrix=A, likelihood_covariance=obs_noise_matrix, posterior_mean_vec=mN_post, posterior_covariance=SN_post)

        # plot predicted (mean) Stokes V with prediction uncertainty
        pred_std = np.sqrt(np.diag(SN_pred))
        pred_std_vec = pred_std.reshape(-1, 1)
        post_std = np.sqrt(np.diag(SN_postpred))
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN_post, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        calculated_mean_deviation = np.sqrt(np.sum((np.squeeze(mN_pred)-prfV_vec_)**2 / obs_length))
        calculated_mean_deviation_std = np.sqrt(np.sum((np.squeeze(pred_std_vec)) ** 2 / obs_length))
        print(f'Mean deviation: {calculated_mean_deviation}')
        print(f'Mean marginal std: {np.mean(pred_std_vec)}')
        pred_std = pred_std.reshape((obs_ntimes,nv))
        post_std = post_std.reshape((obs_ntimes,nv))
        plot_predictive_distribution(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep, obs_phases,
                                     errV, obsV, obsI, pred_std, post_std,
                                     filename='output\predictive_undertainty_case1.pdf')

        plot_predictive_distribution_phase_subset(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                                         obs_phases, errV,
                                         obsV, obsI, pred_std, post_std,
                                         filename='output\predictive_undertainty_case1_subset.pdf')

    if plotPredictiveDistribution_Case2:  # plot prediction uncertainty for Case 2

        # calculate parameters of the predictive and posterior predictive distribution
        num_components = 1000
        eta_range = np.linspace(16, 421, num_components)
        mN_pred = np.zeros(obs_length)
        var_intermediate = np.zeros(obs_length)
        var_intermediate_post = np.zeros(obs_length)
        coef_pred = np.zeros(num_params)

        for comp in range(len(eta_range)):
            l_factor = 2
            eta = eta_range[comp]
            print(f'Component {comp}, eta {eta}')

            # calculate posterior distribution for current component
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_post, SN_post = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

            # calculate predictive distribution for current component
            mN_pred_comp, SN_pred_comp, SN_predpost = compute_predictive_params(data_matrix=A,
                                                                      likelihood_covariance=obs_noise_matrix,
                                                                      posterior_mean_vec=mN_post,
                                                                      posterior_covariance=SN_post)

            # intermediate computations for mean and variance of mixture distribution
            mN_pred += np.squeeze(mN_pred_comp)
            coef_pred += np.squeeze(mN_post)
            var_intermediate += np.squeeze(mN_pred_comp) ** 2 + np.diag(SN_pred_comp)
            var_intermediate_post += np.squeeze(mN_pred_comp) ** 2 + np.diag(SN_predpost)

        # calculate mean and variance of mixture distribution (predictive and posterior)
        coef_pred /= num_components
        mN_pred /= num_components
        var_intermediate /= num_components
        var_intermediate_post /= num_components
        pred_std = np.sqrt(var_intermediate - np.squeeze(mN_pred) ** 2)
        post_std = np.sqrt(var_intermediate_post - np.squeeze(mN_pred) ** 2)
        pred_std_vec = pred_std.reshape(-1, 1)
        print(f'Mean marginal std: {np.mean(pred_std_vec)}')

        # calculate predicted (mean) Stokes V profile
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(coef_pred, area=area, lat=lat,
                                                                                lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        # calculate mean deviation of predicted Stokes V profile
        calculated_mean_deviation = np.sqrt(np.sum((np.squeeze(mN_pred)-prfV_vec_)**2 / obs_length))
        calculated_mean_deviation_std = np.sqrt(np.sum((np.squeeze(pred_std_vec))**2 / obs_length))
        print(f'Mean deviation: {calculated_mean_deviation}')

        # plot Stokes V with prediction uncertainty
        pred_std, post_std = pred_std.reshape((obs_ntimes,nv)), post_std.reshape((obs_ntimes,nv))
        plot_predictive_distribution(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                                     obs_phases,
                                     errV, obsV, obsI, pred_std, post_std,
                                     filename='output\predictive_undertainty_case2.pdf')

        plot_predictive_distribution_phase_subset(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                                         obs_phases, errV,
                                         obsV, obsI, pred_std, post_std,
                                         filename='output\predictive_undertainty_case2_subset.pdf')

        # compare mean fits of components with eta=16 and eta=421
        num_samples_ = 2
        sample_spectra = np.zeros((obs_length, 2))

        # calculate posterior distribution for eta=16
        l_factor = 2
        eta = 16
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_post, SN_post = compute_posterior_params(data_matrix=A, obs_vector=obs,
                                                    likelihood_precision=obs_noise_precision, prior_cov_mat=S0,
                                                    prior_mean_vec=prior_mean)

        # generate Stokes V predictions
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN_post, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        sample_spectra[:, 1] = prfV_.reshape((obs_length,))
        calculated_mean_deviation = np.sqrt(np.sum((sample_spectra[:,1] - prfV_vec_) ** 2 / obs_length))
        print(f'Mean deviation tau=16: {calculated_mean_deviation}')

        # calculate posterior distribution for eta=421
        eta = 421
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_post, SN_post = compute_posterior_params(data_matrix=A, obs_vector=obs,
                                                    likelihood_precision=obs_noise_precision, prior_cov_mat=S0,
                                                    prior_mean_vec=prior_mean)

        # generate Stokes V predictions
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(mN_post, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        sample_spectra[:, 0] = prfV_.reshape((obs_length,))
        calculated_mean_deviation = np.sqrt(np.sum((sample_spectra[:,
                                                    0] - prfV_vec_) ** 2 / obs_length))
        print(f'Mean deviation tau=421: {calculated_mean_deviation}')

        # plot the predicted (mean) Stokes V profiles for independent models using the respective values of eta, for a subset of phases
        plot_predictive_distribution_phase_subset(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep,
                                         obs_phases, errV,
                                         obsV, obsI, prediction_uncertainty=None, samples=sample_spectra,
                                         num_samples=num_samples_,
                                         filename='output\predictive_mean_eta16_eta421.pdf')


    if plotPredictiveDistribution_Case3:  # plot prediction uncertainty for Case 3

        # calculate parameters of predictive distribution for component 1
        l_factor=2
        eta = 100.

        # calculate posterior distribution
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_post1, SN_post1 = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

        # calculate predictive distribution
        mN_pred1, SN_pred1, SN_postpred1 = compute_predictive_params(data_matrix=A,
                                                                     likelihood_covariance=obs_noise_matrix,
                                                                     posterior_mean_vec=mN_post1,
                                                                     posterior_covariance=SN_post1)

        # calculate parameters of predictive distribution for component 2
        l_factor=1
        eta = 275.47

        # calculate posterior distribution
        S0 = prior_cov_matrix(eta, params=num_params)
        mN_post2, SN_post2 = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

        # calculate predictive distribution
        mN_pred2, SN_pred2, SN_postpred2 = compute_predictive_params(data_matrix=A,
                                                                     likelihood_covariance=obs_noise_matrix,
                                                                     posterior_mean_vec=mN_post2,
                                                                     posterior_covariance=SN_post2)

        # calculate parameters of mixture posterior using equal mixture weights
        mN_pred = (mN_pred1+mN_pred2)/2
        var_intermediate = ((np.squeeze(mN_pred1)**2 + np.diag(SN_pred1)) + (np.squeeze(mN_pred2)**2 + np.diag(SN_pred2)))/2
        pred_std = np.sqrt(var_intermediate - np.squeeze(mN_pred)**2)
        pred_std_vec = pred_std.reshape(-1, 1)
        print(f'Mean marginal std: {np.mean(pred_std_vec)}')
        var_intermediate_post = ((np.squeeze(mN_pred1) ** 2 + np.diag(SN_postpred1)) + (
                    np.squeeze(mN_pred2) ** 2 + np.diag(SN_postpred2))) / 2
        post_std = np.sqrt(var_intermediate_post - np.squeeze(mN_pred)**2)

        coef_pred = (mN_post2+mN_post1)/2
        prfI_, prfV_, Bfield, obs_phases, obs_v = generate_true_stokes_spectrum(coef_pred, area=area, lat=lat, lon=lon,
                                                                                lmax=lmax,
                                                                                ntot=ntot,
                                                                                star_vsini=star_vsini,
                                                                                star_incl=star_incl,
                                                                                star_limbd=star_limbd,
                                                                                line_width=line_width,
                                                                                line_wave=line_wave,
                                                                                line_lande=line_lande,
                                                                                line_depth=line_depth,
                                                                                obs_ntimes=obs_ntimes,
                                                                                obs_noise=0.0, weakfield=weak_field,
                                                                                ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                nv=nv, obs_phases=obs_phases)

        calculated_mean_deviation = np.sqrt(np.sum((np.squeeze(mN_pred)-prfV_vec_)**2/obs_length))
        calculated_mean_deviation_std = np.sqrt(np.sum((np.squeeze(pred_std_vec)) ** 2/obs_length))
        print(f'L1L2: Mean deviation: {calculated_mean_deviation}')

        # plot Stokes V with prediction uncertainty
        pred_std = pred_std.reshape((obs_ntimes,nv))
        post_std = post_std.reshape((obs_ntimes,nv))
        plot_predictive_distribution(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes, obs_vstep, obs_phases,
                                     errV, obsV, obsI, pred_std, post_std,
                                     filename='output\predictive_undertainty_case3.pdf')

        plot_predictive_distribution_phase_subset(lat, lon, Bfield, prfI_, prfV_, star_incl, obs_v, obs_ntimes,
                                                  obs_vstep,
                                                  obs_phases, errV,
                                                  obsV, obsI, pred_std, post_std,
                                                  filename='output\predictive_undertainty_case3_subset.pdf')


    if plotSamples_Case2: # plot samples from posterior distribution based on statistical model with a mixture prior with 1000 eta-dependent components
        num_samples = 10  # we draw this number of samples twice
        num_components = 1000
        eta_range = np.linspace(16, 421, num_components)

        # draw component indices randomly
        c = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        c2 = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        for comp in range(len(c)):
            l_factor = 2
            eta = eta_range[c[comp]-1]
            print(f'Sample {comp}, component {c[comp]}, eta {eta}, eta2 {eta_range[c2[comp]-1]}')

            # draw sample from the current component
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs1 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs1 = (generated_coefs1.T)[:,0]
            prfI_, prfV_, Bfield1, obs_phases, obs_v = generate_true_stokes_spectrum(generated_coefs1, area=area, lat=lat, lon=lon,
                                                                                    lmax=lmax,
                                                                                    ntot=ntot,
                                                                                    star_vsini=star_vsini,
                                                                                    star_incl=star_incl,
                                                                                    star_limbd=star_limbd,
                                                                                    line_width=line_width,
                                                                                    line_wave=line_wave,
                                                                                    line_lande=line_lande,
                                                                                    line_depth=line_depth,
                                                                                    obs_ntimes=obs_ntimes,
                                                                                    obs_noise=0.0, weakfield=weak_field,
                                                                                    ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                    nv=nv, obs_phases=obs_phases)

            # draw sample from the second component
            l_factor = 2
            eta = eta_range[c2[comp]-1]
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs2 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs2 = (generated_coefs2.T)[:,0]
            prfI_, prfV_, Bfield2, obs_phases, obs_v = generate_true_stokes_spectrum(generated_coefs2, area=area, lat=lat, lon=lon,
                                                                                    lmax=lmax,
                                                                                    ntot=ntot,
                                                                                    star_vsini=star_vsini,
                                                                                    star_incl=star_incl,
                                                                                    star_limbd=star_limbd,
                                                                                    line_width=line_width,
                                                                                    line_wave=line_wave,
                                                                                    line_lande=line_lande,
                                                                                    line_depth=line_depth,
                                                                                    obs_ntimes=obs_ntimes,
                                                                                    obs_noise=0.0, weakfield=weak_field,
                                                                                    ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                    nv=nv, obs_phases=obs_phases)
            # plot samples
            plot_bfield(lat, lon, Bfield1, Bfield2, prfI_, prfV_, star_incl, obs_v, obs_ntimes,
                            obs_vstep,
                            obs_phases, filename=f'output\sample_{comp}_case2.pdf')


    if plotSamples_Case3: # plot samples from posterior distribution based on statistical model with a mixture prior with two components
        num_samples = 10  # we draw this number of samples twice
        num_components = 2

        # draw component indices randomly
        c = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        c2 = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        for comp in range(len(c)):
            print(f'Sample {comp}, component {c[comp]}')
            l_factor = c[comp]
            if c[comp] == 2:
                eta = 100.
            elif c[comp] == 1:
                eta = 275.47

            # draw sample from current component
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs1 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs1 = (generated_coefs1.T)[:,0]
            prfI_, prfV_, Bfield1, obs_phases, obs_v = generate_true_stokes_spectrum(generated_coefs1, area=area, lat=lat, lon=lon,
                                                                                    lmax=lmax,
                                                                                    ntot=ntot,
                                                                                    star_vsini=star_vsini,
                                                                                    star_incl=star_incl,
                                                                                    star_limbd=star_limbd,
                                                                                    line_width=line_width,
                                                                                    line_wave=line_wave,
                                                                                    line_lande=line_lande,
                                                                                    line_depth=line_depth,
                                                                                    obs_ntimes=obs_ntimes,
                                                                                    obs_noise=0.0, weakfield=weak_field,
                                                                                    ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                    nv=nv, obs_phases=obs_phases)

            l_factor = c2[comp]
            if c2[comp] == 2:
                eta = 100.
            elif c2[comp] == 1:
                eta = 275.47

            # draw sample from second component
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs2 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs2 = (generated_coefs2.T)[:,0]
            prfI_, prfV_, Bfield2, obs_phases, obs_v = generate_true_stokes_spectrum(generated_coefs2, area=area, lat=lat, lon=lon,
                                                                                    lmax=lmax,
                                                                                    ntot=ntot,
                                                                                    star_vsini=star_vsini,
                                                                                    star_incl=star_incl,
                                                                                    star_limbd=star_limbd,
                                                                                    line_width=line_width,
                                                                                    line_wave=line_wave,
                                                                                    line_lande=line_lande,
                                                                                    line_depth=line_depth,
                                                                                    obs_ntimes=obs_ntimes,
                                                                                    obs_noise=0.0, weakfield=weak_field,
                                                                                    ibet=ibet, igam=igam, obs_v=obs_v,
                                                                                    nv=nv, obs_phases=obs_phases)

            # plot samples
            plot_bfield(lat, lon, Bfield1, Bfield2, prfI_, prfV_, star_incl, obs_v, obs_ntimes,
                            obs_vstep,
                            obs_phases, filename=f'output\sample_{comp}_case3.pdf')


    if plotMagneticEnergyDistribution_Case1:  # plot numerical distribution of magnetic energy spectrum for Case 1

        start_time = time.time()
        num_samples = 1000  # we draw this number of samples twice
        grid_points = ntot

        theta = 0.5 * math.pi - lat
        phi = lon

        l_factor = 2
        eta = 100.
        El_array = np.zeros((lmax, num_samples*2))
        El_pol_array = np.zeros((lmax, num_samples * 2))
        El_tor_array = np.zeros((lmax, num_samples * 2))
        Ept_array = np.zeros((2, num_samples*2))
        paxi_array = np.zeros(num_samples*2)
        cnt = 0

        L2_samples = None

        # draw samples from the posterior distribution
        for i in range(num_samples):
            print(f'Sample {i}')
            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs2 = np.random.multivariate_normal(mN_comp[:, 0], SN_comp, size=1)
            generated_coefs2 = (generated_coefs2.T)[:, 0]

            generated_coefs1= np.random.multivariate_normal(mN_comp[:, 0], SN_comp, size=1)
            generated_coefs1 = (generated_coefs1.T)[:, 0]

            if L2_samples is None:
                L2_samples = np.vstack((generated_coefs1, generated_coefs2))
            else:
                L2_samples = np.vstack((L2_samples, generated_coefs1, generated_coefs2))

            # calculate magnetic energy for the current samples
            El1, Ept1, El_pol1, El_tor1, axi1 = SH_energy(generated_coefs1, lmax, ibet, igam) #SH_energy_numerical(theta, phi, area, generated_coefs1, lmax, ibet, igam) #SH_energy(generated_coefs1, lmax, ibet, igam) # El: relative energy as a function of l, Ept: Relative energy of poloidal and toroidal modes
            El_array[:,cnt], Ept_array[:,cnt], El_pol_array[:,cnt], El_tor_array[:,cnt] = El1, Ept1, El_pol1, El_tor1
            paxi_array[cnt] = axi1
            El2, Ept2, El_pol2, El_tor2, axi2 = SH_energy(generated_coefs2, lmax, ibet, igam) #SH_energy_numerical(theta, phi, area, generated_coefs2, lmax, ibet, igam)# SH_energy(generated_coefs2, lmax, ibet, igam)
            El_array[:,cnt+1], Ept_array[:,cnt+1], El_pol_array[:,cnt+1], El_tor_array[:,cnt+1] = El2, Ept2, El_pol2, El_tor2
            paxi_array[cnt+1] = axi2
            cnt+=2

        # plot numerical distribution as violin plots
        filepaths_violin = ['output\el_array1_violin_case1.pdf', 'output\el_array2_violin_case1.pdf', 'output\ept_array_violin_case1.pdf']
        plotRelativeMagneticEnergy(El_array, El_tor_array, El_pol_array, Ept_array, paxi_array, filepaths_violin)


    if plotMagneticEnergyDistribution_Case2:  # plot numerical distribution of magnetic energy spectrum for Case 2
        start_time = time.time()
        num_samples = 100000  # we draw this number of samples twice
        num_components = 1000
        eta_range = np.linspace(16, 421, 1000)
        c = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        c2 = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        El_array = np.zeros((lmax, num_samples*2))
        El_pol_array = np.zeros((lmax, num_samples * 2))
        El_tor_array = np.zeros((lmax, num_samples * 2))
        Ept_array = np.zeros((2, num_samples*2))
        paxi_array = np.zeros(num_samples*2)
        cnt = 0
        comp_samples = None

        # draw samples from the posterior distribution
        for comp in range(len(c)):
            l_factor = 2
            eta = eta_range[c[comp]-1]
            print(f'Sample {comp}, component {c[comp]}, eta {eta}, eta2 {eta_range[c2[comp]-1]}')

            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs1 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs1 = (generated_coefs1.T)[:,0]

            l_factor = 2
            eta = eta_range[c2[comp]-1]

            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs2 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs2 = (generated_coefs2.T)[:,0]

            if comp_samples is None:
                comp_samples = np.vstack((generated_coefs1, generated_coefs2))
            else:
                comp_samples = np.vstack((comp_samples, generated_coefs1, generated_coefs2))

            # calculate magnetic energy for the current samples
            El1, Ept1, El_pol1, El_tor1, axi1 = SH_energy(generated_coefs1, lmax, ibet,
                                                          igam)  # El: relative energy as a function of l, Ept: Relative energy of poloidal and toroidal modes
            El_array[:, cnt], Ept_array[:, cnt], El_pol_array[:, cnt], El_tor_array[:,
                                                                       cnt] = El1, Ept1, El_pol1, El_tor1
            paxi_array[cnt] = axi1
            El2, Ept2, El_pol2, El_tor2, axi2 = SH_energy(generated_coefs2, lmax, ibet, igam)
            El_array[:, cnt + 1], Ept_array[:, cnt + 1], El_pol_array[:, cnt + 1], El_tor_array[:,
                                                                                   cnt + 1] = El2, Ept2, El_pol2, El_tor2
            paxi_array[cnt + 1] = axi2
            cnt += 2

        # plot numerical distribution as violin plots
        filepaths_violin = ['output\el_array1_violin_case2.pdf', 'output\el_array2_violin_case2.pdf', 'output\ept_array_violin_case2.pdf']
        plotRelativeMagneticEnergy(El_array, El_tor_array, El_pol_array, Ept_array, paxi_array, filepaths_violin)


    if plotMagneticEnergyDistribution_Case3:  # plot numerical distribution of magnetic energy spectrum for Case 3
        num_samples = 10000  # we draw this number of samples twice
        num_components = 2
        c = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        c2 = np.random.randint(low=1, high=num_components+1, size=num_samples, dtype=int)
        El_array = np.zeros((lmax, num_samples*2))
        El_pol_array = np.zeros((lmax, num_samples * 2))
        El_tor_array = np.zeros((lmax, num_samples * 2))
        Ept_array = np.zeros((2, num_samples*2))
        paxi_array = np.zeros(num_samples*2)
        cnt = 0
        L1L2_samples = None

        # draw samples from the posterior distribution
        for comp in range(len(c)):
            print(f'Sample {comp}, component {c[comp]}')
            l_factor = c[comp]
            if c[comp] == 2:
                eta = 100.
            elif c[comp] == 1:
                eta = 275.47

            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)

            generated_coefs1 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs1 = (generated_coefs1.T)[:,0]

            l_factor = c2[comp]
            if c2[comp] == 2:
                eta = 100.
            elif c2[comp] == 1:
                eta = 275.47

            S0 = prior_cov_matrix(eta, params=num_params)
            mN_comp, SN_comp = compute_posterior_params(data_matrix=A, obs_vector=obs, likelihood_precision=obs_noise_precision, prior_cov_mat=S0, prior_mean_vec=prior_mean)
            generated_coefs2 = np.random.multivariate_normal(mN_comp[:,0], SN_comp, size=1)
            generated_coefs2 = (generated_coefs2.T)[:,0]

            if L1L2_samples is None:
                L1L2_samples = np.vstack((generated_coefs1, generated_coefs2))
            else:
                L1L2_samples = np.vstack((L1L2_samples, generated_coefs1, generated_coefs2))

            # calculate magnetic energy for the current samples
            El1, Ept1, El_pol1, El_tor1, axi1 = SH_energy(generated_coefs1, lmax, ibet,
                                                          igam)  # El: relative energy as a function of l, Ept: Relative energy of poloidal and toroidal modes
            El_array[:, cnt], Ept_array[:, cnt], El_pol_array[:, cnt], El_tor_array[:,
                                                                       cnt] = El1, Ept1, El_pol1, El_tor1
            paxi_array[cnt] = axi1
            El2, Ept2, El_pol2, El_tor2, axi2 = SH_energy(generated_coefs2, lmax, ibet, igam)
            El_array[:, cnt + 1], Ept_array[:, cnt + 1], El_pol_array[:, cnt + 1], El_tor_array[:,
                                                                                   cnt + 1] = El2, Ept2, El_pol2, El_tor2
            paxi_array[cnt + 1] = axi2
            cnt += 2

        # plot numerical distribution as violin plots
        filepaths_violin = ['output\el_array1_violin_case3.pdf', 'output\el_array2_violin_case3.pdf','output\ept_array_violin_case3.pdf']
        plotRelativeMagneticEnergy(El_array, El_tor_array, El_pol_array, Ept_array, paxi_array, filepaths_violin)



