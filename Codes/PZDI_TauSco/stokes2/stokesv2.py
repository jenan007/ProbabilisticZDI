import math
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.special import lpmv
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = (16, 12)
plt.rc('font', family='arial')
plt.rcParams.update({'font.size':16}) #24 #30
font = fm.FontProperties(family='arial')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def read_obs(filename, ncomment=8):
    """
    Reads observational data from a file and profile information.

    Parameters:
    filename (str): Path to the input file containing the observational data.
    ncomment (int): Number of comment/header lines to skip in the file (default: 8).

    Returns:
    nphase (int): Number of rotational phases.
    nv (int): Number of velocity points.
    phase (ndarray): Array of rotational phases.
    vobs (ndarray): Array of observed velocities.
    sigv (ndarray): Array of uncertainties for the Stokes V parameter.
    prfi (ndarray): Intensity profiles (I) for each phase.
    prfv (ndarray): Stokes V profiles for each phase.
    """

    # open input file
    infile = open(filename, 'r')

    # skip comment lines
    for i in range(ncomment):
        a = infile.readline()

    # number of rotational phases, number of velocity points
    a = infile.readline()
    arr = str.split(a.strip())
    nphase = int(arr[0])
    nv = int(arr[1])

    # velocity array
    a = infile.readline()
    arr = str.split(a.strip())
    vobs = np.array(arr, dtype=np.float64)

    # define phase, sigma and profile arrays
    sigv = np.empty(nphase)
    phase = np.empty(nphase)
    prfi = np.empty([nphase, nv], dtype=np.float64)
    prfv = np.empty([nphase, nv], dtype=np.float64)

    # loop over phases
    for i in range(nphase):
        # rotational phase and uncertainty for Stokes V
        a = infile.readline()
        arr = str.split(a.strip())
        phase[i] = float(arr[0])
        sigv[i] = float(arr[1])

        # I profile for this phase
        a = infile.readline()
        arr = str.split(a.strip())
        prfi[i, :] = np.array(arr, dtype=np.float64)

        # V profile for this phase
        a = infile.readline()
        arr = str.split(a.strip())
        prfv[i, :] = np.array(arr, dtype=np.float64)

    infile.close()
    return nphase, nv, phase, vobs, sigv, prfi, prfv


def surface(ntot: int = 3909):
    """
        Computes the surface grid discretization.

        Parameters:
        ntot (int): Total number of points to distribute on the sphere (default: 3909).

        Returns:
        nlon (ndarray): Number of longitude divisions for each latitude.
        xyz (ndarray): Cartesian coordinates (x, y, z) of the points on the sphere.
        area (ndarray): Area associated with each grid point.
        latitude (ndarray): Latitude values for each point in radians.
        longitude (ndarray): Longitude values for each point in radians.
        """
    nlat = round(0.5 * (1 + np.sqrt(1 + math.pi * ntot))) - 1
    nlon = np.zeros(nlat, dtype=int)

    xlat = math.pi * (np.arange(nlat) + 0.5) / nlat - math.pi / 2
    xcirc = 2 * np.cos(xlat[1:])

    nlon[1:] = np.around(xcirc * nlat) + 1
    nlon[0] = ntot - np.sum(nlon[1:])

    if abs(nlon[0] - nlon[nlat - 1]) > nlat:
        nlon[1:] = nlon[1:] + (nlon[0] - nlon[nlat - 1]) / nlat

    if nlon[0] < nlon[nlat - 1]:
        nlon[1:] = nlon[1:] - 1

    nlon[0] = ntot - np.sum(nlon[1:])

    xyz = np.zeros((3, ntot))
    latitude = np.zeros(ntot)
    longitude = np.zeros(ntot)
    area = np.zeros(ntot)

    slt = np.concatenate([[0.0],
                          (xlat[1:nlat] + xlat[0:nlat - 1]) * 0.5 + math.pi * 0.5,
                          [math.pi]])
    j = 0

    for i in range(nlat):
        coslat = np.cos(xlat[i])
        sinlat = np.sin(xlat[i])

        xlon = 2 * math.pi * (np.arange(nlon[i], dtype=float) + 0.5) / nlon[
            i]

        sinlon = np.sin(xlon)
        coslon = np.cos(xlon)

        xyz[0, j:j + nlon[i]] = coslat * sinlon
        xyz[1, j:j + nlon[i]] = -coslat * coslon
        xyz[2, j:j + nlon[i]] = sinlat

        latitude[j:j + nlon[i]] = xlat[i]
        longitude[j:j + nlon[i]] = xlon
        area[j:j + nlon[i]] = 2 * math.pi * (np.cos(slt[i]) - np.cos(slt[i + 1])) / nlon[i]

        j = j + nlon[i]

    return nlon, xyz, area, latitude, longitude


def coef2fld(theta, phi, SHcoef: jnp.ndarray, lmax, renorm=True, ibet=0, igam=0):
    """
        Computes the magnetic field vector components from spherical harmonics coefficients.

        Parameters:
        theta (float): Colatitude in radians.
        phi (float): Longitude in radians.
        SHcoef (jnp.ndarray): Array of spherical harmonics coefficients.
        lmax (int): Maximum spherical harmonics degree.
        renorm (bool): Whether to apply renormalization to the coefficients (default: True).
        ibet (int): If set to 1, we add beta coefficients in the spherhical hamonic expansion (independent radial and horizontal poloidal field) (default is 0).
        igam (int): If set to 1, we add gamma coefficients in the spherical harmonic expansion (toroidal field) (default is 0).

        Returns:
        jnp.ndarray: Magnetic field components [Bx, By, Bz] as a stacked array.
        """
    cost = np.cos(theta)
    sint = np.sin(theta)

    acc_x, acc_y, acc_z = [], [], []
    sh_alpha, sh_beta, sh_gamma = [], [], []

    i = 0
    for l in range(1, lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            Clm = math.sqrt((2 * l + 1) / 4 / math.pi * math.factorial(l - am) / math.factorial(l + am))

            if renorm:
                if m != 0:
                    Clm = Clm * math.sqrt(2)

            Plm = lpmv(am, l, cost)
            Pl1m = lpmv(am, l + 1, cost)
            DPlm = ((l - am + 1) * Pl1m - (l + 1) * cost * Plm) / sint

            if m >= 0:
                Y = -Clm * Plm * np.cos(am * phi)
                Z = Clm / (l + 1) * DPlm * np.cos(am * phi)
                X = -Clm / (l + 1) * Plm / sint * am * np.sin(am * phi)
            else:
                Y = -Clm * Plm * np.sin(am * phi)
                Z = Clm / (l + 1) * DPlm * np.sin(am * phi)
                X = Clm / (l + 1) * Plm / sint * am * np.cos(am * phi)


            # poloidal field
            coef_a = SHcoef[i]

            if ibet == 1:
                coef_b = SHcoef[i + lmax * (lmax + 2)]
                if renorm: coef_b = coef_b * np.sqrt((l + 1) / l)
            else:
                if renorm: coef_a = coef_a * np.sqrt((l + 1) / (2 * l + 1))
                coef_b = coef_a

            # toroidal field
            coef_g = 0.0
            if igam == 1:
                coef_g = SHcoef[i + lmax * (lmax + 2) * (1 + ibet)]
                if renorm: coef_g = coef_g * np.sqrt((l + 1) / l)

            sh_alpha.append(coef_a)
            sh_beta.append(coef_b)
            sh_gamma.append(coef_g)

            acc_x.append(X)
            acc_y.append(Y)
            acc_z.append(Z)

            i = i + 1

    Bfield = jnp.vstack([(-jnp.vstack(sh_alpha) * jnp.vstack(acc_y)).sum(axis=0),
                          (jnp.vstack(sh_beta) * jnp.vstack(acc_z) + jnp.vstack(sh_gamma) * jnp.vstack(acc_x)).sum(axis=0),
                         (-jnp.vstack(sh_beta) * jnp.vstack(acc_x) + jnp.vstack(sh_gamma) * jnp.vstack(acc_z)).sum(axis=0)])

    return Bfield


def LocPrf(Bx, By, Bz, mu, vv, line_par):
    """
    Calculate local Stokes parameter profiles (I and V).

    Parameters:
    Bx, By, Bz (ndarray): Magnetic field components in Cartesian coordinates.
    line_par (list): Line parameters:
        - line_par[0]: Method selector (0 for weak field, 1 for Milne-Eddington).

    Returns:
        locI (ndarray): Stokes I profile.
        locV (ndarray): Stokes V profile.
    """
    # static constants
    Vc = 299792.458
    Cz = 4.66864377e-10

    # Gaussian function parameters
    sig2fwhm = np.sqrt(8 * np.log(2))
    sigm = line_par[3] / sig2fwhm
    isig = -0.5 / (sigm * sigm)

    # dimensions of vv array
    nv, ns = vv.shape
    vv1 = np.ones(nv)

    # weak field approximation with Gaussian profile
    if line_par[0] == 0:
        Bz1 = jnp.outer(vv1, Bz)
        Bconst = -Cz * Vc * line_par[1] * line_par[2]
        ex = jnp.exp(isig * vv * vv)
        locI = 1 - line_par[4] * ex
        locV = -Bconst * line_par[4] * isig * 2 * vv * ex * Bz1

    # Milne-Eddington analytical solution with Gaussian absorption profile, assuming Zeeman triplet splitting
    else:
        Bmod = jnp.sqrt(Bx * Bx + By * By + Bz * Bz)
        Btan = jnp.sqrt(Bx * Bx + By * By)

        sint = Btan/Bmod
        cost = Bz/Bmod

        sinc = By/Btan
        cosc = Bx/Btan
        sin2c = 2 * sinc * cosc
        cos2c = cosc ** 2 - sinc ** 2

        Bmod = jnp.outer(vv1, Bmod)
        cost = jnp.outer(vv1, cost)
        cost2 = cost ** 2
        sint2 = jnp.outer(vv1, sint ** 2)
        sin2c = jnp.outer(vv1, sin2c)
        cos2c = jnp.outer(vv1, cos2c)

        Vz = Cz * Vc * Bmod * line_par[1] * line_par[2]
        ex0 = jnp.exp(isig * (vv) ** 2)
        ex1 = jnp.exp(isig * (vv + Vz) ** 2)
        ex2 = jnp.exp(isig * (vv - Vz) ** 2)

        Kl = 1 / (1 - line_par[4] * (1 + line_par[5]) / line_par[5]) - 1

        tp = Kl * ex0
        tb = Kl * ex1
        tr = Kl * ex2
        rp = Kl * isig * 2 * vv * ex0
        rb = Kl * isig * 2 * (vv + Vz) * ex1
        rr = Kl * isig * 2 * (vv - Vz) * ex2

        kI = 0.5 * (tp * sint2 + 0.5 * (tb + tr) * (1 + cost2))
        kQ = 0.5 * (tp - 0.5 * (tb + tr)) * cos2c * sint2
        kU = 0.5 * (tp - 0.5 * (tb + tr)) * sin2c * sint2
        kV = 0.5 * (tr - tb) * cost
        fQ = 0.5 * (rp - 0.5 * (rb + rr)) * cos2c * sint2
        fU = 0.5 * (rp - 0.5 * (rb + rr)) * sin2c * sint2
        fV = 0.5 * (rr - rb) * cost

        betmu = jnp.outer(vv1, line_par[5] * mu)
        betmu = betmu / (1 + betmu)
        kI1 = 1 + kI
        kI2 = kI1 * kI1
        delta = kI2 * kI2 + kI2 * (fQ * fQ + fU * fU + fV * fV - kQ * kQ - kU * kU - kV * kV) - (
                    kQ * fQ + kU * fU + kV * fV) ** 2
        bdelta = betmu / delta

        locI = 1 - betmu * (1 - kI1 / delta * (kI2 + fQ * fQ + fU * fU + fV * fV))
        locV = -bdelta * (kI2 * kV + fV * (kQ * fQ + kU * fU + kV * fV))

    return locI, locV


def generate_stokes_spectrum(SHcoef: jnp.ndarray,
                             area, lat, lon, lmax=5, ntot=10173, star_vsini=50., star_incl=60., star_limbd=0.5,
                             line_width=5.0, line_wave=5000., line_lande=1.2, line_depth=0.5,
                             obs_ntimes=20, obs_noise=5.e-5, obs_vstep=1.0, weakfield=True, ibet=0, igam=0, obs_v=None, nv=None, obs_phases=None):
    """
        Generates synthetic Stokes I and V profiles for a star using spherical harmonics coefficients.

        *Input parameters are defined in the main script.*

        Returns:
        prfI (ndarray): Stokes I profiles for all phases and velocities.
        prfV (ndarray): Stokes V profiles for all phases and velocities.
        Bfield (jnp.ndarray): Magnetic field vector components.
        obs_phases (ndarray): Observational phases.
        obs_v (ndarray): Velocity grid for the observations.
    """
    if weakfield:
        line_type = 0
    else:
        line_type = 1

    line_beta = 2.

    # group line parameters in a single array
    line_par = np.array([line_type, line_wave, line_lande, line_width, line_depth, line_beta])

    theta = 0.5 * math.pi - lat
    phi = lon

    Bfield = coef2fld(theta, phi, SHcoef, lmax, ibet=ibet, igam=igam)

    if obs_phases is None:
        obs_phases = np.arange(obs_ntimes, dtype=float) / obs_ntimes
    obs_vrange = 1.5 * np.sqrt(star_vsini ** 2 + line_width ** 2)

    if obs_v is None:
        obs_v = np.linspace(-obs_vrange, obs_vrange, round(2 * obs_vrange / obs_vstep) + 1)

    if nv is None:
        nv = len(obs_v)

    cosi = np.cos(np.deg2rad(star_incl))
    sini = np.sin(np.deg2rad(star_incl))
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    prfIs = []
    prfVs = []

    for i in range(obs_ntimes):
        coslon = np.cos(lon + 2 * np.pi * obs_phases[i])
        sinlon = np.sin(lon + 2 * np.pi * obs_phases[i])
        mu = sinlat * cosi + coslat * sini * coslon
        ivis = (mu > 0).nonzero()

        wgt = area[ivis] * mu[ivis] * (1 - star_limbd + star_limbd * mu[ivis])
        wgt = wgt / np.sum(wgt)
        wgt = np.outer(np.ones(nv), wgt)

        vshift = -star_vsini * sinlon[ivis] * coslat[ivis]

        v2d = np.outer(obs_v, np.ones(len(vshift))) + np.outer(np.ones(nv), vshift)

        Bx = \
            (coslat[ivis] * sinlon[ivis]) * Bfield[0, ivis] - \
            (sinlat[ivis] * sinlon[ivis]) * Bfield[1, ivis] + \
            (coslon[ivis]) * Bfield[2, ivis]  # Bx (normal to line of sight) magnetic field vector component
        By = \
            (sinlat[ivis] * sini - coslat[ivis] * cosi * coslon[ivis]) * Bfield[0, ivis] + \
            (coslat[ivis] * sini + sinlat[ivis] * cosi * coslon[ivis]) * Bfield[1, ivis] + \
            (cosi * sinlon[ivis]) * Bfield[2, ivis]  # By (normal to line of sight) magnetic field vector component
        Bz = \
            (sinlat[ivis] * cosi + coslat[ivis] * sini * coslon[ivis]) * Bfield[0, ivis] + \
            (coslat[ivis] * cosi - sinlat[ivis] * sini * coslon[ivis]) * Bfield[1, ivis] - \
            (sini * sinlon[ivis]) * Bfield[2, ivis]  # Bz (line of sight) magnetic field vector component

        locI, locV = LocPrf(Bx, By, Bz, mu[ivis], v2d, line_par)

        prfIs.append((locI * wgt).sum(axis=1))
        prfVs.append((locV * wgt).sum(axis=1))

    prfI = jnp.vstack(prfIs)
    prfV = jnp.vstack(prfVs)

    return prfI, prfV, Bfield, obs_phases, obs_v


def generate_true_stokes_spectrum(SHcoef: jnp.ndarray,
                             area, lat, lon, lmax=5, ntot=10173, star_vsini=50., star_incl=60., star_limbd=0.5,
                             line_width=5.0, line_wave=5000., line_lande=1.2, line_depth=0.5,
                             obs_ntimes=20, obs_noise=5.e-5, obs_vstep=1.0, weakfield=True, ibet=0, igam=0, obs_v=None, nv=None, obs_phases=None):
    """
    Generates synthetic Stokes I and V profiles for a star using spherical harmonics coefficients.

    *Input parameters are defined in the main script.*

    Returns:
    prfI (ndarray): Stokes I profiles for all phases and velocities.
    prfV (ndarray): Stokes V profiles for all phases and velocities.
    Bfield (jnp.ndarray): Magnetic field vector components.
    obs_phases (ndarray): Observational phases.
    obs_v (ndarray): Velocity grid for the observations.
    """

    if weakfield:
        line_type = 0
    else:
        line_type = 1

    line_beta = 2.

    # group line parameters
    line_par = np.array([line_type, line_wave, line_lande, line_width, line_depth, line_beta])

    theta = 0.5 * math.pi - lat
    phi = lon
    obs_noise = obs_noise

    Bfield = coef2fld(theta, phi, SHcoef, lmax, ibet=ibet, igam=igam)

    if obs_phases is None:
        obs_phases = np.arange(obs_ntimes, dtype=float) / obs_ntimes
    obs_vrange = 1.5 * np.sqrt(star_vsini ** 2 + line_width ** 2)
    if obs_v is None:
        obs_v = np.linspace(-obs_vrange, obs_vrange, round(2 * obs_vrange / obs_vstep) + 1)

    if nv is None:
        nv = len(obs_v)

    cosi = np.cos(np.deg2rad(star_incl))
    sini = np.sin(np.deg2rad(star_incl))
    coslat = np.cos(lat)
    sinlat = np.sin(lat)

    prfIs = []
    prfVs = []

    for i in range(obs_ntimes):
        coslon = np.cos(lon + 2 * np.pi * obs_phases[i])
        sinlon = np.sin(lon + 2 * np.pi * obs_phases[i])
        mu = sinlat * cosi + coslat * sini * coslon
        ivis = (mu > 0).nonzero()

        wgt = area[ivis] * mu[ivis] * (1 - star_limbd + star_limbd * mu[ivis])
        wgt = wgt / np.sum(wgt)
        wgt = np.outer(np.ones(nv), wgt)

        vshift = -star_vsini * sinlon[ivis] * coslat[ivis]

        v2d = np.outer(obs_v, np.ones(len(vshift))) + np.outer(np.ones(nv), vshift)

        Bx = \
            (coslat[ivis] * sinlon[ivis]) * Bfield[0, ivis] - \
            (sinlat[ivis] * sinlon[ivis]) * Bfield[1, ivis] + \
            (coslon[ivis]) * Bfield[2, ivis]  # Bx (normal to line of sight) magnetic field vector component
        By = \
            (sinlat[ivis] * sini - coslat[ivis] * cosi * coslon[ivis]) * Bfield[0, ivis] + \
            (coslat[ivis] * sini + sinlat[ivis] * cosi * coslon[ivis]) * Bfield[1, ivis] + \
            (cosi * sinlon[ivis]) * Bfield[2, ivis]  # By (normal to line of sight) magnetic field vector component
        Bz = \
            (sinlat[ivis] * cosi + coslat[ivis] * sini * coslon[ivis]) * Bfield[0, ivis] + \
            (coslat[ivis] * cosi - sinlat[ivis] * sini * coslon[ivis]) * Bfield[1, ivis] - \
            (sini * sinlon[ivis]) * Bfield[2, ivis]  # Bz (line of sight) magnetic field vector component

        locI, locV = LocPrf(Bx, By, Bz, mu[ivis], v2d, line_par)

        if obs_noise > 0:
            prfIs.append((locI * wgt).sum(axis=1)+np.random.normal(size=nv) * obs_noise)
            prfVs.append((locV * wgt).sum(axis=1)+np.random.normal(size=nv) * obs_noise)
        else:
            prfIs.append((locI * wgt).sum(axis=1))
            prfVs.append((locV * wgt).sum(axis=1))

    prfI = jnp.vstack(prfIs)
    prfV = jnp.vstack(prfVs)

    return prfI, prfV, Bfield, obs_phases, obs_v


def PlotMap(fig, ax, x, y, z, levels, star_incl=None, cmap='RdBu_r', title=None, xx_label='Longitude', yy_label='Latitude'):
    """
    Plots a rectangular map of the stellar surface.

    Parameters:
    fig (matplotlib.figure.Figure): Figure object to contain the plot.
    ax (matplotlib.axes.Axes): Axes object corresponding to plot.
    x (ndarray): Longitude values (in degrees).
    y (ndarray): Latitude values (in degrees).
    z (ndarray): Values to be contoured (e.g., magnetic field strengths).
    levels (list or ndarray): Contour levels to use in the plot.
    star_incl (float, optional): Stellar inclination angle (degrees). If provided, adds a grey mask for the invisible region.
    cmap (str): Colormap for the contour plot (default: 'RdBu_r').
    title (str, optional): Title of the plot.
    xx_label (str): Label for the x-axis (default: 'Longitude').
    yy_label (str): Label for the y-axis (default: 'Latitude').
    """
    ax.tricontour(x, y, z, levels=levels, linewidths=0.5, colors='k')
    cntr = ax.tricontourf(x, y, z, levels=levels, cmap=cmap)
    fig.colorbar(cntr, ax=ax)
    if star_incl is not None: ax.add_patch(Rectangle((0, -90), 360, 90 - star_incl, facecolor='grey', alpha=0.5))
    ax.set(xlim=(0, 360), ylim=(-90, 90))
    ax.set_title(title)
    ax.set_xlabel(xx_label)
    ax.set_ylabel(yy_label)

    if xx_label != 'Longitude':
        ax.set_xticks([])
    if yy_label != 'Latitude':
        ax.set_yticks([])



def plot_predictive_distribution(lat, lon, Bfield, prfI, prfV, star_incl, obs_v, obs_ntimes, obs_vstep, obs_phases, errV, obsV, obsI, prediction_uncertainty=None, posterior_uncertainty=None, samples=None, num_samples=5,
                           filename='temp'):
    """
        Plots predictive mean Stokes V profiles. Optionally displays uncertainties and sample trajectories.

        Parameters:
        prediction_uncertainty (ndarray): Predictive uncertainty for each phase and velocity (optional).
        posterior_uncertainty (ndarray): Posterior uncertainty for each phase and velocity (optional).
        samples (ndarray): Sample trajectories for each phase and velocity (optional).
        num_samples (int): Number of sample trajectories to plot (default: 5).
        filename (str): Filename to save the plot (default: 'temp').
        *Other parameters are defined in the main script.*

        Returns:
        None. Saves the plot to the specified filename.
        """

    plt.rcParams["figure.figsize"] = (10, 16)
    cmap_list = ['red', 'cornflowerblue', 'red']
    style_list = ['solid', 'solid', 'solid']
    prfcol = 2
    num_stds = 3
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, prfcol, figure=fig)
    ax4 = fig.add_subplot(gs[:, -prfcol])
    if prfcol == 2: ax5 = fig.add_subplot(gs[:, -1])

    x = np.rad2deg(lon)
    y = np.rad2deg(lat)
    n_levels = 15
    vmax = np.max(abs(Bfield[0, :]))
    vmin = -vmax

    # plot time series of spectral line profiles
    xr = [obs_v[0], obs_v[-1]]
    xr[0] = xr[0] - 0.15 * (obs_v[-1] - obs_v[0])
    xr[1] = xr[1] + 0.25 * (obs_v[-1] - obs_v[0])
    ax4.set_xlim(xmin=xr[0], xmax=xr[1])

    plot_type = 1  # 0=Stokes I, 1=Stokes V
    overlap_i = 0.3
    overlap_v = 0.4

    if prfcol == 2:
        nmax1 = int(np.ceil(obs_ntimes * 0.5))
        nmax2 = obs_ntimes - nmax1
        i0 = nmax1
    else:
        nmax1 = obs_ntimes

    if plot_type == 0:
        ystep = overlap_i * (np.max(prfI) - np.min(prfI))

        ax4.set_xlim(xmin=xr[0], xmax=xr[1])
        for i in range(nmax1):
            yo = ystep * (nmax1 - 1 - i)
            if 'obsI' in locals():
                ax4.plot(obs_v, obsI[i, :] + yo, color='k')
                ax4.plot(obs_v, prfI[i, :] + yo, color='r')
            else:
                ax4.plot(obs_v, prfI[i, :] + yo, color='k')
            ax4.text(obs_v[-1] + obs_vstep, 1 + yo, "{:.3f}".format(obs_phases[i]), color='b', fontsize='x-small')
        ax4.set_title('Stokes I')
        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('I/Ic')

        if prfcol == 2:
            ax5.set_xlim(xmin=xr[0], xmax=xr[1])
            for i in range(nmax2):
                yo = ystep * (nmax2 - 1 - i)
                if 'obsI' in locals():
                    ax5.plot(obs_v, obsI[i + i0, :] + yo, color='k')
                    ax5.plot(obs_v, prfI[i + i0, :] + yo, color='r')
                else:
                    ax5.plot(obs_v, prfI[i + i0, :] + yo, color='k')
                ax5.text(obs_v[-1] + obs_vstep, 1 + yo, "{:.3f}".format(obs_phases[i + i0]), color='b',
                         fontsize='x-small')
            ax5.set_title('Stokes I')
            ax5.set_xlabel('Velocity (km/s)')
    else:
        if 'obsV' in locals():
            ystep = overlap_v * (np.max(obsV) - np.min(obsV)) * 100
        else:
            ystep = overlap_v * (np.max(prfV) - np.min(prfV)) * 100

        ax4.set_xlim(xmin=xr[0], xmax=xr[1])
        for i in range(nmax1):
            yo = ystep * (nmax1 - 1 - i)
            ax4.plot([obs_v[0], obs_v[-1]], [yo, yo], 'b:')
            if 'obsV' in locals():
                ax4.plot(obs_v, obsV[i, :] * 100 + yo, color='k')
                ax4.plot(obs_v, prfV[i, :] * 100 + yo, color='r')
                if prediction_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * prediction_uncertainty[i, :])*100+yo, (prfV[i, :] - num_stds * prediction_uncertainty[i, :])*100+yo , color='lightgray')
                if posterior_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     color='darkgray')
                if samples is not None:
                    for path_idx in range(num_samples):
                        path = samples[:, path_idx].reshape((49, 18))
                        ax4.plot(obs_v, path[i, :] * 100 + yo, color=cmap_list[path_idx], linestyle=style_list[path_idx])
            else:
                ax4.plot(obs_v, prfV[i, :] * 100 + yo, color='k')
                if prediction_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * prediction_uncertainty[i, :])*100+yo, (prfV[i, :] - num_stds * prediction_uncertainty[i, :])*100+yo , color='lightgray')
                if posterior_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     color='darkgray')
                if samples is not None:
                    for path_idx in range(num_samples):
                        path = samples[:, path_idx].reshape((49, 18))
                        ax4.plot(obs_v, path[i, :] * 100 + yo, color=cmap_list[path_idx], linestyle=style_list[path_idx]) # color=cmap(path_idx #linestyle='dashed'

            if 'errV' in locals():
                ax4.errorbar(obs_v[0] - 0.05 * (xr[1] - xr[0]), yo, yerr=num_stds*errV[i] * 100, fmt='none', capsize=1.0,
                             elinewidth=0.5)
            ax4.text(obs_v[-1] + obs_vstep, yo, "{:.3f}".format(obs_phases[i]), color='b', fontsize='x-small')

        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('V/Ic (%)')

        if prfcol == 2:
            ax5.set_xlim(xmin=xr[0], xmax=xr[1])
            for i in range(nmax2):
                yo = ystep * (nmax2 - 1 - i)
                ax5.plot([obs_v[0], obs_v[-1]], [yo, yo], 'b:')
                if 'obsV' in locals():
                    ax5.plot(obs_v, obsV[i + i0, :] * 100 + yo, color='k')
                    ax5.plot(obs_v, prfV[i + i0, :] * 100 + yo, color='r')
                    if prediction_uncertainty is not None:
                        ax5.fill_between(obs_v, (prfV[i + i0, :] + num_stds * prediction_uncertainty[i + i0, :])*100+yo,
                                         (prfV[i + i0, :] - num_stds * prediction_uncertainty[i + i0, :])*100+yo, color='lightgray')
                    if posterior_uncertainty is not None:
                        ax5.fill_between(obs_v, (prfV[i + i0, :] + num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         color='darkgray')
                    if samples is not None:
                        for path_idx in range(num_samples):
                            path = samples[:, path_idx].reshape((49, 18))
                            ax5.plot(obs_v, path[i + i0, :] * 100 + yo, color=cmap_list[path_idx], linestyle=style_list[path_idx])
                else:
                    ax5.plot(obs_v, prfV[i + i0, :] * 100 + yo, color='k')
                    if prediction_uncertainty is not None:
                        ax5.fill_between(obs_v, (prfV[i + i0, :] + num_stds * prediction_uncertainty[i + i0, :])*100+yo,
                                         (prfV[i + i0, :] - num_stds * prediction_uncertainty[i + i0, :])*100+yo, color='lightgray')
                    if posterior_uncertainty is not None:
                        ax5.fill_between(obs_v, (prfV[i + i0, :] + num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         color='darkgray')
                    if samples is not None:
                        for path_idx in range(num_samples):
                            path = samples[:, path_idx].reshape((49, 18))
                            ax5.plot(obs_v, path[i + i0, :] * 100 + yo, color=cmap_list[path_idx], linestyle=style_list[path_idx])

                if 'errV' in locals():
                    ax5.errorbar(obs_v[0] - 0.05 * (xr[1] - xr[0]), yo, yerr=num_stds*errV[i + i0] * 100, fmt='none',
                                 capsize=1.0, elinewidth=0.5)
                ax5.text(obs_v[-1] + obs_vstep, yo, "{:.3f}".format(obs_phases[i + i0]), color='b', fontsize='x-small')
            ax5.set_xlabel('Velocity (km/s)')

    plt.savefig(filename, transparent=True)


def plot_predictive_distribution_phase_subset(lat, lon, Bfield, prfI, prfV, star_incl, obs_v, obs_ntimes, obs_vstep,
                                          obs_phases, errV, obsV, obsI, prediction_uncertainty=None,
                                          posterior_uncertainty=None, samples=None, num_samples=5,
                                          filename='temp'):
    """See the function 'plot_predictive_distribution'. This function is similar but plots two LSD Stokes V predictions for a subset of phases."""

    plt.rcParams["figure.figsize"] = (10, 16)
    cmap_list = ['red', 'cornflowerblue', 'red']
    style_list = ['solid', 'solid', 'solid']

    prfcol = 1
    num_stds = 3
    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(1, prfcol, figure=fig)
    ax4 = fig.add_subplot(gs[:, -prfcol])
    if prfcol == 2: ax5 = fig.add_subplot(gs[:, -1])

    x = np.rad2deg(lon)
    y = np.rad2deg(lat)
    n_levels = 15
    vmax = np.max(abs(Bfield[0, :]))
    vmin = -vmax

    # plot time series of spectral line profiles
    xr = [obs_v[0], obs_v[-1]]
    xr[0] = xr[0] - 0.15 * (obs_v[-1] - obs_v[0])
    xr[1] = xr[1] + 0.25 * (obs_v[-1] - obs_v[0])
    ax4.set_xlim(xmin=xr[0], xmax=xr[1])

    plot_type = 1
    overlap_i = 0.3
    overlap_v = 0.4

    if prfcol == 2:
        nmax1 = int(np.ceil(obs_ntimes * 0.5))
        nmax2 = obs_ntimes - nmax1
        i0 = nmax1
    else:
        nmax1 = obs_ntimes

    # subset of phases to skip
    my_set = {0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34,
              35, 36, 37, 38, 39, 40, 42, 43, 44, 46, 48}

    if plot_type == 0:
        ystep = overlap_i * (np.max(prfI) - np.min(prfI))

        ax4.set_xlim(xmin=xr[0], xmax=xr[1])

        for i in range(nmax1):
            if i in my_set:
                continue

            yo = ystep * (nmax1 - 1 - i)
            if 'obsI' in locals():
                ax4.plot(obs_v, obsI[i, :] + yo, color='k')
                ax4.plot(obs_v, prfI[i, :] + yo, color='r')
            else:
                ax4.plot(obs_v, prfI[i, :] + yo, color='k')
            ax4.text(obs_v[-1] + obs_vstep, 1 + yo, "{:.3f}".format(obs_phases[i]), color='b', fontsize='x-small')

        ax4.set_title('Stokes I')
        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('I/Ic')

        if prfcol == 2:
            ax5.set_xlim(xmin=xr[0], xmax=xr[1])
            for i in range(nmax2):
                yo = ystep * (nmax2 - 1 - i)
                if 'obsI' in locals():
                    ax5.plot(obs_v, obsI[i + i0, :] + yo, color='k')
                    ax5.plot(obs_v, prfI[i + i0, :] + yo, color='r')
                else:
                    ax5.plot(obs_v, prfI[i + i0, :] + yo, color='k')
                ax5.text(obs_v[-1] + obs_vstep, 1 + yo, "{:.3f}".format(obs_phases[i + i0]), color='b',
                         fontsize='x-small')
            ax5.set_title('Stokes I')
            ax5.set_xlabel('Velocity (km/s)')
    else:
        if 'obsV' in locals():
            ystep = overlap_v * (np.max(obsV) - np.min(obsV)) * 100
        else:
            ystep = overlap_v * (np.max(prfV) - np.min(prfV)) * 100

        # ax4.set_xlim(xmin=xr[0], xmax=xr[1])
        ax4.set_xlim(xmin=xr[0], xmax=-xr[0])

        ax4.set_yticklabels([])

        cnt = 0
        for i in range(nmax1):
            if i in my_set:
                continue

            yo = ystep * (nmax1 - 1 - cnt)
            cnt = cnt + 1

            ax4.plot([obs_v[0], obs_v[-1]], [yo, yo], 'b:')
            if 'obsV' in locals():
                ax4.plot(obs_v, obsV[i, :] * 100 + yo, color='k')
                ax4.plot(obs_v, prfV[i, :] * 100 + yo, color='r')
                if prediction_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * prediction_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * prediction_uncertainty[i, :]) * 100 + yo,
                                     color='lightgray')
                if posterior_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     color='darkgray')
                if samples is not None:
                    for path_idx in range(num_samples):
                        path = samples[:, path_idx].reshape((49, 18))
                        ax4.plot(obs_v, path[i, :] * 100 + yo, color=cmap_list[path_idx],
                                 linestyle=style_list[path_idx])  # color=cmap(path_idx #linestyle='dashed'
            else:
                ax4.plot(obs_v, prfV[i, :] * 100 + yo, color='k')
                if prediction_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * prediction_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * prediction_uncertainty[i, :]) * 100 + yo,
                                     color='lightgray')
                if posterior_uncertainty is not None:
                    ax4.fill_between(obs_v, (prfV[i, :] + num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     (prfV[i, :] - num_stds * posterior_uncertainty[i, :]) * 100 + yo,
                                     color='darkgray')
                if samples is not None:
                    for path_idx in range(num_samples):
                        path = samples[:, path_idx].reshape((49, 18))
                        ax4.plot(obs_v, path[i, :] * 100 + yo, color=cmap_list[path_idx],
                                 linestyle=style_list[path_idx])  # color=cmap(path_idx #linestyle='dashed'

            if 'errV' in locals():
                ax4.errorbar(obs_v[0] - 0.05 * (xr[1] - xr[0]), yo, yerr=num_stds * errV[i] * 100, fmt='none',
                             capsize=1.0,
                             elinewidth=0.5)
            ax4.text(obs_v[-1] + obs_vstep, yo, "{:.3f}".format(obs_phases[i]), color='b', fontsize='x-small')
        ax4.set_xlabel('Velocity (km/s)')
        ax4.set_ylabel('V/Ic (%)')

        if prfcol == 2:
            ax5.set_xlim(xmin=xr[0], xmax=xr[1])
            for i in range(nmax2):

                yo = ystep * (nmax2 - 1 - cnt)
                cnt = cnt + 1

                ax5.plot([obs_v[0], obs_v[-1]], [yo, yo], 'b:')
                if 'obsV' in locals():
                    ax5.plot(obs_v, obsV[i + i0, :] * 100 + yo, color='k')
                    ax5.plot(obs_v, prfV[i + i0, :] * 100 + yo, color='r')
                    if prediction_uncertainty is not None:
                        ax5.fill_between(obs_v,
                                         (prfV[i + i0, :] + num_stds * prediction_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * prediction_uncertainty[i + i0, :]) * 100 + yo,
                                         color='lightgray')
                    if posterior_uncertainty is not None:
                        ax5.fill_between(obs_v,
                                         (prfV[i + i0, :] + num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         color='darkgray')
                    if samples is not None:
                        for path_idx in range(num_samples):
                            path = samples[:, path_idx].reshape((49, 18))
                            ax5.plot(obs_v, path[i + i0, :] * 100 + yo, color=cmap_list[path_idx],
                                     linestyle=style_list[path_idx])  # cmap(path_idx) #linestyle='dashed'
                else:
                    ax5.plot(obs_v, prfV[i + i0, :] * 100 + yo, color='k')
                    if prediction_uncertainty is not None:
                        ax5.fill_between(obs_v,
                                         (prfV[i + i0, :] + num_stds * prediction_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * prediction_uncertainty[i + i0, :]) * 100 + yo,
                                         color='lightgray')
                    if posterior_uncertainty is not None:
                        ax5.fill_between(obs_v,
                                         (prfV[i + i0, :] + num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         (prfV[i + i0, :] - num_stds * posterior_uncertainty[i + i0, :]) * 100 + yo,
                                         color='darkgray')
                    if samples is not None:
                        for path_idx in range(num_samples):
                            path = samples[:, path_idx].reshape((49, 18))
                            ax5.plot(obs_v, path[i + i0, :] * 100 + yo, color=cmap_list[path_idx],
                                     linestyle=style_list[path_idx])  # cmap(path_idx) #linestyle='dashed'

                if 'errV' in locals():
                    ax5.errorbar(obs_v[0] - 0.05 * (xr[1] - xr[0]), yo, yerr=num_stds * errV[i + i0] * 100, fmt='none',
                                 capsize=1.0, elinewidth=0.5)
                ax5.text(obs_v[-1] + obs_vstep, yo, "{:.3f}".format(obs_phases[i + i0]), color='b', fontsize='x-small')
            ax5.set_xlabel('Velocity (km/s)')

    plt.savefig(filename, transparent=True)


def plot_bfield(lat, lon, Bfield, Bfield2, prfI, prfV, star_incl, obs_v, obs_ntimes, obs_vstep, obs_phases,
                           filename='temp', sample_collection=False):
    """
    Plots two different sets of rectangular magnetic field maps for comparison.

    Parameters:
    - Bfield (ndarray): First magnetic field vector components (radial, meridional, azimuthal).
    - Bfield2 (ndarray): Second magnetic field vector components for comparison.
    - filename (str): Filename to save the plot (default: 'temp').
    - sample_collection (boolean): If specified, sets symmetric levels for contour plots.
    *Other parameters are defined in the main script.*

    Returns:
    None. Saves the plot to the specified filename.
    """

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[2, 0:2])

    ax5 = fig.add_subplot(gs[0, 2:])
    ax6 = fig.add_subplot(gs[1, 2:])
    ax7 = fig.add_subplot(gs[2, 2:])

    x = np.rad2deg(lon)
    y = np.rad2deg(lat)

    n_levels = 15
    levels1 = np.linspace(-np.max(abs(Bfield[0, :])), np.max(abs(Bfield[0, :])), n_levels)
    levels2 = np.linspace(-np.max(abs(Bfield[1, :])), np.max(abs(Bfield[1, :])), n_levels)
    levels3 = np.linspace(-np.max(abs(Bfield[2, :])), np.max(abs(Bfield[2, :])), n_levels)
    levels11 = np.linspace(-np.max(abs(Bfield2[0, :])), np.max(abs(Bfield2[0, :])), n_levels)
    levels22 = np.linspace(-np.max(abs(Bfield2[1, :])), np.max(abs(Bfield2[1, :])), n_levels)
    levels33 = np.linspace(-np.max(abs(Bfield2[2, :])), np.max(abs(Bfield2[2, :])), n_levels)

    if sample_collection:
        n_levels = 15
        vmax = sample_collection
        vmin = -vmax
        levels = np.linspace(vmin, vmax, n_levels)
        levels1 = levels
        levels2 = levels
        levels3 = levels

    PlotMap(fig, ax1, x, y, Bfield[0, :], levels1, star_incl=star_incl, xx_label=None) #title='Radial field'
    PlotMap(fig, ax2, x, y, Bfield[1, :], levels2, star_incl=star_incl, xx_label=None) #title='Meridional field'
    PlotMap(fig, ax3, x, y, Bfield[2, :], levels3, star_incl=star_incl) #title='Azimuthal field'

    PlotMap(fig, ax5, x, y, Bfield2[0, :], levels11, star_incl=star_incl, xx_label=None, yy_label=None)
    PlotMap(fig, ax6, x, y, Bfield2[1, :], levels22, star_incl=star_incl, xx_label=None, yy_label=None)
    PlotMap(fig, ax7, x, y, Bfield2[2, :], levels33, star_incl=star_incl, yy_label=None)
    plt.savefig(filename, transparent=True)


def plot_bfield_var(lat, lon, Bfield, Bfield2, prfI, prfV, star_incl, obs_v, obs_ntimes, obs_vstep, obs_phases,
                           filename='temp', cmap_='Reds', max_bfield=False, sample_collection=None):
    """
        Plots rectangular magnetic field maps along with the corresponding uncertainty maps.

        Parameters:
        - Bfield (ndarray): Magnetic field vector components (radial, meridional, azimuthal).
        - Bfield2 (ndarray): Vector of uncertainties for comparison, e.g. standard deviation at each grid point
        - filename (str): Filename to save the plot (default: 'temp').
        - cmap_ (str): Colormap to use for uncertainty plots (default: 'Reds').
        - sample_collection (boolean): If specified, sets symmetric levels for contour plots.
        *Other parameters are defined in the main script.*

        Returns:
        None. Saves the plot to the specified filename.
        """

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0:2])
    ax3 = fig.add_subplot(gs[2, 0:2])

    ax5 = fig.add_subplot(gs[0, 2:])
    ax6 = fig.add_subplot(gs[1, 2:])
    ax7 = fig.add_subplot(gs[2, 2:])

    x = np.rad2deg(lon)
    y = np.rad2deg(lat)

    n_levels = 15
    levels1 = np.linspace(-np.max(abs(Bfield[0, :])), np.max(abs(Bfield[0, :])), n_levels)
    levels2 = np.linspace(-np.max(abs(Bfield[1, :])), np.max(abs(Bfield[1, :])), n_levels)
    levels3 = np.linspace(-np.max(abs(Bfield[2, :])), np.max(abs(Bfield[2, :])), n_levels)

    if max_bfield:
        n_levels = 15
        vmax = np.max(abs(Bfield))
        vmin = -vmax
        levels = np.linspace(vmin, vmax, n_levels)
        levels1 = levels
        levels2 = levels
        levels3 = levels

    if sample_collection is not None:
        n_levels = 15
        vmax = sample_collection
        vmin = -vmax
        levels = np.linspace(vmin, vmax, n_levels)
        levels1 = levels
        levels2 = levels
        levels3 = levels

    PlotMap(fig, ax1, x, y, Bfield[0, :], levels1, star_incl=star_incl, xx_label=None)
    PlotMap(fig, ax2, x, y, Bfield[1, :], levels2, star_incl=star_incl, xx_label=None)
    PlotMap(fig, ax3, x, y, Bfield[2, :], levels3, star_incl=star_incl)

    PlotMap(fig, ax5, x, y, Bfield2[0, :], levels=15, star_incl=None, cmap=cmap_, xx_label=None, yy_label=None)
    PlotMap(fig, ax6, x, y, Bfield2[1, :], levels=15, star_incl=None, cmap=cmap_, xx_label=None, yy_label=None)
    PlotMap(fig, ax7, x, y, Bfield2[2, :], levels=15, star_incl=None, cmap=cmap_, yy_label=None)
    plt.savefig(filename, transparent=True)
