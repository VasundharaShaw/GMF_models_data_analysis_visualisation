# ============================================================
# 0) Imports & basic setup
# ============================================================

import os, glob, re, warnings, time
from datetime import date
import math
import numpy as np
import pandas as pd

import healpy as hp
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, PowerNorm

from scipy import stats
from scipy.linalg import block_diag
from skimage.metrics import structural_similarity as ssim
from astropy.io import fits
from astropy import units as u

import seaborn as sns
import cmcrameri.cm as cmc
import colorcet as cc

warnings.filterwarnings('ignore')

today = date.today().strftime("%b-%d-%Y")
print("today =", today)

# ============================================================
# 1) Globals / constants
# ============================================================

nside_out = 16
npix_out  = hp.nside2npix(nside_out)

smooth_ang = 0
thresh = 50

deg = np.degrees
rad = np.radians
sqrt = np.sqrt
log10 = np.log10
pi = np.pi

# Base dirs (keep your originals)
direc_data = '/lustre/fs24/group/that/work_vasu/MU_CBASS/fits_files/'
fits_dir   = '/lustre/fs24/group/that/work_vasu/Obs_data_diff_freq/fits_files/'
new_fits   = 'New_Fits_Files/'
template_30_path = '/lustre/fs24/group/that/work_vasu/Template_fit/MagneticFieldNotebooks/Unger_30GHz_files/'
template_5_path  = '/lustre/fs24/group/that/work_vasu/Template_fit/MagneticFieldNotebooks/Unger_5GHz_files/'
kst_fits_dir     = '/lustre/fs24/group/that/work_vasu/Template_fit/Fits_files/'
out_dir          = 'New_Data/8Quad'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# 2) Helper utilities
# ============================================================

def ud_grade_q_u_to_pi(Q, U, nside_out=16, scale=None):
    """
    Degrade Q/U to nside_out and return PI = sqrt(Q^2 + U^2).
    Optionally multiply by 'scale' at the end.
    """
    _, q, u = hp.ud_grade([Q*0, Q, U], nside_out=nside_out)
    pi_map = np.sqrt(q**2 + u**2)
    if scale is not None:
        pi_map = pi_map * scale
    return _, q, u, pi_map

def read_map_div(path, div=1.0):
    """
    Read a healpy map and divide by 'div' for unit conversion.
    """
    m = hp.read_map(path)
    return m / div

def calculate_alpha(d, t, C):
    """
    Solve for alpha minimizing chi^2 with covariance C.
    """
    C_inv_d = np.linalg.solve(C, d)
    C_inv_t = np.linalg.solve(C, t)
    return (t.T @ C_inv_d) / (t.T @ C_inv_t)

def fade_color(rgb, alpha):
    white = np.array([1, 1, 1])
    return tuple(alpha * np.array(rgb[:3]) + (1 - alpha) * white)

# ============================================================
# 3) Load observational polarization data (Q/U)
# ============================================================

# --- Observational polarization data (Q/U) ---
# You can swap Franken with ANY polarization dataset (e.g. C-BASS, S-PASS, WMAP/Planck, QUIJOTE),
# as long as your *model* templates are computed or scaled to the SAME observing frequency as this data.
# Below we read Franken Q/U and later compute PI. If you change dataset/frequency, make sure to
# (a) scale/compute model maps to that frequency, or (b) rescale Q/U with an appropriate synchrotron
#     spectral index law before comparing/fitting.

franken_q = hp.read_map(fits_dir + 'Franken_Q_2024-07-09.fits')
franken_u = hp.read_map(fits_dir + 'Franken_U_2024-07-09.fits')

# Variance maps (nside_out)
var_maps_16 = hp.read_map('Fits_files/Smooth_Frank_Var_nside_out_16_smoothing_0_.fits')
var_maps    = var_maps_16.copy()

# Intensity (for optional masking/inspection)
franken_I = hp.read_map(fits_dir + 'Combined_CBASS_SPASS_Intensity_nside_32.fits')
franken_I = hp.ud_grade(franken_I, nside_out)

# (Q,U) → (Q,U) at nside_out
_, obs_data_q, obs_data_u = hp.ud_grade([franken_q*0, franken_q, franken_u], nside_out=nside_out)

# Frankenstein PI at 5 GHz minus variance (your choice)
cb_sp_combi_5ghz_data_nside_16 = (np.sqrt(obs_data_q**2 + obs_data_u**2)) - var_maps_16

# RM mask (0=mask) and apply to PI
combi_rm_map = hp.read_map(fits_dir + f'Combi_rm_map_RM_deg_{thresh}_nside_{nside_out}.fits')
cb_sp_combi_5ghz_data_nside_16[np.where(combi_rm_map == 0)] = hp.UNSEEN

Real_data_1 = cb_sp_combi_5ghz_data_nside_16.copy()

# ============================================================
# 4) GMF synchrotron templates (Q/U → PI)
# ============================================================

# --- GMF synchrotron templates (Q/U → PI) ---
# Pattern to load ANY number of GMF model synchrotron maps:
#   1) Read Q/U (model) maps
#   2) Convert units (here we divide by 1e9; see note below)
#   3) Degrade to the working nside
#   4) Compute PI = sqrt(Q^2 + U^2)
#
# NOTE on units: the “/1e9” factor is used here to convert model intensities from
#  1/(cm^2 s sr) to Kelvin within this pipeline’s unit convention.
#  If your GMF outputs use different units, adjust this conversion accordingly.
#  (Keep this scaling for consistency across all GMF templates you load below.)

# --- SVT22 structured:
# Unit conversion: divide by 1e9 to convert model intensities from 1/(cm^2 s sr) to Kelvin in this workflow.
Q = read_map_div(new_fits + 'Bstr_5GHz_TM_nside_out_32_Q_CGS.fits', div=1e9)
U = read_map_div(new_fits + 'Bstr_5GHz_TM_nside_out_32_U_CGS.fits', div=1e9)
_, smooth_q, smooth_u, smooth_svt22_str = ud_grade_q_u_to_pi(Q, U, nside_out=nside_out)

hp.mollview(smooth_svt22_str, norm='hist', cmap='jet', title='SVT22 Template'); hp.graticule()

# --- SVT22 structured + turbulent
svt22_str_tur_Q = read_map_div(new_fits + '5GHz_SVT22_Str_Tur_Q_CGS.fits', div=1e9)  # /1e9 unit conversion
svt22_str_tur_U = read_map_div(new_fits + '5GHz_SVT22_Str_Tur_U_CGS.fits', div=1e9)
_, smooth_svt22_str_tur_q, smooth_svt22_str_tur_u, smooth_svt22_str_tur = \
    ud_grade_q_u_to_pi(svt22_str_tur_Q, svt22_str_tur_U, nside_out=nside_out)
hp.mollview(smooth_svt22_str_tur, norm='hist', cmap='jet', title='SVT22 Str + Tur Template'); hp.graticule()

# --- JF12 family (Tor, X, Disc, combos) ---
# GMF: JF12 family. This same pattern adapts to ANY number of JF12-based synchrotron Q/U templates; just point
# to the Q/U files, keep the /1e9 conversion (unit note above), ud_grade to nside_out, and compute PI.

JF12_tor_X_Q = read_map_div(new_fits + 'JF12_5GHz_Tor_XField_nside_out_32_Q.fits', div=1e9)
JF12_tor_X_U = read_map_div(new_fits + 'JF12_5GHz_Tor_XField_nside_out_32_U.fits', div=1e9)
_, JF12_tor_X_q, JF12_tor_X_u, smooth_pi_JF12_tor_X = \
    ud_grade_q_u_to_pi(JF12_tor_X_Q, JF12_tor_X_U, nside_out=nside_out)
hp.mollview(smooth_pi_JF12_tor_X, norm='hist', cmap='jet', title='JF12 PI (Tor + X)'); hp.graticule()

JF12_tor_disc_Q = read_map_div(new_fits + 'JF12_5GHz_Tor_Disc_nside_out_32_Q.fits', div=1e9)
JF12_tor_disc_U = read_map_div(new_fits + 'JF12_5GHz_Tor_Disc_nside_out_32_U.fits', div=1e9)
_, JF12_tor_disc_q, JF12_tor_disc_u, smooth_pi_JF12_tor_disc = \
    ud_grade_q_u_to_pi(JF12_tor_disc_Q, JF12_tor_disc_U, nside_out=nside_out)
hp.mollview(smooth_pi_JF12_tor_disc, norm='hist', cmap='jet', title='JF12 PI (Tor + Disc)'); hp.graticule()

JF12_X_disc_Q = read_map_div(new_fits + 'JF12_5GHz_XField_Disc_nside_out_32_Q.fits', div=1e9)
JF12_X_disc_U = read_map_div(new_fits + 'JF12_5GHz_XField_Disc_nside_out_32_U.fits', div=1e9)
_, JF12_X_disc_q, JF12_X_disc_u, smooth_pi_JF12_X_disc = \
    ud_grade_q_u_to_pi(JF12_X_disc_Q, JF12_X_disc_U, nside_out=nside_out)
hp.mollview(smooth_pi_JF12_X_disc, norm='hist', cmap='jet', title='JF12 PI (X + Disc)'); hp.graticule()

JF12_full_str_Q = read_map_div(new_fits + 'JF12_5GHz_Str_32_Q.fits', div=1e9)
JF12_full_str_U = read_map_div(new_fits + 'JF12_5GHz_Str_32_U.fits', div=1e9)
_, JF12_full_str_q, JF12_full_str_u, smooth_pi_JF12_full_str = \
    ud_grade_q_u_to_pi(JF12_full_str_Q, JF12_full_str_U, nside_out=nside_out)
hp.mollview(smooth_pi_JF12_full_str, norm='hist', cmap='jet', title='JF12 PI (Structured full)'); hp.graticule()

# --- LogSpiral template
# NOTE: U here is divided by 1e8 (source-specific units). If needed, harmonize to /1e9 like the others
# for consistency in Kelvin—adjust only if you’re sure about the original units/normalization.
LogSpiral_Q = read_map_div(new_fits + 'LogSpiral_nside_out32_zmax_1_Q_CGS.fits', div=1e9)
LogSpiral_U = read_map_div(new_fits + 'LogSpiral_nside_out32_zmax_1_U_CGS.fits', div=1e8)
_, LogSpiral_q, LogSpiral_u, smooth_pi_LogSpiral = \
    ud_grade_q_u_to_pi(LogSpiral_Q, LogSpiral_U, nside_out=nside_out)
hp.mollview(smooth_pi_LogSpiral, norm='hist', cmap='jet', title='LogSpiral PI'); hp.graticule()

# --- JF12 full (structured + turbulent)
# GMF: JF12 (full structured + turbulent). Adaptable: repeat for any GMF combo; keep the /1e9 conversion.
JF12_full_str_tur_Q = read_map_div(new_fits + 'JF12_5GHz_Full_Tur_nside_out_32_Q.fits', div=1e9)
JF12_full_str_tur_U = read_map_div(new_fits + 'JF12_5GHz_Full_Tur_nside_out_32_U.fits', div=1e9)
_, JF12_full_str_tur_q, JF12_full_str_tur_u, smooth_pi_JF12_full_str_tur = \
    ud_grade_q_u_to_pi(JF12_full_str_tur_Q, JF12_full_str_tur_U, nside_out=nside_out)
hp.mollview(smooth_pi_JF12_full_str_tur, norm='hist', cmap='jet', title='JF12 PI (Full Str+Tur)'); hp.graticule()

# --- UF23 family (Base/cre10/expX/nebCor/neCL/spur/synCG/twistX)
# GMF: UF23 family. Add as many UF23 Q/U templates as you like by repeating the pattern
# (read → /1e9 → ud_grade → PI). Some “*_striated” maps include an extra multiplicative
# factor after PI; keep or adjust as needed.

UF23_twist_Q = read_map_div(new_fits + 'UF23_5GHz_twistX_nside_out_32_Q.fits', div=1e9)
UF23_twist_U = read_map_div(new_fits + 'UF23_5GHz_twistX_nside_out_32_U.fits', div=1e9)
_, UF23_twist_q, UF23_twist_u, smooth_pi_UF23_twist = \
    ud_grade_q_u_to_pi(UF23_twist_Q, UF23_twist_U, nside_out=nside_out)
smooth_pi_UF23_twist_striated = smooth_pi_UF23_twist * (1 + 0.78)**2
hp.mollview(smooth_pi_UF23_twist, norm='hist', cmap='jet', title='UF23 PI twist'); hp.graticule()

UF23_spur_Q = read_map_div(new_fits + 'UF23_5GHz_Spur_nside_out_32_Q.fits', div=1e9)
UF23_spur_U = read_map_div(new_fits + 'UF23_5GHz_Spur_nside_out_32_U.fits', div=1e9)
_, UF23_spur_q, UF23_spur_u, smooth_pi_UF23_spur = \
    ud_grade_q_u_to_pi(UF23_spur_Q, UF23_spur_U, nside_out=nside_out)
smooth_pi_UF23_spur_striated = smooth_pi_UF23_spur * (1 + 0.330)**2
hp.mollview(smooth_pi_UF23_spur, norm='hist', cmap='jet', title='UF23 PI spur'); hp.graticule()

UF23_neCL_Q = read_map_div(new_fits + 'UF23_5GHz_neCL_nside_out_32_Q.fits', div=1e9)
UF23_neCL_U = read_map_div(new_fits + 'UF23_5GHz_neCL_nside_out_32_U.fits', div=1e9)
_, UF23_neCL_q, UF23_neCL_u, smooth_pi_UF23_neCL = \
    ud_grade_q_u_to_pi(UF23_neCL_Q, UF23_neCL_U, nside_out=nside_out)
smooth_pi_UF23_neCL_striated = smooth_pi_UF23_neCL * (1 + 0.336)**2
hp.mollview(smooth_pi_UF23_neCL, norm='hist', cmap='jet', title='UF23 PI neCL'); hp.graticule()

UF23_nebCor_Q = read_map_div(new_fits + 'UF23_5GHz_nebCor_nside_out_32_Q.fits', div=1e9)
UF23_nebCor_U = read_map_div(new_fits + 'UF23_5GHz_nebCor_nside_out_32_U.fits', div=1e9)
_, UF23_nebCor_q, UF23_nebCor_u, smooth_pi_UF23_nebCor = \
    ud_grade_q_u_to_pi(UF23_nebCor_Q, UF23_nebCor_U, nside_out=nside_out)
smooth_pi_UF23_nebCor_striated = smooth_pi_UF23_nebCor * (1 + 0)**2
hp.mollview(smooth_pi_UF23_nebCor, norm='hist', cmap='jet', title='UF23 PI nebCor'); hp.graticule()

UF23_synCG_Q = read_map_div(new_fits + 'UF23_5GHz_synCG_nside_out_32_Q.fits', div=1e9)
UF23_synCG_U = read_map_div(new_fits + 'UF23_5GHz_synCG_nside_out_32_U.fits', div=1e9)
_, UF23_synCG_q, UF23_synCG_u, smooth_pi_UF23_synCG = \
    ud_grade_q_u_to_pi(UF23_synCG_Q, UF23_synCG_U, nside_out=nside_out)
smooth_pi_UF23_synCG_striated = smooth_pi_UF23_synCG * (1 + 0.63)**2
hp.mollview(smooth_pi_UF23_synCG, norm='hist', cmap='jet', title='UF23 PI synCG'); hp.graticule()

UF23_Base_Q = read_map_div(new_fits + 'UF23_5GHz_Base_nside_out_32_Q.fits', div=1e9)
UF23_Base_U = read_map_div(new_fits + 'UF23_5GHz_Base_nside_out_32_U.fits', div=1e9)
_, UF23_Base_q, UF23_Base_u, smooth_pi_UF23_Base = \
    ud_grade_q_u_to_pi(UF23_Base_Q, UF23_Base_U, nside_out=nside_out)
smooth_pi_UF23_Base_striated = smooth_pi_UF23_Base * (1 + 0.345)**2
hp.mollview(smooth_pi_UF23_Base, norm='hist', cmap='jet', title='UF23 PI Base'); hp.graticule()

UF23_cre10_Q = read_map_div(new_fits + 'UF23_5GHz_cre10_nside_out_32_Q.fits', div=1e9)
UF23_cre10_U = read_map_div(new_fits + 'UF23_5GHz_cre10_nside_out_32_U.fits', div=1e9)
_, UF23_cre10_q, UF23_cre10_u, smooth_pi_UF23_cre10 = \
    ud_grade_q_u_to_pi(UF23_cre10_Q, UF23_cre10_U, nside_out=nside_out)
smooth_pi_UF23_cre10_striated = smooth_pi_UF23_cre10 * (1 + 0.25)**2
hp.mollview(smooth_pi_UF23_cre10, norm='hist', cmap='jet', title='UF23 PI cre10'); hp.graticule()

UF23_expX_Q = read_map_div(new_fits + 'UF23_5GHz_expX_nside_out_32_Q.fits', div=1e9)
UF23_expX_U = read_map_div(new_fits + 'UF23_5GHz_expX_nside_out_32_U.fits', div=1e9)
_, UF23_expX_q, UF23_expX_u, smooth_pi_UF23_expX = \
    ud_grade_q_u_to_pi(UF23_expX_Q, UF23_expX_U, nside_out=nside_out)
smooth_pi_UF23_expX_striated = smooth_pi_UF23_expX * (1 + 0.51)**2
hp.mollview(smooth_pi_UF23_expX, norm='hist', cmap='jet', title='UF23 PI expX'); hp.graticule()

# --- Xu & Han (XH19) template
Han_Q = read_map_div(new_fits + 'Xu_Han_5GHz_nside_out_32_Q.fits', div=1e9)
Han_U = read_map_div(new_fits + 'Xu_Han_5GHz_nside_out_32_U.fits', div=1e9)
_, Han_q, Han_u, smooth_pi_Xu_Han = ud_grade_q_u_to_pi(Han_Q, Han_U, nside_out=nside_out)
hp.mollview(smooth_pi_Xu_Han, norm='hist', cmap='jet', title='Xu Han (XH19)'); hp.graticule()

# ============================================================
# 5) Frequency scaling example (30 GHz → 4.76 GHz)
# ============================================================

# Frequency scaling example (30 GHz → 4.76 GHz) using a synchrotron spectral index of -3.1.
# This shows how to adapt ANY number of GMF Q/U templates computed at one frequency to your data’s
# frequency. Extend this block by adding more files to the list and applying the same ratio.
# IMPORTANT: Make sure your observational data (Franken/C-BASS/etc.) and *scaled* model maps are
# at the same frequency before fitting/TT plots.

freq_30 = 30.0
freq_5  = 4.76
ratio_30_to_5 = (freq_5/freq_30)**-3.1

def load_uf23_30_to_5(name):
    arr = np.loadtxt(os.path.join(template_30_path, f'{name}.txt'))
    # order_in Nested → Ring out
    _, q, u = hp.ud_grade((arr[:,0]*0, arr[:,1], arr[:,2]),
                          nside_out=nside_out, order_in='Nested', order_out='Ring')
    pi30 = np.sqrt(q**2 + u**2)
    return (pi30 * ratio_30_to_5) / 1e6  # to Kelvin in your convention

uf23_base_5_from30   = load_uf23_30_to_5('uf23_base')
uf23_cre10_5_from30  = load_uf23_30_to_5('uf23_cre10')
uf23_expX_5_from30   = load_uf23_30_to_5('uf23_expX')
uf23_nebCor_5_from30 = load_uf23_30_to_5('uf23_nebCor')
uf23_neCL_5_from30   = load_uf23_30_to_5('uf23_neCL')
uf23_spur_5_from30   = load_uf23_30_to_5('uf23_spur')
uf23_synCG_5_from30  = load_uf23_30_to_5('uf23_synCG')
uf23_twistX_5_from30 = load_uf23_30_to_5('uf23_twistX')

hp.mollview(uf23_base_5_from30, norm='hist', cmap='jet', title='UF23 base (scaled 30→4.76 GHz)'); hp.graticule()

# ============================================================
# 6) Unger 5 GHz templates (already near target frequency)
# ============================================================

# Unger 5 GHz templates (already close to the target frequency). Same adaptable pattern:
# read → ud_grade → PI → (unit) scaling. Here “/1e6” converts to Kelvin in this pipeline’s convention.
# Keep consistent units across all templates you compare.

def load_uf23_5(name):
    arr = np.loadtxt(os.path.join(template_5_path, f'{name}.xml.txt'))
    _, q, u = hp.ud_grade((arr[:,0]*0, arr[:,1], arr[:,2]),
                          nside_out=nside_out, order_in='Nested', order_out='Ring')
    pi5 = np.sqrt(q**2 + u**2) / 1e6
    return pi5

uf23_base_5   = load_uf23_5('UF23_base')
uf23_cre10_5  = load_uf23_5('UF23_cre10')
uf23_expX_5   = load_uf23_5('UF23_expX')
uf23_nebCor_5 = load_uf23_5('UF23_nebCor')
uf23_neCL_5   = load_uf23_5('UF23_neCL')
uf23_spur_5   = load_uf23_5('UF23_spur')
uf23_synCG_5  = load_uf23_5('UF23_synCG')
uf23_twistX_5 = load_uf23_5('UF23_twistX')

# Striated versions (multipliers)
uf23_base_5_striated   = uf23_base_5   * (1+0.345)**2
uf23_cre10_5_striated  = uf23_cre10_5  * (1+0.25)**2
uf23_expX_5_striated   = uf23_expX_5   * (1+0.51)**2
uf23_nebCor_5_striated = uf23_nebCor_5 * (1+0)**2
uf23_neCL_5_striated   = uf23_neCL_5   * (1+0.336)**2
uf23_spur_5_striated   = uf23_spur_5   * (1+0.33)**2
uf23_synCG_5_striated  = uf23_synCG_5  * (1+0.63)**2
uf23_twistX_5_striated = uf23_twistX_5 * (1+0.78)**2

# ============================================================
# 7) KST24 (bubble / galaxy / full) @ 4.76 GHz
# ============================================================

def read_kst_q_u(stub):
    hdul = fits.open(os.path.join(kst_fits_dir, f'KST24_StokesQ_4.76GHz_nside64{stub}.fits'))
    # bubble in [1], galaxy in [0]
    Q_bubble = hdul[1].data
    Q_galaxy = hdul[0].data
    hdul.close()

    hdul = fits.open(os.path.join(kst_fits_dir, f'KST24_StokesU_4.76GHz_nside64{stub}.fits'))
    U_bubble = hdul[1].data
    U_galaxy = hdul[0].data
    hdul.close()

    _, Q_b_32, U_b_32 = hp.ud_grade([Q_bubble*0, Q_bubble, U_bubble], nside_out=nside_out)
    _, Q_g_32, U_g_32 = hp.ud_grade([Q_galaxy*0, Q_galaxy, U_galaxy], nside_out=nside_out)

    Q_full = Q_bubble + Q_galaxy
    U_full = U_bubble + U_galaxy
    _, Q_f_32, U_f_32 = hp.ud_grade([Q_full*0, Q_full, U_full], nside_out=nside_out)

    PI_b = np.sqrt(Q_b_32**2 + U_b_32**2)
    PI_g = np.sqrt(Q_g_32**2 + U_g_32**2)
    PI_f = np.sqrt(Q_f_32**2 + U_f_32**2)
    return PI_b, PI_g, PI_f

# r16z4
KST24_bubble_PI_r16z4, KST24_galaxy_PI_r16z4, KST24_full_PI_r16z4 = read_kst_q_u('_creWMAPr16z4')
# r5z6
KST24_bubble_PI_r5z6,  KST24_galaxy_PI_r5z6,  KST24_full_PI_r5z6  = read_kst_q_u('_creWMAPr5z6')
# Dragon
KST24_bubble_PI_Dragon, KST24_galaxy_PI_Dragon, KST24_full_PI_Dragon = read_kst_q_u('')

# Quick previews
hp.mollview(KST24_bubble_PI_r16z4, title='KST24 Bubble (r16z4)', cmap='jet', norm='hist'); hp.graticule()

# ============================================================
# 8) GC symmetric mask (±30° radius around l=b=0) + basic latitude mask
# ============================================================

nside = nside_out
npix = hp.nside2npix(nside)
pix_index = np.arange(npix)
theta_all, phi_all = hp.pix2ang(nside, pix_index)  # radians
lon = np.degrees(phi_all)
lat = 90 - np.degrees(theta_all)

gc_lon, gc_lat = 0.0, 0.0
mask_radius = 30.0
cos_dist = (np.sin(np.radians(lat)) * np.sin(np.radians(gc_lat)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(gc_lat)) * np.cos(np.radians(lon - gc_lon)))
ang_dist = np.degrees(np.arccos(np.clip(cos_dist, -1, 1)))
healpix_map = np.ones(npix)
healpix_map[ang_dist <= mask_radius] = 0

hp.mollview(healpix_map, title="GC symmetric mask (30 deg)", cmap="gray", unit="Value"); hp.graticule(); plt.show()

# Quadrant masks (Q1...Q4, each split into a and b halves)
theta_deg, phi_deg = np.degrees(theta_all), np.degrees(phi_all)

def build_region_mask(theta_cond, phi_cond, data_map, rm_mask, gc_mask, high_lat_mask=None, apply_high_lat=False):
    m1 = theta_cond & phi_cond
    m2 = (data_map != hp.UNSEEN)
    mask_combined = m1 & m2
    mask_tot = np.ones(npix)
    tmp = data_map.copy()
    tmp[~mask_combined] = hp.UNSEEN
    mask_tot[tmp == hp.UNSEEN] = 0
    mask_tot[rm_mask == 0] = 0
    mask_tot[gc_mask == 0] = 0
    if apply_high_lat and high_lat_mask is not None:
        mask_tot[high_lat_mask == 0] = 0
    return mask_tot

# External high-lat masks (optional)
South_mask   = hp.ud_grade(hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/South_mask_N0064_DR2.0.fits'), nside)
virA_rem     = hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/masks/Mask_HighLat_VirA_rem_128.fits')
masks_paddy  = hp.ud_grade(hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/North_mask.fits'), nside)
mask_FS      = hp.ud_grade(hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/masks/NPS_mask_01_nside128.fits'), nside)
mask_fan     = hp.ud_grade(hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/masks/GP_mask_01_nside128.fits'), nside)
mask_dec     = hp.ud_grade(hp.read_map('/lustre/fs24/group/that/work_vasu/Structure_Func_RokeCodes/masks/dec_mask_01_nside128.fits'), nside)

inverse_paddys_mask = np.ones(12*nside**2)
inverse_paddys_mask[masks_paddy==1] = 0
inverse_paddys_mask[masks_paddy==0] = 1
inverse_paddys_mask[mask_fan == 1] = 0
mask_FS2 = hp.ud_grade(inverse_paddys_mask, nside)
test_mask = np.zeros(12*nside**2)
test_mask[South_mask == 0] = mask_FS2[South_mask == 0]
test_mask_inv = np.ones(12*nside**2)
test_mask_inv[test_mask == 1] = 0
test_mask_inv[test_mask == 0] = 1
test_mask_inv[mask_fan == 1] = 0
new_mask_highL = test_mask_inv.copy()
new_mask_highL[hp.ud_grade(mask_dec, nside) == 0] = hp.UNSEEN

# Build all 8 quadrant submasks
masks_Qs = {}
masks_Qs['Q_1a'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (0   <= phi_deg) & (phi_deg <=  90), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_1b'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (90  <= phi_deg) & (phi_deg <= 180), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_2a'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (180 <= phi_deg) & (phi_deg <= 270), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_2b'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (270 <= phi_deg) & (phi_deg <= 360), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_3a'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (0   <= phi_deg) & (phi_deg <=  90), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_3b'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (90  <= phi_deg) & (phi_deg <= 180), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_4a'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (180 <= phi_deg) & (phi_deg <= 270), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_4b'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (270 <= phi_deg) & (phi_deg <= 360), Real_data_1, combi_rm_map, healpix_map)

# Merge to Q_1..Q_4
masks_Qs['Q_1'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (0 <= phi_deg) & (phi_deg <= 180), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_2'] = build_region_mask((0 < theta_deg) & (theta_deg <= 80),  (180 <= phi_deg) & (phi_deg <= 360), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_3'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (0 <= phi_deg) & (phi_deg <= 180), Real_data_1, combi_rm_map, healpix_map)
masks_Qs['Q_4'] = build_region_mask((100 < theta_deg) & (theta_deg <= 180), (180 <= phi_deg) & (phi_deg <= 360), Real_data_1, combi_rm_map, healpix_map)

# Additional combined regions
masks_dict = dict(masks_Qs)
masks_dict["North"] = ((masks_Qs["Q_1"] + masks_Qs["Q_2"]) > 0).astype(int)
masks_dict["South"] = ((masks_Qs["Q_3"] + masks_Qs["Q_4"]) > 0).astype(int)
masks_dict["Right"] = ((masks_Qs["Q_2a"] + masks_Qs["Q_2b"] + masks_Qs["Q_4a"] + masks_Qs["Q_4b"]) > 0).astype(int)
masks_dict["Left"]  = ((masks_Qs["Q_1a"] + masks_Qs["Q_1b"] + masks_Qs["Q_3a"] + masks_Qs["Q_3b"]) > 0).astype(int)

# Visualize combined 8-subregion mask
region_names = [
    'Q_1a','Q_1b','Q_2a','Q_2b','Q_3a','Q_3b','Q_4a','Q_4b'
]
region_alphas = [1,1,1,1,1,1,1,1]
cmap = plt.cm.jet
faded_rgb_list = [fade_color(cmap(i/len(region_names)), region_alphas[i]) for i in range(len(region_names))]
faded_cmap = ListedColormap(faded_rgb_list)
region_ids = {region: i for i, region in enumerate(region_names)}

base_map_regions = np.full(npix, hp.UNSEEN)
for region in region_names:
    base_map_regions[masks_Qs[region] == 1] = region_ids[region]

plt.figure(figsize=(20, 12))
hp.mollview(base_map_regions, cmap=faded_cmap, title="", cbar=False, badcolor='white')
hp.graticule()
plt.gca().set_title("Combined Region Mask", fontsize=28, family='sans-serif')
plt.savefig(os.path.join(out_dir, '8quad_Region_mask.png'), dpi=300, bbox_inches='tight')

# ============================================================
# 9) Template arrays to fit (match your original “template_smooth_pi_*” names)
# ============================================================

temp_amps = np.logspace(0,0,1)  # single amplitude 1.0 → shape compatibility
fac = 1.0  # 

# Build templates “[:, None] * temp_amps[None, :] * fac” to keep shapes identical to your eval() usage
template_smooth_pi_svt22_str        =  smooth_svt22_str[:,None]*temp_amps[None,:]*fac
template_smooth_pi_svt22_str_tur    =  smooth_svt22_str_tur[:,None]*temp_amps[None,:]*fac
template_smooth_pi_JF12_tor_X       =  smooth_pi_JF12_tor_X[:,None]*temp_amps[None,:]*fac
template_smooth_pi_JF12_tor_disc    =  smooth_pi_JF12_tor_disc[:,None]*temp_amps[None,:]*fac
template_smooth_pi_JF12_X_disc      =  smooth_pi_JF12_X_disc[:,None]*temp_amps[None,:]*fac
template_smooth_pi_JF12_full_str    =  smooth_pi_JF12_full_str[:,None]*temp_amps[None,:]*fac
template_smooth_pi_LogSpiral        =  smooth_pi_LogSpiral[:,None]*temp_amps[None,:]*fac
template_smooth_pi_JF12_full_str_tur=  smooth_pi_JF12_full_str_tur[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_base_striated_WMAP   = smooth_pi_UF23_Base_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_cre10_striated_WMAP  = smooth_pi_UF23_cre10_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_nebCor_striated_WMAP = smooth_pi_UF23_nebCor_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_neCL_striated_WMAP   = smooth_pi_UF23_neCL_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_spur_striated_WMAP   = smooth_pi_UF23_spur_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_synCG_striated_WMAP  = smooth_pi_UF23_synCG_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_twist_striated_WMAP  = smooth_pi_UF23_twist_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_expX_striated_WMAP   = smooth_pi_UF23_expX_striated[:,None]*temp_amps[None,:]*fac
template_smooth_pi_uf23_base_striated_Dragon = uf23_base_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_cre10_striated_Dragon= uf23_cre10_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_nebCor_striated_Dragon= uf23_nebCor_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_neCL_striated_Dragon  = uf23_neCL_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_spur_striated_Dragon  = uf23_spur_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_synCG_striated_Dragon = uf23_synCG_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_twistX_striated_Dragon= uf23_twistX_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_uf23_expX_striated_Dragon  = uf23_expX_5_striated[:,None]*temp_amps[None,:]
template_smooth_pi_KST24_bubble_r16z4 = KST24_bubble_PI_r16z4[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_galaxy_r16z4 = KST24_galaxy_PI_r16z4[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_full_r16z4   = KST24_full_PI_r16z4[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_bubble_r5z6  = KST24_bubble_PI_r5z6[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_galaxy_r5z6  = KST24_galaxy_PI_r5z6[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_full_r5z6    = KST24_full_PI_r5z6[:,None]*temp_amps[None,:]*fac
template_smooth_pi_KST24_bubble_Dragon= KST24_bubble_PI_Dragon[:,None]*temp_amps[None,:]
template_smooth_pi_KST24_galaxy_Dragon= KST24_galaxy_PI_Dragon[:,None]*temp_amps[None,:]
template_smooth_pi_KST24_full_Dragon  = KST24_full_PI_Dragon[:,None]*temp_amps[None,:]
template_smooth_pi_Xu_Han             = smooth_pi_Xu_Han[:,None]*temp_amps[None,:]*fac

# Model name list (matches your eval() usage)
model_names = [
    "svt22_str", "svt22_str_tur", "JF12_tor_X", "JF12_tor_disc",
    "JF12_X_disc", "JF12_full_str", "LogSpiral", "JF12_full_str_tur",
    "uf23_base_striated_WMAP", "uf23_cre10_striated_WMAP", "uf23_nebCor_striated_WMAP", "uf23_neCL_striated_WMAP",
    "uf23_spur_striated_WMAP", "uf23_synCG_striated_WMAP", "uf23_twist_striated_WMAP", "uf23_expX_striated_WMAP",
    "uf23_base_striated_Dragon", "uf23_cre10_striated_Dragon", "uf23_nebCor_striated_Dragon", "uf23_neCL_striated_Dragon",
    "uf23_spur_striated_Dragon", "uf23_synCG_striated_Dragon", "uf23_twistX_striated_Dragon", "uf23_expX_striated_Dragon",
    "KST24_bubble_r16z4", "KST24_galaxy_r16z4", "KST24_full_r16z4",
    "KST24_bubble_r5z6", "KST24_galaxy_r5z6", "KST24_full_r5z6",
    "KST24_bubble_Dragon", "KST24_galaxy_Dragon", "KST24_full_Dragon", "Xu_Han"
]

# ============================================================
# 10) Fit on full masked sky (single mask example) + save
# ============================================================

# Build one global mask like your earlier loop
mask_1 = np.where(((0 <= theta_deg) & (theta_deg <= 80) | (100 <= theta_deg) & (theta_deg <= 180)), True, False)
mask_2 = np.where(Real_data_1 != hp.UNSEEN, True, False)

mask_tot = np.ones(npix)
mask_tot[np.where(mask_1 == False)] = 0
mask_tot[np.where(mask_2 == 0)]     = 0
mask_tot[np.where(combi_rm_map == 0)] = 0
mask_tot[np.where(healpix_map == 0)]  = 0
mask_bool = mask_tot.astype(bool)

# Fit each model and save fitted_amps + Spearman
fitted_amps = {m: np.zeros((len(temp_amps),1)) for m in model_names}
spear_cof   = {m: np.zeros((len(temp_amps),1)) for m in model_names}

for ind2 in range(len(temp_amps)):
    d = np.hstack([Real_data_1[mask_bool]]).T
    C = np.diag(np.array(var_maps)[mask_bool])

    for model in model_names:
        template = eval(f"template_smooth_pi_{model}[:,ind2][mask_bool]").T
        alpha = calculate_alpha(d, template, C)
        fitted_amps[model][ind2, 0] = alpha
        recon = (alpha * template)
        spear_cof[model][ind2, 0] = stats.spearmanr(Real_data_1[mask_bool], recon)[0]

# Save single-sky results
for model in model_names:
    file_name = os.path.join(out_dir, f'8Quad_Models_Full_Sky_fitted_spear_cof_{thresh}_{model}_{nside_out}_smooth_ang_{smooth_ang}.txt')
    to_save = np.column_stack((fitted_amps[model].flatten(), spear_cof[model].flatten()))
    np.savetxt(file_name, to_save, header="fitted_amps spear_cof", comments='')

# ============================================================
# 11) Fit per-region (Q_1, Q_1a, ..., Q_4b, Right/Left/North/South) and save
# ============================================================

file_templates = {
    "Q_1":  os.path.join(out_dir, "8Quad_Models_Q_1_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_1a": os.path.join(out_dir, "8Quad_Models_Q_1_0_90_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_1b": os.path.join(out_dir, "8Quad_Models_Q_1_90_180_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_2":  os.path.join(out_dir, "8Quad_Models_Q_2_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_2a": os.path.join(out_dir, "8Quad_Models_Q_2_180_270_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_2b": os.path.join(out_dir, "8Quad_Models_Q_2_270_360_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_3":  os.path.join(out_dir, "8Quad_Models_Q_3_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_3a": os.path.join(out_dir, "8Quad_Models_Q_3_0_90_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_3b": os.path.join(out_dir, "8Quad_Models_Q_3_90_180_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_4":  os.path.join(out_dir, "8Quad_Models_Q_4_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_4a": os.path.join(out_dir, "8Quad_Models_Q_4_180_270_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Q_4b": os.path.join(out_dir, "8Quad_Models_Q_4_270_360_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Right":os.path.join(out_dir, "8Quad_Models_Right_Half_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "Left": os.path.join(out_dir, "8Quad_Models_Left_Half_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "North":os.path.join(out_dir, "8Quad_Models_North_Half_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
    "South":os.path.join(out_dir, "8Quad_Models_South_Half_Sky_fitted_spear_cof_{thresh}_{model}_{nside}_smooth_ang_{smooth}.txt"),
}

for region_name, mask_array in masks_dict.items():
    mask_bool = mask_array.astype(bool)
    for model in model_names:
        fitted_vals, spear_vals = [], []
        for ind2 in range(len(temp_amps)):
            template = eval(f"template_smooth_pi_{model}[:,ind2][mask_bool]").T
            data_vec = Real_data_1[mask_bool]
            C = np.diag(np.array(var_maps)[mask_bool])
            alpha = calculate_alpha(data_vec[:, None], template[:, None], C)
            recon = (alpha * template).flatten()
            fitted_vals.append(alpha.item())
            spear_vals.append(stats.spearmanr(data_vec.flatten(), recon)[0])
        file_name = file_templates[region_name].format(thresh=thresh, model=model, nside=nside_out, smooth=smooth_ang)
        np.savetxt(file_name, np.column_stack((fitted_vals, spear_vals)), header="fitted_amps spear_cof", comments='')

# ============================================================
# 12) Load all saved results → DataFrames → pivot tables → heatmaps
# ============================================================

# Small regions table
rows = []
for region_name, template in file_templates.items():
    if region_name in ["North","South","Left","Right"]:  # skip big regions here
        continue
    for model in model_names:
        file_name = template.format(thresh=thresh, model=model, nside=nside_out, smooth=smooth_ang)
        try:
            data = np.loadtxt(file_name, skiprows=1)
            if data.ndim == 1: data = data.reshape(1, -1)
            if data.shape[1] != 2: continue
            for f, s in data:
                rows.append({"region": region_name, "model": model, "fitted_amp": f, "spear_cof": s})
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

df_small = pd.DataFrame(rows)
df_small['Fit (Spearman)'] = df_small.apply(lambda r: f"{r['fitted_amp']:.1f} ({r['spear_cof']:.2f})", axis=1)

region_order = [
    "Q_1", "Q_1a", "Q_1b",
    "Q_2", "Q_2a", "Q_2b",
    "Q_3", "Q_3a", "Q_3b",
    "Q_4", "Q_4a", "Q_4b"
]
model_order_raw = [
    "svt22_str", "svt22_str_tur", "JF12_tor_X", "JF12_tor_disc",
    "JF12_X_disc", "JF12_full_str", "LogSpiral", "JF12_full_str_tur",
    "uf23_base_striated_WMAP", "uf23_cre10_striated_WMAP", "uf23_nebCor_striated_WMAP", "uf23_neCL_striated_WMAP",
    "uf23_spur_striated_WMAP", "uf23_synCG_striated_WMAP", "uf23_twist_striated_WMAP","uf23_expX_striated_WMAP",
    "uf23_base_striated_Dragon", "uf23_cre10_striated_Dragon", "uf23_nebCor_striated_Dragon", "uf23_neCL_striated_Dragon",
    "uf23_spur_striated_Dragon", "uf23_synCG_striated_Dragon", "uf23_twistX_striated_Dragon","uf23_expX_striated_Dragon",
    "KST24_galaxy_r5z6", "KST24_galaxy_Dragon", "KST24_full_r5z6", "KST24_full_Dragon", "Xu_Han"
]

def format_model_name(name):
    if not isinstance(name, str): return ""
    name = name.replace('_', ' ')
    parts = name.split()
    if parts and parts[0].lower() in ['uf23','jf12','kst24','svt22']:
        parts[0] = parts[0].upper()
    return ' '.join(parts)

df_small['Model'] = pd.Categorical(df_small['model'], categories=model_order_raw, ordered=True)
df_small.sort_values('Model', inplace=True)

pivot_small = df_small.pivot_table(index='Model', columns='region', values='Fit (Spearman)', aggfunc='first')
pivot_small = pivot_small[[c for c in region_order if c in pivot_small.columns]]
pivot_small.index = [format_model_name(m) for m in pivot_small.index]
pivot_small.index = ["XH19" if x.lower().replace(" ","_")=="xu_han" else x for x in pivot_small.index]
pivot_small.to_csv(os.path.join(out_dir, f'8Quad_pivot_table_all_models_thresh_{thresh}_May20.csv'))

# Heatmaps (small regions)
df_disp = pivot_small.copy()
def extract_amp(cell): 
    try: return float(cell.split('(')[0].strip())
    except: return np.nan
def extract_spear(cell):
    try: return float(cell.split('(')[1].replace(')','').strip())
    except: return np.nan

amp_df      = df_disp.applymap(extract_amp)
spearman_df = df_disp.applymap(extract_spear)

mode_spearmans = [
    "svt22_str", "svt22_str_tur", "JF12_tor_X", "JF12_tor_disc",
    "JF12_X_disc", "JF12_full_str", "LogSpiral", "JF12_full_str_tur",
    "uf23_base_striated_WMAP", "uf23_base_striated_Dragon",
    "uf23_cre10_striated_WMAP", "uf23_cre10_striated_Dragon", 
    "uf23_nebCor_striated_WMAP", "uf23_nebCor_striated_Dragon", 
    "uf23_neCL_striated_WMAP", "uf23_neCL_striated_Dragon",
    "uf23_spur_striated_WMAP", "uf23_spur_striated_Dragon", 
    "uf23_synCG_striated_WMAP", "uf23_synCG_striated_Dragon",
    "uf23_twist_striated_WMAP", "uf23_twistX_striated_Dragon",
    "uf23_expX_striated_WMAP", "uf23_expX_striated_Dragon",
    "KST24_galaxy_r5z6", "KST24_galaxy_Dragon",
    "KST24_full_r5z6", "KST24_full_Dragon", "Xu_Han"
]
model_display_order = [format_model_name(m) for m in model_order_raw]
model_display_order = ["XH19" if m.lower().replace(" ","_")=="xu_han" else m for m in model_display_order]
spearman_display = [format_model_name(m) for m in mode_spearmans]
spearman_display = ["XH19" if m.lower().replace(" ","_")=="xu_han" else m for m in spearman_display]

amp_df = amp_df.loc[amp_df.index.intersection(model_display_order)].reindex(model_display_order)
spearman_df_filtered = spearman_df.loc[spearman_df.index.intersection(spearman_display)].reindex(spearman_display)

sqrt_norm = PowerNorm(gamma=0.5)
fig, axes = plt.subplots(1,2, figsize=(30,30), dpi=100)
sns.set(font_scale=1.5)
sns.heatmap(amp_df, annot=True, fmt=".1f", cmap="inferno_r", linewidths=0.5,
            cbar_kws={'label': 'Fitted Amplitude'}, norm=sqrt_norm, ax=axes[0])
axes[0].set_title("Fitted Amplitudes Across Sky Regions", fontsize=18)
axes[0].set_xlabel("Region"); axes[0].set_ylabel("Model")
plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right')

sns.heatmap(spearman_df_filtered, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            cbar_kws={'label': "Spearman's r"}, ax=axes[1])
axes[1].set_title("Spearman's Correlation Across Sky Regions", fontsize=18)
axes[1].set_xlabel("Region"); axes[1].set_ylabel("")
plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

for ax in axes:
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        key = text.lower().replace(" ","_")
        if key in ["kst24_full_r5z6", "kst24_galaxy_r5z6"]:
            updated_text = re.sub(r"r5z6", "", text, flags=re.IGNORECASE)
            new_labels.append(updated_text)
        else:
            new_labels.append(text)
    ax.set_yticklabels(new_labels)
    for label in ax.get_yticklabels():
        if "Dragon" in label.get_text():
            label.set_color("blue")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'August_8Quad_New_Heatmap_amp_spearmans_thresh_{thresh}.png'),
            dpi=100, bbox_inches='tight')

# ============================================================
# 13) Big regions table + heatmaps (North/South/Right/Left/Full)
# ============================================================

regions_big = ["North", "South", "Right", "Left"]
rows_big = []
for region in regions_big:
    template = file_templates[region]
    for model in model_names:
        fname = template.format(thresh=thresh, model=model, nside=nside_out, smooth=smooth_ang)
        if not os.path.exists(fname): 
            print(f"Missing: {fname}")
            continue
        data = np.loadtxt(fname, skiprows=1)
        if data.ndim == 1: data = data.reshape(1, -1)
        if data.shape[1] != 2: continue
        for fitted, spear in data:
            rows_big.append({"Region": region, "Model": model, "Fitted_Amplitude": fitted, "Spearman_Coefficient": spear})

df_big = pd.DataFrame(rows_big)
df_big['Fit (Spearman)'] = df_big.apply(lambda r: f"{r['Fitted_Amplitude']:.1f} ({r['Spearman_Coefficient']:.2f})", axis=1)

pivot_big = df_big.pivot(index='Model', columns='Region', values='Fit (Spearman)')
pivot_big.index = [format_model_name(m) for m in pivot_big.index]
pivot_big.index = ["XH19" if x.lower().replace(" ","_")=="xu_han" else x for x in pivot_big.index]
pivot_big.to_csv(os.path.join(out_dir, f'Big_regions_pivot_table_all_models_thresh{thresh}_May20.csv'))

# Heatmaps for big regions
df_disp_big = pivot_big.copy()
amp_df_big      = df_disp_big.applymap(extract_amp)
spearman_df_big = df_disp_big.applymap(extract_spear)

amp_df_big = amp_df_big.loc[amp_df_big.index.intersection(model_display_order)].reindex(model_display_order)
spearman_df_big = spearman_df_big.loc[spearman_df_big.index.intersection(spearman_display)].reindex(spearman_display)

fig, axes = plt.subplots(1,2, figsize=(30,20), dpi=100)
sns.set(font_scale=1.5)
sns.heatmap(amp_df_big, annot=True, fmt=".1f", cmap="inferno_r", linewidths=0.5,
            cbar_kws={'label': 'Fitted Amplitude'}, ax=axes[0])
axes[0].set_title("Fitted Amplitudes (Big Regions)"); plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right')
sns.heatmap(spearman_df_big, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
            cbar_kws={'label': "Spearman's r"}, ax=axes[1])
axes[1].set_title("Spearman's Correlation (Big Regions)"); plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right')

for ax in axes:
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        key = text.lower().replace(" ","_")
        if key in ["kst24_full_r5z6", "kst24_galaxy_r5z6"]:
            updated_text = re.sub(r"r5z6", "", text, flags=re.IGNORECASE)
            new_labels.append(updated_text)
        else:
            new_labels.append(text)
    ax.set_yticklabels(new_labels)
    for label in ax.get_yticklabels():
        if "Dragon" in label.get_text():
            label.set_color("blue")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, f'August_Big_Regions_New_Heatmap_amp_spearmans_striated_thresh_{thresh}.png'),
            dpi=100, bbox_inches='tight')

# ============================================================
# 14) Example region comparisons & T–T plots (KST24 in Q1; UF23 cre10 in Q4)
# ============================================================

def tt_overlay(x_list, y_list, names, x_label, y_label, out_png):
    styles = {
        names[0]: {"edgecolors":"green","facecolors":"none","s":100, "color":"green"},
        names[1]: {"color":"black","alpha":0.6,"s":20},
        names[2]: {"color":"red","alpha":0.6,"s":20},
    }
    plt.figure(figsize=(10,10))
    for x_data, y_data, name in zip(x_list, y_list, names):
        mask = (x_data != hp.UNSEEN) & (y_data != hp.UNSEEN)
        xv, yv = x_data[mask], y_data[mask]
        slope, intercept = np.polyfit(xv, yv, 1)
        rho, _ = stats.spearmanr(xv, yv)
        lbl = f'{name} (ρ={rho:.2f}, m={slope:.2f})'
        st = styles[name]
        if name == names[0]:
            plt.scatter(xv, yv, edgecolors=st["edgecolors"], facecolors=st["facecolors"], s=st["s"], label=lbl)
        else:
            plt.scatter(xv, yv, color=st["color"], alpha=st["alpha"], s=st["s"], label=lbl)
        fit_line = slope * xv + intercept
        plt.plot(np.sort(xv), np.sort(fit_line), color=st["color"], linewidth=2)
    plt.xlabel(x_label); plt.ylabel(y_label)
    plt.grid(True); plt.axis("equal"); plt.legend()
    plt.savefig(out_png, dpi=150); plt.show()

# Q1 family vs KST24 full Dragon
kst24_full = KST24_full_PI_Dragon.copy()
cbass_Q1   = Real_data_1.copy()
cbass_Q1a  = Real_data_1.copy()
cbass_Q1b  = Real_data_1.copy()
kst24_Q1   = kst24_full.copy(); kst24_Q1[masks_Qs["Q_1"]==0] = hp.UNSEEN
kst24_Q1a  = kst24_full.copy(); kst24_Q1a[masks_Qs["Q_1a"]==0] = hp.UNSEEN
kst24_Q1b  = kst24_full.copy(); kst24_Q1b[masks_Qs["Q_1b"]==0] = hp.UNSEEN
cbass_Q1[masks_Qs["Q_1"]==0]   = hp.UNSEEN
cbass_Q1a[masks_Qs["Q_1a"]==0] = hp.UNSEEN
cbass_Q1b[masks_Qs["Q_1b"]==0] = hp.UNSEEN
tt_overlay([cbass_Q1, cbass_Q1a, cbass_Q1b],
           [kst24_Q1, kst24_Q1a, kst24_Q1b],
           ["Q1","Q1a","Q1b"], "CBASS Real Data", "KST24 (PI Dragon)", os.path.join(out_dir,"KST24_Q1_overlay.png"))

# Q4 family vs UF23 cre10 striated (Dragon)
uf23_cre10 = uf23_cre10_5_striated.copy()
cbass_Q4   = Real_data_1.copy()
cbass_Q4a  = Real_data_1.copy()
cbass_Q4b  = Real_data_1.copy()
uf23_Q4    = uf23_cre10.copy(); uf23_Q4[masks_Qs["Q_4"]==0]   = hp.UNSEEN
uf23_Q4a   = uf23_cre10.copy(); uf23_Q4a[masks_Qs["Q_4a"]==0] = hp.UNSEEN
uf23_Q4b   = uf23_cre10.copy(); uf23_Q4b[masks_Qs["Q_4b"]==0] = hp.UNSEEN
cbass_Q4[masks_Qs["Q_4"]==0]   = hp.UNSEEN
cbass_Q4a[masks_Qs["Q_4a"]==0] = hp.UNSEEN
cbass_Q4b[masks_Qs["Q_4b"]==0] = hp.UNSEEN
tt_overlay([cbass_Q4, cbass_Q4a, cbass_Q4b],
           [uf23_Q4, uf23_Q4a, uf23_Q4b],
           ["Q4","Q4a","Q4b"], "CBASS Real Data", "UF23 cre10 striated Dragon (PI)",
           os.path.join(out_dir,"UF23_cre10_Q4_overlay.png"))

print("\n✅ Pipeline complete. All tables and plots saved in:", out_dir)
