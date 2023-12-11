import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

from statsmodels.stats.libqsturng import  psturng

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 20

# Create data
# SVI, Dropout, MCMC, MCMC+PCA, SSVS, QR
means_acea = np.array([0.32050878954054074, 0.20118250074711308, 0.6729795458797574, 0.4400971670604122, 2.03364171275579, 0.1589135702952164])
stds_acea = np.array([0.0029827585759533504, 0.01721450728978099, 0.12378335592579016, 0.10957709126541404, 1.3133681549804117, 0.001720997326170403])
means_spain = np.array([0.3879571347592224, 0.2623567270937137, 0.57977440031648, 0.6817018786899769, 0.7533900900300368, 0.28836314267836644])
stds_spain = np.array([0.02305707237302801, 0.015045107136093716, 0.23725837307203929, 0.14845452270820267, 0.27643831240789307, 0.005496869946647081])


def perform_games_howell_test(m, s):
    # Perform Games-Howell test

    # Compute pairwise differences
    n = len(m)
    diff = np.zeros((n, n))
    sigma = np.zeros((n, n))
    deg_f = np.zeros((n, n))
    p_value = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            sigma[i,j] = np.sqrt(s[i]**2 + s[j]**2) / np.sqrt(n)
            # Degrees of freedom with Welch's correction
            deg_f[i,j] = np.ceil((n-1)*(s[i]**2 + s[j]**2)**2/(s[i]**4 + s[j]**4))
            # t-value
            t_value = np.abs(m[i] - m[j]) / sigma[i,j]
            # p-value from the studentized range distribution
            p_value[i,j] = psturng(t_value * np.sqrt(2), 6, deg_f[i,j])
            diff[i,j] = np.abs(m[i] - m[j]) / p_value[i,j]
            
    # Plot triangular matrix
    fig, ax = plt.subplots(figsize=(10, 10))  # Set the figsize to your desired size
    im = ax.imshow(p_value, cmap='coolwarm')

    # Set color scales for values below 0.05 and above 0.05
    im.set_clim(vmin=0, vmax=1)  # Color scale for values below 0.05 and above 0.05
    im.set_cmap('coolwarm')  # Set colormap to coolwarm

    # Highlight values not equal to 0
    for i in range(n):
        for j in range(n):
            if i >= j:
                p_value[i, j] = np.nan
            else:
                ax.text(j, i, f'{p_value[i, j]:.2f}', ha='center', va='center', color='white', fontweight='bold')

    # Color pixels with values above 0.05 red and below 0.05 blue
    im.set_clim(vmin=0, vmax=1)  # Color scale for values below 0.05 and above 0.05
    im.set_cmap('coolwarm')  # Set colormap to coolwarm
    im.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True))
    im.set_array(np.where(np.isnan(p_value), np.nan, np.where(p_value > 0.05, 1, 0)))

    # Set ticks as text
    labels = ['SVI', 'Dropout', 'MCMC', 'MCMC+PCA', 'SSVS', 'QR']
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')  # Set rotation angle and alignment
    ax.set_yticklabels(labels, rotation=45, ha='right')

    plt.show()


def perform_welch_anova_test(m, s):
    w = 10 / s**2
    mean_w = (w*m).sum() / w.sum()
    sstr_w = (w * (m - mean_w)**2).sum()
    mstr_w = sstr_w / (len(m) - 1)
    lambda_w = ((1 - w/w.sum())**2).sum() / (3*(6**2-1))

    F_w = mstr_w / (1 + (2*lambda_w*4)/3)

    p_value = scipy.stats.f.cdf(F_w, 5, np.ceil(1/lambda_w))

    print(p_value)


perform_welch_anova_test(means_acea, stds_acea)
perform_games_howell_test(means_acea, stds_acea)

perform_welch_anova_test(means_spain, stds_spain)
perform_games_howell_test(means_spain, stds_spain)


