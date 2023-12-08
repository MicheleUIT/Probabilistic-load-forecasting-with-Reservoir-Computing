import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
            sigma[i,j] = np.sqrt(s[i]**2 + s[j]**2) / np.sqrt(2*n)
            # Degrees of freedom with Welch's correction
            deg_f[i,j] = (n-1)*(s[i]**2 + s[j]**2)**2/(s[i]**4 + s[j]**4)
            # t-value
            t_value = np.abs(m[i] - m[j]) / sigma[i,j]
            # p-value from the studentized range distribution
            p_value[i,j] = psturng(t_value * np.sqrt(2), 6, deg_f[i,j])
            diff[i,j] = np.abs(m[i] - m[j]) / p_value[i,j]
            
    # Plot triangular matrix
    fig, ax = plt.subplots(figsize=(10, 10))  # Set the figsize to your desired size
    im = ax.imshow(p_value, cmap='coolwarm')

    # Set color scales for values below 1 and above 1
    im.set_clim(vmin=0, vmax=2)  # Color scale for values below 1 and above 1
    cbar = ax.figure.colorbar(im, ax=ax, extend='both')  # Add colorbar for values below 1 and above 1
    cbar.set_ticks([0, 1, 2])  # Set colorbar ticks
    cbar.set_ticklabels(['< 1', '1', '> 1'])  # Set colorbar tick labels

    # Set ticks as text
    labels = ['SVI', 'Dropout', 'MCMC', 'MCMC+PCA', 'SSVS', 'QR']
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')  # Set rotation angle and alignment
    ax.set_yticklabels(labels, rotation=45, ha='right')

    # Highlight values not equal to 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{p_value[i, j]:.2f}', ha='center', va='center', color='white', fontweight='bold')

    print(p_value)
    plt.show()


perform_games_howell_test(means_spain, stds_spain)


