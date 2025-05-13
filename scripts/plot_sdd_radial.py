#!/usr/bin/env python
'''Script for radial plot of shift in snow disappearance date

Usage: plot_sdd_radial.py basin wy
'''
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

# Set seaborn palette
sns.set_palette('icefire')
def plot_radial(combo_ds, elevation_bins=None, cmap=None, elev_fontcolor="black",
                title=None, vminnorm=-45, vmaxnorm=45, num_aspect_bins=16, aspect_labels=None,
                verbose=True, outname=None
                ):
    '''Need input for aspect_labels if num_aspect_bins is not 16
    TODO check where the aspec slice edges are
    '''
    aspects = combo_ds['aspect'].values.flatten()
    elevation = combo_ds['dem'].values.flatten()
    peak_swe_shift = combo_ds['sdd_shift'].values.flatten()

    valid_data = ~np.isnan(peak_swe_shift) & ~np.isnan(elevation) & ~np.isnan(aspects)

    # Apply the valid data mask
    aspects = aspects[valid_data]
    elevation = elevation[valid_data]
    peak_swe_shift = peak_swe_shift[valid_data]

    # Step 1: Bin the aspect values into 16 bins (one for each direction)
    aspect_bins = np.linspace(0, num_aspect_bins, num_aspect_bins + 1).astype(int)  # 16 bins for aspect
    aspect_binned = np.digitize(aspects, aspect_bins) - 1  # 0-indexed bins

    # Step 2: Bin the elevation values using the custom bins
    if elevation_bins is None:
        # Equally spaced bins from minimum to maximum elevation rounded to the nearest 100
        binsize = 200
        steps = (round(elevation.max(), ndigits=-2) - round(elevation.min(), ndigits=-2)) / binsize
        elevation_bins = np.linspace(round(elevation.min(), ndigits=-2), round(elevation.max(), ndigits=-2), int(steps + 1))
    elevation_binned = np.digitize(elevation, elevation_bins) - 1  # 0-indexed bins
    # Number of bins
    num_elevation_bins = len(elevation_bins) - 1 # Based on custom elevation bins

    # Step 3: Aggregate peak_swe_shift by both aspect and elevation
    aggregated_values = []
    for aspect_bin in range(num_aspect_bins):
        for elev_bin in range(num_elevation_bins):  # Based on number of elevation intervals
            # Mask for the given aspect and elevation bin
            mask = (aspect_binned == aspect_bin) & (elevation_binned == elev_bin)
            valid_values = peak_swe_shift[mask]

            if len(valid_values) > 0:
                aggregated_values.append(np.mean(valid_values))
            else:
                aggregated_values.append(np.nan)

    aggregated_values = np.array(aggregated_values)

    # Step 4: Create a polar plot where the angle corresponds to aspect,
    # radius corresponds to reversed elevation bin, and color represents peak_swe_shift

    # Set up the polar plot
    _, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 6.5), dpi=300)

    # Angles for each aspect bin
    angles = np.linspace(0, 2 * np.pi, num_aspect_bins, endpoint=False)

    # Reverse the radius so higher elevations are at the center
    radius = np.linspace(1, 0.2, num_elevation_bins)  # Higher elevation closer to the center

    # Set the color map and normalize with symmetric limits around 0
    if cmap is None:
        cmap = sns.palettes.color_palette('rocket', as_cmap=True)
    norm = mcolors.Normalize(vmin=vminnorm, vmax=vmaxnorm)

    # Plot the radial bars without spaces between directions
    bar_width = 2 * np.pi / num_aspect_bins  # Full coverage per aspect

    for aspect_bin in range(num_aspect_bins):
        for elev_bin in range(num_elevation_bins):
            # Calculate the radius based on the reversed elevation bin
            r = radius[elev_bin]

            # Angle for this aspect bin
            theta = angles[aspect_bin]

            # Color based on the aggregated peak_swe_shift value
            color = cmap(norm(aggregated_values[aspect_bin * num_elevation_bins + elev_bin]))

            # Plot the bar at the correct angle and radius with no spacing
            ax.bar(theta, r, width=bar_width, bottom=0, color=color, alpha=0.8)

    # Adjust the orientation of the plot (rotate so North is at the top)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Calculate angles for the grid lines between directions
    half_bin_width = np.pi / num_aspect_bins  # Half-width to position grid lines between directions
    grid_angles = np.linspace(0, 2 * np.pi, num_aspect_bins, endpoint=False) + half_bin_width  # Boundary positions

    # Set grid lines at the boundaries between directions
    ax.set_xticks(grid_angles)  # Align grid lines between aspect bins

    # Define the angles for label placement at the center of each direction
    label_angles = np.linspace(0, 2 * np.pi, num_aspect_bins, endpoint=False)  # Center positions

    # Set labels for each direction only
    if num_aspect_bins == 16:
        aspect_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                        'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    ax.set_xticks(label_angles, labels=aspect_labels, size =11)
    ax.grid(False)  # Turns off all grid lines
    # Set labels at centers without affecting grid lines

    # Optional: Customize grid line appearance for clarity
    #ax.grid(color='grey', linestyle='--', linewidth=0.5)  # Adjust style if desired

    # Reverse radius labels to match reversed elevation bins
    ax.set_yticks(radius)
    ax.set_yticklabels([f"{int(elev)} m" for elev in elevation_bins[:-1]],
                    size=7,
                    color=elev_fontcolor)  # Label each elevation bin
    ax.set_ylim(0, 1)

    # Add colorbar for peak_swe_shift values
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', pad=0.1)
    cbar.set_label('SDD shift (Days)')#, size=16)

    # Add title
    if title is None:
        title='Insert title here'
    ax.set_title(title, va='bottom');#, size=18)
    if outname is not None:
        if verbose:
            print(f'Saving to {outname}')
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def parse_arguments():
    """Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate and plot snow disappearance dates for a given basin and water year.')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='PuOr')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/sdd/radial')
    parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
    return parser.parse_args()

def __main__():
    # Parse command line args
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    sns.set_palette(palette)

    # Set up directories
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'

    # Locate sdd_shift file
    if verbose:
        print('Locating SDD shift file...')
    sdd_shift_fn = h.fn_list(workdir, f'{basin}*/wy{WY}/{basin}_sdd_shift_wy{WY}.nc')[0]
    if verbose:
        print(f'{sdd_shift_fn}\n')
    # Locate terrain file
    if verbose:
        print('Locating terrain file...')
    terrain_fn = h.fn_list(script_dir, f'{basin}*_setup/data/{basin}_terrain.nc')[0]
    if verbose:
        print(f'{terrain_fn}\n')

    # Load the terrain and SDD shift datasets
    if verbose:
        print('Loading datasets...')
    ds = xr.open_dataset(terrain_fn, drop_variables=['spatial_ref', 'hs', 'slope'])
    sdd_diff_ds = xr.open_dataset(sdd_shift_fn)

    # combine the two datasets
    combo_ds = xr.merge([ds, sdd_diff_ds])

    # Derive dem elevation ranges from the terrain dataset using hundreds place rounding and flexible bin number
    if verbose:
        print('Binning elevation data...')
    _, dem_elev_ranges = proc.bin_elev(dem=combo_ds['dem'], basinname=basin,
                                            plot_on=False, round_on=True, p=None, verbose=verbose)
    # Extract elevation bins from the ranges
    # Convert dict values into a flat list
    elevation_bins = [f[0] for jdx, f in enumerate(dem_elev_ranges.values())]
    # add the last bin (max elev)
    elevation_bins.append(list(dem_elev_ranges.values())[-1][-1])
    if verbose:
        print(f'{elevation_bins}\n')
    # Drop this here for now
    num_aspect_bins = 16
    if verbose:
        print('Plotting...')
    outname = f'{outdir}/{basin}_sdd_shift_radial_wy{WY}_{num_aspect_bins}.png'
    title = f'{basin.capitalize()} WY {WY}'
    cmap = sns.palettes.color_palette(palette, as_cmap=True)
    plot_radial(combo_ds=combo_ds, elevation_bins=elevation_bins, cmap=cmap, num_aspect_bins=num_aspect_bins,
                elev_fontcolor="white", title=title, vmaxnorm=45, outname=outname)



if __name__ == "__main__":
    __main__()