# Port functions from model_eval.ipynb and generalize
# Add handling for:
# resampling to grids with specified extent, resolution, crs, and resampling method
# default output dict to csv

# def build_ds_pairs(basin, WY, asodir, ending='original', verbose=True):
#     # Pull reprojected depth and depth differnece files for the basin and water year
#     nc_fns = h.fn_list(asodir, f'{basin}_wy{WY}*depth*{ending}.nc')

#     # Extract the unique dates
#     dates = set(f.split('_')[-3] for f in nc_fns)
#     # Sort them to be in order
#     dates = sorted(dates)
#     if verbose:
#         print(dates)

#     # Rearrange the filenames by date
#     nc_fns_by_date = {date: [] for date in dates}
#     for fn in nc_fns:
#         date = fn.split('_')[-3]
#         nc_fns_by_date[date].append(fn)

#     if verbose:
#         print(nc_fns_by_date)

#     # read in the files by date
#     ds_by_date = {date: [] for date in dates}
#     for date, fns in nc_fns_by_date.items():
#         date_list = []
#         for fn in fns:
#             ds = xr.open_dataset(fn)
#             date_list.append(ds)
#         ds_by_date[date] = date_list

#     # Pull model title from filenames for each pair within each date, while retaining order
#     model_titles = set(fn.split('_')[-5] for fn in fns)
#     # Convert to list for subscriptability
#     model_titles = list(model_titles)
#     model_titles.sort()

#     if verbose:
#         print(model_titles)

#     return dates, ds_by_date, model_titles

# def clean_arrays(da1, da2, remove_zeros=False):
#     '''Flatten, remove nans and zeros (optionally)'''
#      # Flatten arrays
#     da1 = da1.values.flatten()
#     da2 = da2.values.flatten()

#     # Remove NaNs
#     mask = ~np.isnan(da1) & ~np.isnan(da2)
#     da1 = da1[mask]
#     da2 = da2[mask]

#     # Remove zeroes
#     if remove_zeros:
#         mask = (da1 == 0) & (da2 == 0)
#         da1 = da1[~mask]
#         da2 = da2[~mask]

#     return da1, da2

# def calc_lrm_stats(aso_depth, model_depth, ax, lims, fit_yloc):
#     # Add a line of best fit
#     m, b = np.polyfit(aso_depth, model_depth, 1)
#     x = np.linspace(lims[0], lims[1], 100)
#     ax.plot(x, m * x + b, 'r-', alpha=0.5, zorder=10, label='Line of Best Fit')

#     # Add the fit equation to the plot
#     fit_eq = f'y = {m:.2f}x + {b:.2f}'
#     # Add the correlation coefficient
#     r = np.corrcoef(aso_depth, model_depth)[0, 1]
#     fit_eq += f'\nr = {r:.2f}'
#     ax.text(0.05, fit_yloc, fit_eq, transform=ax.transAxes,
#             fontsize=10, verticalalignment='top')
#     # Add the coefficient of determination R2
#     r_squared = r ** 2
#     r_squared_text = f'$R^2$ = {r_squared:.2f}'
#     ax.text(0.05, fit_yloc - 0.1, r_squared_text, transform=ax.transAxes,
#             fontsize=10, verticalalignment='top')
#     # Add the number of points
#     n_points = len(aso_depth)
#     n_points_text = f'n = {n_points}'
#     ax.text(0.05, fit_yloc - 0.2, n_points_text, transform=ax.transAxes,
#             fontsize=10, verticalalignment='top')
#     return ax, r

# def calculate_kge(r, obs, model):
#     '''Calculate Kling-Gupta Efficiency (KGE) between two depth arrays'''
#     # KGE is calculated as:
#     # KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
#     # where r is the correlation coefficient, alpha is the ratio of the standard deviations,
#     # and beta is the ratio of the means.
#     kge = 1 - np.sqrt((r - 1) ** 2 + (np.std(model) / np.std(obs) - 1) ** 2 + (np.mean(model) / np.mean(obs) - 1) ** 2)
#     return kge

# # Calculate RMSE first
# def calculate_rmse(model_depth, aso_depth):
#     '''Calculate RMSE between two depth arrays'''
#     rmse = np.sqrt(np.nanmean((model_depth - aso_depth) ** 2))
#     return rmse

# # Now based on a defined normalization factor, calculate the normalized RMSE
# def calculate_nrmse(model_depth, aso_depth, normalization_factor):
#     '''Calculate normalized RMSE between two depth arrays
#     Normalization factors: range of ASO depth, mean ASO depth, standard deviation of ASO depth
#     '''
#     rmse = calculate_rmse(model_depth, aso_depth)
#     if normalization_factor == 'range':
#         normalization_factor = np.nanmax(aso_depth) - np.nanmin(aso_depth)
#     elif normalization_factor == 'mean':
#         normalization_factor = np.nanmean(aso_depth)
#     elif normalization_factor == 'std':
#         normalization_factor = np.nanstd(aso_depth)
#     nrmse = rmse / normalization_factor
#     return nrmse

# def calculate_mae(model_depth, aso_depth):
#     '''Calculate Mean Absolute Error between two depth arrays'''
#     mae = np.nanmean(np.abs(model_depth - aso_depth))
#     return mae

# def scatterplot_pairs(basin, WY, dates, ds_by_date, model_titles, figsize=(16, 4), sharex=True, sharey=True, color='darkblue', fit_yloc=0.95, normalization_factor='range'):
#     # Prepare an empty dict to hold stats for each date_pair
#     stats_dict = dict()
#     # Scatterplot the first date by ds pairs
#     for date in dates:
#         ds_list = ds_by_date[date]
#         _, axa = plt.subplots(1, len(model_titles), figsize=figsize, sharex=sharex, sharey=sharey)
#         for pair_idx in range(len(model_titles)):
#             ax = axa[pair_idx]
#             aso_ds = ds_list[pair_idx * 2]
#             model_ds_diff = ds_list[pair_idx * 2 + 1]
#             model_ds = model_ds_diff['depth_diff'] + aso_ds['aso_depth']
#             model_depth, aso_depth = clean_arrays(model_ds, aso_ds['aso_depth'])

#             # Scatterplot this, adding a 1:1 line, and line of best fit
#             ax.scatter(model_depth,
#                     aso_depth,
#                     color=color,
#                     s=1, alpha=0.05)
#             ax.set_xlabel(f'{model_titles[pair_idx].upper()} Depth (m)')
#             ax.set_ylabel('ASO Depth (m)')
#             # Add a 1:1 line
#             lims = [
#                 min(np.nanmin(aso_depth), np.nanmin(model_depth)),
#                 max(np.nanmax(aso_depth), np.nanmax(model_depth))
#             ]
#             ax.plot(lims, lims, 'k--', alpha=0.5, zorder=10, label='1:1 Line')

#             ax, r = calc_lrm_stats(aso_depth, model_depth, ax, lims, fit_yloc)

#             # Add a legend on the last subplot
#             if pair_idx == len(model_titles) - 1:
#                 ax.legend(loc='lower right')
#             rmse = calculate_rmse(model_depth, aso_depth)
#             nrmse = calculate_nrmse(model_depth, aso_depth, normalization_factor=normalization_factor)
#             mae = calculate_mae(model_depth, aso_depth)
#             kge = calculate_kge(r, aso_depth, model_depth)

#             # Add the RMSE and NRMSE to the plot
#             rmse_text = f'RMSE = {rmse:.2f} m'
#             nrmse_text = f'NRMSE {normalization_factor} = {nrmse:.2f}'
#             mae_text = f'MAE = {mae:.2f} m'
#             kge_text = f'KGE = {kge:.2f}'
#             ax.text(0.05, fit_yloc - 0.25, rmse_text, transform=ax.transAxes,
#                     fontsize=10, verticalalignment='top')
#             ax.text(0.05, fit_yloc - 0.3, nrmse_text, transform=ax.transAxes,
#                     fontsize=10, verticalalignment='top')
#             ax.text(0.05, fit_yloc - 0.35, mae_text, transform=ax.transAxes,
#                     fontsize=10, verticalalignment='top')
#             ax.text(0.05, fit_yloc - 0.4, kge_text, transform=ax.transAxes,
#                     fontsize=10, verticalalignment='top')

#             # Set the x and y limits to the same
#             ax.set_xlim(lims)
#             ax.set_ylim(lims)
#             # Set the aspect ratio to be equal
#             ax.set_aspect('equal')

#             # Add entry to stats_dict
#             # correlation coefficient, coefficient of determination, number of points, rmse, nrmse, mae, kge
#             stats_dict[f'{basin}_{date}_{model_titles[pair_idx].upper()}'] = [r, r**2, len(aso_depth), rmse, nrmse, mae, kge]
#         plt.suptitle(f'{basin.capitalize()} WY {WY} {date}')
#     return stats_dict