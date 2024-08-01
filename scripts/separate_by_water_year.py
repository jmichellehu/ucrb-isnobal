#!/usr/bin/env python
'''Script to separate daily extracted MODIS albedo geotiffs by water year.'''
import sys
import os
import argparse
import glob
import pathlib as p

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

# Parse command line arguments
parser = argparse.ArgumentParser(description='Script to separate daily extracted MODIS albedo geotiffs by water year.')
parser.add_argument('albedo_indir', help='Input directory containing albedo files')
parser.add_argument('outdir', help='Output directory')
args = parser.parse_args()

albedo_indir = args.albedo_indir
outdir = args.outdir

def extract_water_year(fn):
    '''Determine WY from filename'''
    # Extract date from filename
    dt = p.PurePath(fn).stem.split('_Terra_')[1].split('_')[0]
    # split into year, month, day
    y, m, d = dt[:4], dt[4:6], dt[6:]
    # If the date is before October, the WY is equivalent to the year
    # otherwise, the WY is the year + 1
    if m < '10': wy = int(y)
    else: wy = int(y) + 1
    return wy

def locate_water_year_files(indir, wy):
    '''Locate all files in a water year'''
    print(f'Locating files for WY{wy}')
    # from october through end of dec
    oct_dec_files = h.fn_list(indir, f'westernUS_Terra_{wy-1}1*albedo.tif')
    # from january through end of september
    jan_sept_files = h.fn_list(indir, f'westernUS_Terra_{wy}0*albedo.tif')
    # Combine files in order
    wy_files = oct_dec_files + jan_sept_files 
    return wy_files
    
def __main__():
    # Check if the output directory exists, if not create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Get the list of albedo files
    albedo_files = h.fn_list(albedo_indir, '*.tif')
    print(f'{len(albedo_files)} MODIS albedo files found')
    
    # Get list of the first day of all WYs
    wy_starts = h.fn_list(albedo_indir, 'westernUS_Terra_*1001_albedo.tif')
    wy_ends = h.fn_list(albedo_indir, 'westernUS_Terra_*0930_albedo.tif')
    
    # Check that there are an equal number of starts and ends
    print(f'{len(wy_starts)} WY start files found')
    print(f'{len(wy_ends)} WY end files found')
    
    # Locate the first water year, note the date
    start_dt = p.PurePath(wy_starts[0]).stem.split('_Terra_')[1].split('_')[0]
    start_WY = f'{int(start_dt[:4]) + 1}'
    end_dt = h.fn_list(albedo_indir, f'*{start_WY}0930_albedo.tif')[0].split('_Terra_')[1].split('_')[0]

    last_WY = extract_water_year(wy_ends[-1])
    last_start_dt = h.fn_list(albedo_indir, f'*{last_WY - 1}1001_albedo.tif')[0].split('_Terra_')[1].split('_')[0]
    last_end_dt = p.PurePath(wy_ends[-1]).stem.split('_Terra_')[1].split('_')[0]

    print(f'First water year: WY{start_WY} --- start date: {start_dt} --- end date: {end_dt}')
    print(f'Last water year: WY{last_WY} --- start date: {last_start_dt} --- end date: {last_end_dt}')

    # Loop through the water years and symlink all files in that WY to outdir
    for wy in range(int(start_WY), int(last_WY)+1):
        print(wy)
        if not os.path.exists(f'{outdir}/wy{wy}'):
            print(f'Making water year directory: {outdir}/wy{wy}')
            os.makedirs(f'{outdir}/wy{wy}')
        # Locate files for this water year
        wy_files = locate_water_year_files(albedo_indir, wy)
        
        print(f'Symlinking files for WY{wy}')
        for day_file in wy_files:
            day_name = os.path.basename(day_file)
            sym_name = os.path.join(f'{outdir}/wy{wy}', day_name)
            if not os.path.exists(sym_name):
                os.symlink(day_file, sym_name)
if __name__ == '__main__':
    __main__()