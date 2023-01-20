import pdb, sys, multiprocessing, argparse
import matplotlib as mpl
import MDAnalysis as md
import numpy as np
import matplotlib.pyplot as plt

from MDAnalysis import transformations as trans
from multiprocessing import Pool
from functools import partial
from multiprocessing import cpu_count

# global variable for openMP threading
n_jobs = cpu_count()
print(f'{n_jobs} cores')

# MDAnalysis guesses the masses of these OPLS atom types wrong
# You may need to modify, add to, remove, etc. to suit your needs
mass_map = {
    'CT ': 12.01100 ,
    'CSS': 12.01100 ,
    'OG ': 15.99940 ,
    'HG ': 1.00800  ,
    'NI ': 14.00670 ,
    'SO ': 32.06000 ,
    'OS ': 15.99940 ,
    'CF ': 12.01100 ,
    'FC ': 18.99840 ,
    'LI ': 6.94100  ,
    'NA ': 14.0070  ,
    'CR ': 12.0110  ,
    'CW ': 12.0110  ,
    'C1 ': 12.0110  ,
    'HCR':  1.0080  ,
    'C1A': 12.0110  ,
    'HCW':  1.0080  ,
    'H1 ':  1.0080  ,
    'CE ': 12.0110  ,
    'HC ':  1.0080
}

# function to calculate all linear densities for one frame
def dens_frame(frame_index, x, y, box, sel, nbins):
    sel.universe.trajectory[frame_index]
    blah = np.zeros(((x.shape[0]-1)*(y.shape[0]-1), nbins))
    count = -1
    for i,xx in enumerate(x[:-1]):
        x_mask = np.logical_and(sel.positions[:,0] <= x[i+1], sel.positions[:,0] >= xx)
        xd = x[i+1] - x[i]
        for j,yy in enumerate(y[:-1]):
            count += 1
            y_mask = np.logical_and(sel.positions[:,1] <= y[j+1], sel.positions[:,1] >= yy)
            yd = y[j+1] - y[j]
            mask = x_mask & y_mask
            hist, bin_edges = np.histogram(sel.positions[mask,-1], bins=nbins, weights=sel[mask].masses, range=(0,box[-1]))
            norm = 1./(xd * yd * (bin_edges[1]-bin_edges[0]))

            blah[count] = hist * norm
    return blah

def main():
    parser = argparse.ArgumentParser(description='Decomposes a unit cell into an xy 2D spatial grid, then computes 1D mass density profiles for each grid element along z and is time averaged over a simulation trajectory.')

    #=============#
    # code inputs #
    #=============#
    parser.add_argument('intop',  metavar='topo filename', type=str,
                        help='Path to topology file.')
    parser.add_argument('intrj',  metavar='trj filename', type=str,
                        help='Path to trajectory file.')
    parser.add_argument('select_labels',  metavar='selections', type=str,
                        help='A comma separated string of atom selections using the MDAnalysis selection syntax. If you wish to align the linear densities to some baseline, the first selection provided will be used to determine the baseline.')

    #================#
    # optional flags #
    #================#
    parser.add_argument('--xbins', dest='xbins', type=int, default=20,
                        help='Number of xbins.')
    parser.add_argument('--ybins', dest='ybins', type=int, default=20,
                        help='Number of ybins.')
    parser.add_argument('--zbins', dest='zbins', type=int, default=200,
                        help='Number of ybins.')
    parser.add_argument('--framelimit', metavar='N',dest='nframes', default=None, type=int,
                       help="if specified, limit analysis to the first N frames. Good for debugging.")
    parser.add_argument('--align', dest='align', default=False, action='store_true',
                       help="Align all linear densities to the onsets of the first selection provided.")

    # inputs
    args = parser.parse_args()
    intop = args.intop
    intrj = args.intrj
    select_labels = args.select_labels.split(',')
    xbins = args.xbins
    ybins = args.ybins
    zbins = args.zbins
    nframes = args.nframes
    align = args.align

    fprefix = 'lindens'

    print('Calculating densities for the following selections:')
    print(select_labels)

    # load up the system
    u = md.Universe(intop, intrj)
    if not nframes:
        nframes = u.trajectory.n_frames

    # set correct masses for atomtypes
    # may have to modify this according to your FF spec
    print('Setting atom masses')
    for key in mass_map:
        mm = mass_map[key]
        sel = u.select_atoms(f'name {key}')
        sel.masses = mm

    # create selections for each group
    selections = [u.select_atoms(label) for label in select_labels]
    atoms = u.atoms

    # get all atom coordinates into the unit cell
    transform = md.transformations.wrap(u.atoms)
    u.trajectory.add_transformations(transform)
    box = u.dimensions[:3]

    # setup the x and y grids
    x = np.linspace(0, box[0], xbins+1)
    y = np.linspace(0, box[1], ybins+1)
    dx = x[1]*0.5; dy = y[1]*0.5

    # calculate surfaces of GO
    slab = selections[0]
    slab_bounds = np.zeros((x.shape[0]-1,y.shape[0]-1,2))
    print('Calculating surface grid')
    for i,xx in enumerate(x[:-1]):
        x_mask = np.logical_and(slab.positions[:,0] <= x[i+1], slab.positions[:,0] >= xx)
        for j,yy in enumerate(y[:-1]):
            y_mask = np.logical_and(slab.positions[:,1] <= y[j+1], slab.positions[:,1] >= yy)
            mask = np.logical_and(x_mask, y_mask)
            slab_bounds[i,j] = np.array([slab.positions[mask,2].min(), slab.positions[mask,2].max()])

    # plot surfaces to see what is going on
    xtemp = x[:-1] + dx
    ytemp = y[:-1] + dy
    xxx, yyy = np.meshgrid(xtemp,ytemp,indexing='ij')
    zlow = slab_bounds[:,:,0]
    zhigh = slab_bounds[:,:,1]

    plt.close('all')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xxx,yyy,zhigh, alpha=0.5)
    ax.plot_surface(xxx, yyy, zlow, alpha=0.5)
    ax.set_zlim(0,box[2])
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')
    plt.tight_layout()
    fig.savefig(fprefix+'_surfaces.pdf')

    # loop over different selections
    dens_lists = np.zeros((len(selections), nframes, xbins * ybins, zbins))
    bin_edges = np.linspace(0., box[-1], zbins +1)
    for i, (sel,label) in enumerate(zip( selections, select_labels)):
        print(f'Working on {label}')
        # fancy per frame function
        run_per_frame = partial(dens_frame,
                                x=x,
                                y=y,
                                box=box,
                                sel=sel,
                                nbins=zbins
                               )

        # parallel loop
        frame_values = np.arange(nframes)
        with Pool(n_jobs) as worker_pool:
            res = worker_pool.map(run_per_frame, frame_values)

        # get results and normalise
        np.array(res)
        dens_lists[i] = res

    # save non-shifted mean density profiles
    fact = 1.660538921
    z = bin_edges[:-1] + bin_edges[1]*0.5
    out_data = [z,]
    [out_data.append(fact* d.mean(axis=(0,1))) for d in dens_lists]
    out_data = np.array(out_data).T
    fig,ax = plt.subplots()
    for i in range(3):
        ax.plot(out_data[:,0], out_data[:,i+1], '-')
    fig.savefig(fprefix + '_test.pdf')
    np.savetxt( fprefix + '_linear_densities.txt', out_data, header='z '+ ' '.join(select_labels))

    if align:
        # align density data to the upper and lower GO onsets
        ll,ii,jj,kk = dens_lists.shape
        print(dens_lists.shape)
        aligned_dens = np.zeros((ll-1, 2, ii, jj, kk))
        slab_bounds = np.zeros((ii,jj,2))
        for i in range(ii):
            for j in range(jj):
                god = dens_lists[0,i,j]
                onset = np.nonzero(god)[0][0]
                end = np.nonzero(god)[0][-1]
                slab_bounds[i,j] = np.array([onset,end])
                for l in range(ll-1):
                    aligned_dens[l,0,i,j] = np.roll(dens_lists[l+1,i,j,:], -onset)
                    aligned_dens[l,1,i,j] = np.roll(dens_lists[l+1,i,j,:], -end)

        # plot and save the aligned data
        plt.close('all')
        fig, axs = plt.subplots(1,2, figsize=(7,3), sharey=True)

        # lower surface
        zzz = np.concatenate((-np.flip(z),z))
        ddd = []
        for i in range(ll-1):
            tmp = np.flip(aligned_dens[i,0].mean(axis=(0,1))*fact)
            d = np.concatenate((np.roll(tmp, -len(tmp)), tmp))
            axs[0].plot( zzz, d, '-', label=select_labels[i+1])
            ddd.append(d)

        data_out = [zzz,]
        [data_out.append(d) for d in ddd]
        data_out = np.array(data_out).T
        np.savetxt( fprefix + '_lower_aligned_densities.txt', data_out, header = 'z ' + ' '.join(select_labels[1:]))

        axs[0].axvline(0, color='xkcd:grey')
        axs[0].set_xlim(-5,20)
        axs[0].set_xlabel('Distance from lower surface [Å]')
        axs[0].set_ylabel('Mass density [g/cc]')
        axs[0].tick_params(axis="both", which="both", direction="in", top=True, bottom=True, left=True, right=True)
        axs[0].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        axs[0].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        axs[0].set_ylim(0,1.6)

        # upper surface
        ddd = []
        for i,dens in enumerate(aligned_dens):
            tmp = dens[1].mean(axis=(0,1))*fact
            d = np.concatenate((np.roll(tmp, -len(tmp)), tmp))
            axs[1].plot( zzz, d, '-', label=select_labels[i+1])
            ddd.append(d)

        data_out = [zzz,]
        [data_out.append(d) for d in ddd]
        data_out = np.array(data_out).T
        np.savetxt( fprefix + '_upper_aligned_densities.txt', data_out, header = 'z ' + ' '.join(select_labels[1:]))

        axs[1].axvline(0, color='xkcd:grey')
        axs[1].set_xlim(-5,20)
        axs[1].set_xlabel('Distance from upper surface [Å]')
        axs[1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
        axs[1].tick_params(axis="both", which="both", direction="in", top=True, bottom=True, left=True, right=True)
        axs[1].legend(frameon=False)
        plt.tight_layout()

        fig.savefig( fprefix + '_aligned_densities.pdf' )

if __name__=='__main__':
    main()
