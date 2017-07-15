    #Copyright 2017 Juergen Probst
    #This program is free software; you can redistribute it and/or modify
    #it under the terms of the GNU General Public License as published by
    #the Free Software Foundation; either version 3 of the License, or
    #(at your option) any later version.

    #This program is distributed in the hope that it will be useful,
    #but WITHOUT ANY WARRANTY; without even the implied warranty of
    #MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    #GNU General Public License for more details.

    #You should have received a copy of the GNU General Public License
    #along with this program. If not, see <http://www.gnu.org/licenses/>.

from __future__ import division, print_function
from os import path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import InterpolatedUnivariateSpline
import glob
import log

class PlotOverlay:
    def __init__(self):
        """A simple class that adds scatter points connected with an interpolated spline to a plot"""
        self.values = None
        self.scat = None
        self.interp_curve = None
        self.ax = None
        self.color = None

    def __del__(self):
        if self.scat:
            self.scat.remove()
        if self.interp_curve:
            self.interp_curve.remove()

    def add_to_axis(self, axis, color=None):
        if axis is None:
            return
        self.ax = axis
        if color is None:
            if self.color is None:
                self.color = self.ax._get_patches_for_fill.get_next_color()
        else:
            self.color = color
            # TODO: if color changes when already plotted, update self.scat and self.interp_curve

        if self.values is not None:
            self.update_plot()
            self.update_interpolation_curve()

    def set_data(self, data):
        """data values of -1 will be masked, i.e. not plotted"""
        self.values = np.copy(data)
        self.update_plot()
        self.update_interpolation_curve()

    def has_data(self):
        return self.values is not None and len(self.values) > 0 and np.any(self.values != -1)

    def update_plot(self):
        if self.ax is None:
            return
        X = np.nonzero(self.values != -1)[0]
        if self.scat is None:
            # create new scatter plot:
            self.scat = self.ax.scatter(
                X, self.values[X], facecolors='none', s=100, edgecolors=self.color)
        else:
            # update scatter plot:
            self.scat.set_offsets(np.stack((X, self.values[X])).T)

    def update_interpolation_curve(self):
        if self.ax is None:
            return
        X = np.nonzero(self.values != -1)[0]
        if len(X) <= 3:
            # too little points. Don't plot / remove plot:
            if self.interp_curve is not None:
                self.interp_curve.remove()
                self.interp_curve = None
            return

        # interpolate data, using cubic spline:
        try:
            f_interp = InterpolatedUnivariateSpline(X, self.values[X], k=3)
        except:
            raise ValueError('could not interpolate', X, self.values[X])
        # small steps in x for the interpolated curve:
        xs = np.linspace(0, len(self.values), 400)
        ys = f_interp(xs)
        # plot/update interpolated curve:
        if self.interp_curve is None:
            self.interp_curve, = self.ax.plot(
                xs, ys, ':', color=self.color,
                lw=2, alpha=0.7, label='interpolated')
        else:
            self.interp_curve.set_data(xs, ys)


class ModePicker:
    def __init__(self, outfile, modefile='bands.dat'):
        """Load band diagram data from meep output file. Call the plot method
        to manually select and show frequencies belonging to a band.

        param outfile:
            the meep output
        param modefile:
            File where the picked mode frequencies are saved to or are loaded from.
            Frequencies for every k-vec are space-separated, while each line
            corresponds to a band.
            The file must not exist, it will be created when the band freqs are saved.

        In both filenames, you may use wildcards (*, ?), but this will only find
        existing files (modefile will be overwritten!)

        """
        if glob.has_magic(outfile):
            names = glob.glob(outfile)
            if len(names) == 0:
                raise BaseException("Could not open file: %s" % outfile)
            elif len(names) > 1:
                print('Warning: Globbing found multiple matching filenames, but will only use first one: %s' % names[0])
            # only load the first one found:
            outfile = names[0]
        self.outfile = outfile
        dirname = path.dirname(self.outfile)

        if glob.has_magic(modefile):
            names = glob.glob(path.join(dirname, modefile))
            if len(names) == 0:
                raise BaseException("Could not find file: %s" % modefile)
            elif len(names) > 1:
                print('Warning: Globbing found multiple matching filenames, but will only use first one: %s' % names[0])
            self.modefile = names[0]
        else:
            self.modefile = path.join(dirname, modefile)

        self.kdata, self.hdata = load_bands_data(outfile, freq_tolerance=-1, phase_tolerance=1001)
        self.bands = [PlotOverlay()]
        self.current_band_index = 0
        self.active_band = self.bands[-1]
        self.fig = None
        self.ax = None
        self.load_mode_freqs()

    def __del__(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def on_pick(self, event):
        # get event data:
        try:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            # print('event.ind:', ind, thisline.indexes.shape)
        except AttributeError:
            print('error getting event data')
            return

        # get data saved to the artist in pymeep.postprocess.plot_bands:
        data = {}
        for i in ind:
            kindex = thisline.indexes[0, i]
            bandindex = thisline.indexes[1, i]
            # kvec = thisline.kdata[kindex]
            freq = thisline.hdata[kindex, bandindex, 0]
            if not kindex in data:
                # Make new entry, with the kindex as key;
                # The value is a list of length 2, with first the number of frequencies found so far at this
                # kindex, and second the sum of these frequencies (later, we'll take the arithmetic mean of
                # these frequencies)
                data[kindex] = [1, freq]
            else:
                # update the entry:
                count, fsum = data[kindex]
                data[kindex] = [count + 1, fsum + freq]

        # merge data from multiple points on same k-vec (mean of frequencies):
        current = self.active_band.values
        for k, (count, fsum) in data.items():
            # remove frequency if already in self.mode_freqs:
            if current[k] == fsum / float(count):
                current[k] = -1
            else:
                # update self.mode_freqs with new data:
                current[k] = fsum / float(count)
        self.active_band.set_data(current)

        # update scatter plot:
        self.update_plot_with_picked_data()

    def new_band(self, event=None):
        count = len(self.active_band.values)
        self.bands.append(PlotOverlay())
        self.current_band_index = len(self.bands) - 1
        self.active_band = self.bands[self.current_band_index]
        self.active_band.set_data(np.ones(count) * -1)
        self.active_band.add_to_axis(self.ax)
        self.bcurr.label.set_text(str(self.current_band_index))
        self.bcurr.color = self.active_band.color
        self.bcurr.hovercolor = self.active_band.color
        self.bcurr.ax.set_facecolor(self.active_band.color)
        self.fig.canvas.draw_idle()

    def prev_band(self, event=None):
        self.current_band_index -= 1
        if self.current_band_index < 0:
            self.current_band_index = len(self.bands) - 1
        self.active_band = self.bands[self.current_band_index]
        self.bcurr.label.set_text(str(self.current_band_index))
        self.bcurr.color = self.active_band.color
        self.bcurr.hovercolor = self.active_band.color
        self.bcurr.ax.set_facecolor(self.active_band.color)
        self.fig.canvas.draw_idle()

    def next_band(self, event=None):
        self.current_band_index += 1
        if self.current_band_index >= len(self.bands):
            self.current_band_index = 0
        self.active_band = self.bands[self.current_band_index]
        self.bcurr.label.set_text(str(self.current_band_index))
        self.bcurr.color = self.active_band.color
        self.bcurr.hovercolor = self.active_band.color
        self.bcurr.ax.set_facecolor(self.active_band.color)
        self.fig.canvas.draw_idle()

    def plot(self, y_range=(0.35, 0.65), gap=None,
             coldataindex=3, coldatalabel='abs(mode amplitude)', maxq=2, show=True):
        if self.fig is None:
            plot_bands(
                self.kdata, self.hdata, draw_light_line=True, y_range=y_range, gap=gap,
                coldataindex=coldataindex, coldatalabel=coldatalabel, maxq=maxq, onpick=self.on_pick,
                filename=None, show=False)
            self.ax = plt.gca()
            self.fig = plt.gcf()
            for band in self.bands:
                band.add_to_axis(self.ax)
        if show:
            size = self.fig.get_size_inches()
            self.fig.set_size_inches([size[0], size[1] * 1.2])
            plt.subplots_adjust(bottom=0.2)
            axnew = plt.axes([0.04, 0.05, 0.075, 0.075])
            axprev = plt.axes([0.135, 0.05, 0.075, 0.075])
            axcurr = plt.axes([0.215, 0.05, 0.025, 0.075])
            axnext = plt.axes([0.245, 0.05, 0.075, 0.075])
            axsave = plt.axes([0.34, 0.05, 0.075, 0.075])

            self.bnew = Button(axnew, 'new\nband')
            self.bprev = Button(axprev, 'previous\nband')
            self.bcurr = Button(
                axcurr, str(self.current_band_index),
                color=self.active_band.color, hovercolor=self.active_band.color)
            self.bnext = Button(axnext, 'next\nband')
            self.bsave = Button(axsave, 'save\nbands')

            self.bnew.on_clicked(self.new_band)
            self.bprev.on_clicked(self.prev_band)
            self.bnext.on_clicked(self.next_band)
            self.bsave.on_clicked(self.save_mode_freqs)

            plt.show()

    def save_plot_to_file(
            self, filename, y_range=(0.35, 0.65), gap=None,
            coldataindex=3, coldatalabel='abs(mode amplitude)', maxq=2):
        if self.fig is None:
            self.plot(y_range, gap, coldataindex, coldatalabel, maxq, show=False)
        plt.tight_layout()
        self.fig.savefig(
            filename, transparent=False,
            bbox_inches='tight', pad_inches=0)

    def save_mode_freqs(self, event=None):
        count = len(self.bands)
        # don't save last empty band
        if not self.bands[-1].has_data():
            count -= 1
        mode_freqs = np.ones((count, len(self.bands[0].values))) * -1
        for i in range(count):
            mode_freqs[i] = self.bands[i].values
        np.savetxt(self.modefile, mode_freqs)

    def load_mode_freqs(self):
        """load freqs from self.modefile (every line is one band, frequencies are split by spaces).
        The current bands will be overwritten.

        """
        if path.isfile(self.modefile):
            mode_freqs = np.loadtxt(self.modefile, ndmin=2)
        # patch to update old-style, single-band modefiles:
        if mode_freqs.shape[0] > 1 and mode_freqs.shape[1] == 1:
            mode_freqs = mode_freqs.T
        count = mode_freqs.shape[0]
        # delete excessive bands:
        while len(self.bands) > count:
            del self.bands[-1]
        # add lacking bands:
        while len(self.bands) < count:
            self.bands.append(PlotOverlay())
        for i, band in enumerate(self.bands):
            band.set_data(mode_freqs[i])
            band.add_to_axis(self.ax)
        self.current_band_index = 0
        self.update_plot_with_picked_data()

    def overlay_freqs_positions(self, mode_freqs, color='g', size=100):
        if self.ax is None:
            return
        if isinstance(mode_freqs, PlotOverlay):
            mode_freqs = mode_freqs.values
        X = np.nonzero(mode_freqs != -1)[0]
        return self.ax.scatter(X, mode_freqs[X], facecolors='none', edgecolors=color, s=size)

    def initialize_from_approx_freqs(self, mode_freqs, bandnum=-1):
        """Snap frequencies provided in band_freqs to nearest mode in data with low error and high amplitude.
        If bandnum == -1, a new band will be added (or last band overwritten if it only contains -1 freqs),
        otherwise bandnum specifies which band to overwrite.

        """
        if isinstance(mode_freqs, PlotOverlay):
            mode_freqs = mode_freqs.values
        count = len(self.active_band.values)
        if bandnum == -1 or bandnum >= len(self.bands):
            if bandnum == -1 and not self.bands[-1].has_data():
                # remove empty last band, but reuse color:
                color = np.copy(self.bands[-1].color)
                del self.bands[-1]
            else:
                color = None
            self.bands.append(PlotOverlay())
            self.bands[-1].add_to_axis(self.ax, color)
            dest = len(self.bands) - 1
        else:
            dest = bandnum

        self.active_band = self.bands[dest]
        self.current_band_index = dest
        newvals = np.ones(count) * -1

        for k, freq in enumerate(mode_freqs[:count]):
            if freq == -1:
                continue
            # maximum amplitude of modes at k, needed for normalization:
            maxamp = self.hdata[k, :, 3].max()
            # rate modes in hdata:
            rating = np.zeros(len(self.hdata[k]))
            for i in range(len(rating)):
                if self.hdata.mask[k, i, 0]:
                    # ignore masked frequency:
                    rating[i] = -np.inf
                    continue
                # higher frequency distance leads to worse rating:
                rating[i] -= abs(freq - self.hdata[k, i, 0]) * 500
                # higher amplitude gives higher rating:
                rating[i] += self.hdata[k, i, 3] / maxamp
                # log of error (which is negative) is substracted from rating:
                rating[i] -= np.log(self.hdata[k, i, 6]) / 10
            winner = rating.argmax()
            newvals[k] = self.hdata[k, winner, 0]

        self.active_band.set_data(newvals)
        self.update_plot_with_picked_data()

    def update_plot_with_picked_data(self):
        if self.ax is None:
            return
        #plt.draw()
        self.fig.canvas.draw_idle()

def load_flux_data(filename):
    """Load flux data from meep output of a band simulation.

    :return:
        two-dimensional numpy array, shape=(number of frequencies, 1 + number of flux planes):
            columns: number of frequencies
            first row: frequency
            other rows: flux, one column for each flux plane
    """

    if glob.has_magic(filename):
        names = glob.glob(filename)
        if len(names) == 0:
            raise BaseException("Could not open file: %s" % filename)
        elif len(names) > 1:
            print('Warning: Globbing found multiple matching filenames, but will only use first one.')
        # only load the first one found:
        filename = names[0]

    # load the flux data:
    data = []
    with open(filename) as f:
        for line in f:
            if line.startswith('flux1:'):
                kline = line.split(', ')[1:]
                data.append([float(d) for d in kline])
    return np.array(data)

def load_bands_data(
        filename, freq_tolerance=0.01, phase_tolerance=0.3,
        lattice_vector=(np.sqrt(3.0)/2.0, 0.5, 0), maxbands = 40):
    """Load harminv data and kvecs from meep output of a band simulation.

    The simulation can have one or two harminv output points.
    After the harminv data output, the k-vector components must be output on a line prefixed 
    with 'freqs:, ', like here: 
    https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg02965.html.
    If two harminv outputs, the harminv data will be filtered, such that only the 
    best matching frequencies (within freq_tolerance) whose amplitudes are correctly phase related 
    will be returned, based on discussion from here:
    https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg02971.html
        
    :param freq_tolerance:
        only frequencies whose difference between the two points is within this 
        tolerance will be considered. Ignored if only one harminv output in file.
    :param phase_tolerance:
        frequencies whose amplitudes are not related between the two points by
        exp(ikx) within this tolerance will be rejected 
        (k being the k vector and x the lattice vector).
        Ignored if only one harminv output in file.
    :param lattice_vector:
        the lattice vector between the two harminv points. duh.
        Ignored if only one harminv output in file.
    :param maxbands: 
        maximum number of bands to load; bands above this limit will be truncated
        (with a warning), unused bands upto this limit will be filled up with -1 and masked. 

    :return:
        two numpy arrays in a tuple: (k-vectors, harminv data)
        k-vectors dimension: (number of simulated kvecs, 3)
        harminv data dimension: (number of simulated kvecs, maxbands, 7),
        with the last dimension consisting of 
        real frequency, imag. freq., Q, absolute amplitude, real amplitude, imag. amplitude, harminv error
    """

    if glob.has_magic(filename):
        names = glob.glob(filename)
        if len(names) == 0:
            raise BaseException("Could not open file: %s" % filename)
        elif len(names) > 1:
            print('Warning: Globbing found multiple matching filenames, but will only use first one.')
        # only load the first one found:
        filename = names[0]
        
    # first load the kvec data, 
    # and while we're at it, also check how many harminv outputs we have:
    kvecs = []
    harminv_outputs=0
    first_harminv_line = ''
    for line in open(filename):
        if line.startswith('freqs:'):
            kline = line.split(', ')[2:5]
            kvec = np.array([float(f) for f in kline])
            kvecs.append(kvec)
        elif line.startswith('harminv'):
            if not first_harminv_line:
                # The first line starting with 'harminv'. This should be a table header:
                first_harminv_line = line
                harminv_outputs = 1
            else:
                # We found a line starting with harminv before. 
                # If it is exactly the same than the first harminv line we found, 
                # i.e. also a header with the same number behind 'harminv', then
                # it must be a harminv output at another point.
                if line == first_harminv_line:
                    harminv_outputs += 1
    kvecs = np.array(kvecs)        
    #print('found harminv outputs at %i points' % harminv_outputs)
    if not harminv_outputs:
        raise BaseException("No harminv data found in file %s" % filename)
    elif harminv_outputs > 2:
        raise BaseException(
            "More than 2 harminv outputs in file %s\nSorry, not implemented yet." % filename)

    # now load the harminv data

    # 3rd dim: frequency, imag. freq., Q, abs amp, amp.real, amp.imag, harminv error:
    harminvs1 = -1 * np.ones((len(kvecs), maxbands, 7)) 
    harminvs2 = -1 * np.ones_like(harminvs1)
    harminvs1[:, :, -1] = 1 # unspecified error should not be negative
    harminvs2[:, :, -1] = 1
    kind = 0 # current k index
    bind = 0 # current band index
    nhead = 0 # number of harminv headers encountered

    for line in open(filename):
        if line.startswith('harminv'):
            hline = line.split(', ')[1:]
            if hline[0] == 'frequency':
                # found 'header': "harminv1:, frequency, imag. freq., Q, |amp|, amplitude, error"
                nhead += 1
                if harminv_outputs == 2 and freq_tolerance == -1 and nhead % 2 == 0:
                    # ignore the second header:
                    continue
                bind = 0
                # skip 'header':
                continue
            if bind < maxbands:
                if harminv_outputs == 2 and freq_tolerance != -1:
                    h = [harminvs2, harminvs1][nhead % 2][kind, bind]
                else:
                    h = harminvs1[kind, bind]
                h[:4] = np.array([float(f) for f in hline[:4]])
                amp = complex(hline[4].replace('i', 'j'))
                h[4] = amp.real
                h[5] = amp.imag
                err = complex(hline[5].replace('i', 'j'))
                h[6] = err.real
            else:
                print('WARNING, at kvec #%i, point %i: skipping band %i' % \
                    (kind + 1, 2 - (nhead % 2), bind + 1))
            bind += 1

        if line.startswith('freqs:'):
            kind += 1
            #print 'finished loading kvec #%i' % kind
            
    if harminv_outputs == 2 and freq_tolerance != -1:
        # filter the frequencies:
        harminvs1 = filter_frequencies(
            kvecs, harminvs1, harminvs2, 
            phase_tolerance=phase_tolerance, 
            freq_tolerance=freq_tolerance,
            lattice_vector=lattice_vector)
    return kvecs, defrag_freqs(harminvs1)


def shipping(flist1, flist2, frequency_tolerance = 0.01):
    """Make a 'shipping diagramm' ;P, matching the frequencies in flist1 to flist2.
    
    Frequencies of -1 will be ignored.
    
    :param: flist1, flist2: 1d arrays with frequencies to match
    :param: frequency_tolerance: frequency differences bigger than this will not be matched.
    
    :return: an array with the same length than flist1. 
        Its entries are the index of the matching line of flist2, 
        or -1 if no match was found for the particular frequency in flist1
        
    """
    # build points table:
    points_table = np.zeros((len(flist1), len(flist2)))
    for i, f1 in enumerate(flist1):
        if f1 == -1:
            continue
        for j, f2 in enumerate(flist2):
            d = abs(f2 - f1)
            if d > frequency_tolerance or f2 == -1:
                continue
            if d == 0:
                points_table[i, j] = float('inf')
            points_table[i, j] = 1.0 / d
            
    # build list of matches; first match pair with highest score
    result = np.ones_like(flist1, dtype=np.int) * -1
    index = np.unravel_index(np.argmax(points_table), points_table.shape)
    while points_table[index] != 0:
        # best match:
        result[index[0]] = index[1]
        # clear matched line and column in points_table:
        points_table[index[0], :] = 0
        points_table[:, index[1]] = 0
        index = np.unravel_index(np.argmax(points_table), points_table.shape)
        
    return result

def filter_frequencies(kvec_data, harminv1_data, harminv2_data, phase_tolerance=0.3, freq_tolerance=0.01,
                      lattice_vector=(np.sqrt(3.0)/2.0, 0.5, 0)):
    """Filter harminv data from two harminv points, separated by a lattice vector.
    
    Based on discussion from here:
    https://www.mail-archive.com/meep-discuss@ab-initio.mit.edu/msg02971.html
    
    :param kvec_data:
        k-vectors numpy array with dimension: (number of simulated kvecs, 3)
    :param harminv1_data, harminv2_data:
        harminv data numpy arrays with dimension: 
        (number of simulated kvecs, number of bands, 6),
        with the last dimension consisting of 
        real frequency, imag. freq., Q, real amplitude, imag. amplitude, harminv error;
        Note: All frequencies of -1 within this data will be ignored & masked
    :param freq_tolerance:
        only frequencies whose difference between the two points is within this 
        tolerance will be considered
    :param phase_tolerance:
        frequencies whose amplitudes are not related between the two points by
        exp(ikx) within this tolerance will be rejected 
        (k being the k vector and x the lattice vector)
    :param lattice_vector:
        the lattice vector. duh. 
        
    :return:
        a masked numpy array with the filtered frequencies. 
        Dimension: (number of simulated kvecs, number of bands in harminv1_data, 6),
        with the last dimension consisting of 
        real frequency, imag. freq., Q, real amplitude, imag. amplitude, harminv error;
        All rejected frequencies will be -1 and masked.
    
    """
    R = np.array(lattice_vector) # lattice vector
    
    # 3rd dimension: (freq (re & im), Q-value, amplitude (re & im), error):
    result = -1 * np.ones((len(kvec_data), harminv1_data.shape[1], harminv1_data.shape[2]))
    
    for k, kvec in enumerate(kvec_data):
        # match up the frequencies:
        mlist = shipping(harminv1_data[k, :, 0], harminv2_data[k, :, 0], freq_tolerance)
        for i, match in enumerate(mlist):
            if match == -1:
                # harminv1_data[k, i] has no match in harminv2_data
                continue
            # found a pair with approx. same frequencies
            # now check if their amplitudes have correct phase shift
            amp1 = complex(*harminv1_data[k, i, 3:5])
            amp2 = complex(*harminv2_data[k, match, 3:5])
            # expected phase shift:
            pe = 2 * np.pi * np.dot(R, kvec)
            # actual phase shift:
            pa = np.angle(amp2 / amp1)
            pd = abs(pa - pe)
            while pd > np.pi:
                pd -= 2 * np.pi
            if abs(pd) < phase_tolerance:
                result[k, i, :] = harminv1_data[k, i, :]
                #print('k%i: accepted freq %f -> phase shift (%f)' % (k, harminv1_data[k, i, 0], abs(pd)))
            else:
                #print('k%i: dropped freq %f because of wrong phase shift (%f)' % (k, harminv1_data[k, i, 0], abs(pd)))
                pass

    # mask invalid values:
    result = np.ma.masked_array(result, result == -1)
    return result

def defrag_freqs(freqs):
    """Make the frequency array returned from filter_frequencies more compact by removing as many
    masked frequencies as possible and shrinking the array."""
    # count largest number of unmasked bands (with frequencies != -1) among all k-vecs:
    # start with minimum possible number:
    maxb = 0
    for i in range(freqs.shape[0]):
        # number of bands at this k-vec:
        if isinstance(freqs, np.ma.MaskedArray):
            numb = np.count_nonzero(np.logical_not(freqs.mask[i, :, 0]))
        else:
            numb = np.count_nonzero(freqs[i, :, 0] + 1)
        maxb = max(numb, maxb)
        
    result = -1 * np.ones((freqs.shape[0], maxb, freqs.shape[2]))
    result[:, :, -1] = 1 # unspecified error should not be negative
    for i in range(freqs.shape[0]):
        k = 0
        for j in range(freqs.shape[1]):
            if freqs[i, j, 0] != -1:
                result[i, k, :] = freqs[i, j, :]
                k += 1
    return np.ma.masked_array(result, result == -1)

def get_steps_dict(containing_folder, jobname_regex, datafile_suffix, regroup_to_key_func=lambda x:int(x)):
    """Return a dict with simulation-steps as keys and the data file names as values
    
    regroup_to_key_func: this function will take the first group extracted by jobname_regex and
    its return value will be the dict key."""
    from os import listdir
    import re
    # make list of all subfolders (non-recursive):
    folders = filter(lambda p : path.isdir(path.join(containing_folder, p)), listdir(containing_folder))

    if not folders:
        return dict()
    # build the dict with integer-steps as keys and the data file names as values:
    fdict = dict()
    for fname in folders:
        m = re.match(jobname_regex, fname)
        if m is not None and len(m.groups()) > 0:
            file = path.join(containing_folder, fname, fname + datafile_suffix)
            fdict[regroup_to_key_func(m.groups()[0])] = file
            
    return fdict


def plot_bands(
        kvecs, harminv_data, draw_light_line=False, maxq=None, 
        filename=None, onpick=None, modes_with_calculated_patterns=None, mode_patterns_to_be_calculated=None,
        y_range=None, gap=None, coldataindex=1, coldatalabel='abs(freq.imag)', show=True):
    """Plot bands of triangular lattice from meep-harminv simulation."""
    fig = plt.figure(figsize=(14, 5))
    if filename is None and onpick is not None:
        fig.canvas.mpl_connect('pick_event', onpick)
    kaxis = range(len(kvecs))

    # default: imaginary part of frequency, which is proportional to the decay rate:
    coldata = abs(harminv_data[:, :, coldataindex])# / np.max(abs(harminv_data[:, :, coldataindex]))
    if coldataindex == 6: # make error more linear:
        coldata = -np.log10(coldata)
    sizedata = abs(np.log(abs(harminv_data[:, :, 3]) / np.max(harminv_data[:, :, 3])))
    sizedata = 60 - 55 * sizedata / np.max(sizedata)

    if gap is not None:
        plt.fill([0, len(kvecs), len(kvecs), 0], [0, 0, gap[0], gap[0]], 'b', alpha=0.15)
        plt.fill([0, len(kvecs), len(kvecs), 0], [gap[1], gap[1], 100, 100], 'b', alpha=0.15)

    if maxq is None:
        maxq = np.max(coldata)
    elif np.max(coldata) > maxq:
        log.warning('given maxq (%f) for color data is smaller than data maximum (%f)' % (maxq, np.max(coldata)))
    #print(maxq)
    for bind in range(harminv_data.shape[1]):
        if draw_light_line:
            plt.plot(kaxis, np.linalg.norm(kvecs[kaxis, :], axis=1), c='gray')
        scat = plt.scatter(
            kaxis, harminv_data[:, bind, 0], s=sizedata[:, bind], cmap='jet',
            c=coldata[:, bind], vmin=0, vmax=maxq, 
            facecolors='none', alpha=0.75)#color=next(colours))
    
    if modes_with_calculated_patterns is not None and len(modes_with_calculated_patterns) > 0:
        # plot pre-calculated modes as green circles:
        plt.scatter(
            modes_with_calculated_patterns[:, 0], modes_with_calculated_patterns[:, 1],
            facecolors='none', edgecolors='g', s=90)

    if mode_patterns_to_be_calculated is not None and len(mode_patterns_to_be_calculated) > 0:
        # plot pre-calculated modes as green circles:
        plt.scatter(
            mode_patterns_to_be_calculated[:, 0], mode_patterns_to_be_calculated[:, 1],
            facecolors='none', edgecolors='r', s=100)

    
    if onpick:
        # add invisible dots for picker
        # (if picker added to plots above, the lines connecting the dots
        # will fire picker event as well) - this is ok if no dots are shown
        # Also, combine banddata so it is in one single dataset. This way,
        # the event will fire only once, even when multiple dots coincide:
        # (but then with multiple indices)
        xnum, bands = harminv_data.shape[:2]
        newdata = np.zeros((4, xnum * bands))
        for i, x in enumerate(kaxis):
            for j, y in enumerate(harminv_data[i, :, 0]):
                if harminv_data.mask[i, j, 0]:
                    newdata[:, i + xnum*j] = [x, -1, i, j]
                else:
                    newdata[:, i + xnum * j] = [x, y, i, j]

        frmt = 'o'
        line, = plt.plot(
            newdata[0], newdata[1], frmt, picker=3, alpha=0, zorder=1000)
        # attach data, so it can be used in picker callback function:
        line.indexes = np.array(newdata[2:, :], dtype=np.int, copy=True)
        line.hdata = harminv_data
        line.kdata = kvecs


    if y_range is None:
        plt.ylim(0, np.max(harminv_data[:, :, 0]))
    else:
        plt.ylim(y_range[0], y_range[1])
    plt.xlim(kaxis[0], kaxis[-1])
    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar1 = plt.colorbar(scat, format="%.3f")
    cbar1.ax.set_ylabel(coldatalabel)
    plt.tight_layout()
    if filename is not None:
        fig.savefig(
            filename, transparent=False,
            bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
