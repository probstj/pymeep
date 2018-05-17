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
from os import path, environ, remove, rename, mkdir, uname
import sys
from shutil import rmtree
import subprocess as sp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
import numpy as np
from pymeep import defaults, postprocess, log
from datetime import datetime
import time
from glob import glob1


class Simulation(object): 
    def __init__(
            self, jobname, ctl_template,
            resolution=defaults.default_resolution,
            work_in_subfolder=True, clear_subfolder=True,
            logger=True, quiet=defaults.isQuiet,
            **kwargs):

        self.jobname = jobname
        self.ctl_template = ctl_template
        self.resolution = resolution
        self.quiet = quiet
        # All extra kwargs are going into this object's __dict__.
        # They are supposed to be used in ctl_template:
        self.__dict__.update(kwargs)

        self.work_in_subfolder = work_in_subfolder
        self.clear_subfolder = clear_subfolder
        if isinstance(work_in_subfolder, bool):
            if work_in_subfolder:
                # create default subfolder from jobname:
                self.workingdir = path.abspath(
                    path.join(path.curdir, jobname))
            else:
                # work here, no subfolder:
                self.workingdir = path.abspath(path.curdir)
        else:
            # hopefully a string
            self.workingdir = path.abspath(
                    path.join(path.curdir, work_in_subfolder))

        # the .ctl file that MEEP will use:
        self.ctl_file = jobname + '.ctl'
        # a date & time stamp added to log and output filenames:
        dtstamp = ('_{0.tm_year}-{0.tm_mon:02}-{0.tm_mday:02}'
                   '_{0.tm_hour:02}-{0.tm_min:02}-'
                   '{0.tm_sec:02}').format(time.localtime())
        # the output file, where all MEEP output will go:
        self.out_file = path.join(self.workingdir, jobname + dtstamp + '.out')
        # a log file, where information from pyMEEP will go:
        self.log_file = path.join(self.workingdir, jobname + dtstamp + '.log')
        # the file where MEEP usually saves the dielectric:
        self.eps_file =  path.join(self.workingdir, 'epsilon.h5')

        # logger is not setup yet, because the log file might be placed in a
        # subfolder that still needs to be created. But, I want to log that
        # I created a new directory. So make a simple log buffer:
        to_log = []

        to_log.append('Working in directory ' + self.workingdir)
        if self.work_in_subfolder:
            if path.exists(self.workingdir):
                to_log.append('directory exists already: ' + self.workingdir)
                if self.clear_subfolder:
                    # directory exists, make backup
                    backupdir = self.workingdir + '_bak'
                    if path.exists(backupdir):
                        # previous backup exists already, remove old
                        # backup, but keep .log and .out files (they have
                        # unique names):
                        keepers = (glob1(self.workingdir + '_bak', '*.log') +
                                glob1(self.workingdir + '_bak', '*.out'))
                        to_log.append(
                            ('removing existing backup {0}, but keeping {1}'
                            ' old log and output files').format(
                                backupdir, len(keepers)))
                        for f in keepers:
                            rename(path.join(backupdir, f),
                                path.join(self.workingdir, f))
                        rmtree(backupdir)
                        to_log.append(backupdir + ' removed')
                    # rename current (old) dir to backup:
                    rename(self.workingdir, backupdir)
                    to_log.append('existing ' + self.workingdir +
                                  ' renamed to ' + backupdir)
                    # make new empty working directory:
                    mkdir(self.workingdir)
                    to_log.append(
                        'created directory ' + self.workingdir + '\n')
                else:
                    to_log.append('working in existing directory.')
            else:
                # make new empty working directory:
                mkdir(self.workingdir)
                to_log.append('created directory ' + self.workingdir + '\n')

        if logger:
            if hasattr(logger, 'log') and callable(logger.log):
                # a custom logger was given as parameter, use it:
                log.logger = logger
            else:
                # Create the logger. Afterwards, we can also use
                # log.info() etc. in other modules. All status, logging
                # and stderr output will go through this logger (except
                # MEEP's output during simulation):
                log.setup_logger(
                    'root.' + self.jobname, self.log_file, self.quiet,
                    redirect_stderr=True)

        # now we can log the stuff from before:
        if to_log:
            log.info('\n' + '\n'.join(to_log))
        del to_log

        new_environ_dict = {
            'GUILE_WARN_DEPRECATED': 'no'}
        environ.update(new_environ_dict)
        log.info('added to environment:' + 
                 ''.join(['\n  {0}={1}'.format(key, environ[key]) for key in 
                         new_environ_dict.keys()]))

        log.info(
            'pymeep Simulation created with following properties:' + 
            ''.join(['\npymeepprop: {0}={1!r}'.format(key, val) for key, val in
                self.__dict__.items()]) + '\n\n')


    def __str__(self):
        temp_dict = self.__dict__.copy()
        return (self.ctl_template%temp_dict)

    def write_ctl_file(self, where='./'):
        filename = path.join(where, self.ctl_file)
        log.info("writing ctl file to %s" % filename)
        log.info("### ctl file for reference: ###\n" + 
            str(self) + '\n### end of ctl file ###\n\n')
        with open(filename,'w') as input_file:
            input_file.write(str(self))

    def run_simulation(self, num_processors=2):
        self.write_ctl_file(self.workingdir)

        meep_call_str = defaults.meep_call % dict(num_procs=num_processors)

        with open(self.out_file, 'w') as outputFile:
            log.info("Using MEEP " + defaults.meepversion)
            log.info("Running the MEEP-computation using the following "
                     "call:\n" +
                " ".join([meep_call_str, self.ctl_file]))
            log.info("Writing MEEP output to %s" % self.out_file)
            # write Time and ctl as reference:     
            outputFile.write("This is a simulation started by pyMEEP\n")
            outputFile.write("Run on: " + uname()[1] + "\n")
            starttime = datetime.now()
            outputFile.write("Date: " + str(starttime) + "\n")
            outputFile.write("\n=================================\n")
            outputFile.write("=========== CTL INPUT ===========\n")
            outputFile.write("=================================\n\n")
            outputFile.write(str(self))
            outputFile.write("\n\n==================================\n")
            outputFile.write("=========== MEEP OUTPUT ===========\n")
            outputFile.write("==================================\n\n")
            outputFile.flush()
            log.info('MEEP simulation is running... To see progress, please '
                'check the output file %s' % self.out_file)
            # run MEEP, write output to outputFile:
            # TODO can we also pipe MEEP output to stdout, so the user can
            # see progress?
            p = sp.Popen(meep_call_str.split() + [self.ctl_file], 
                               stdout=outputFile,
                               stderr=sp.STDOUT,
                               cwd=self.workingdir)
            retcode = p.wait()
            endtime = datetime.now()
            outputFile.write("finished on: %s (duration: %s)\n" % 
                             (str(endtime), str(endtime - starttime)))
            outputFile.write("returncode: " + str(retcode))
            log.info("Simulation finished, returncode: " + str(retcode))

        return retcode

    def get_latest_ouput_file_name(self):
        """Return latest output file name (latest .out file in working folder)"""
        canditates = glob1(self.workingdir, self.jobname + '*.out')
        if len(canditates) > 0:
            # make newest first:
            canditates.sort(reverse=True)
            self.out_file = path.join(self.workingdir, canditates[0])
            log.info('Post-processing output file from previous '
                     'simulation run: {0}'.format(self.out_file))
        else:
            # we try again using only the file suffix, maybe the folder or file was renamed:
            candidates = glob1(self.workingdir, '*.out')
            if len(candidates) > 0:
                # make newest first:
                candidates.sort(reverse=True)
                self.out_file = path.join(self.workingdir, candidates[0])
                log.info('Post-processing output file from previous '
                         'simulation run: {0}'.format(self.out_file))
                if len(candidates) > 1:
                    log.warn("Other output files were found in same folder: {0}".format(candidates))
            else:
                log.exception('Cannot post-process, no simulation output '
                              'file found!')
                return
        return self.out_file

    def plot_raw_fluxes(self, interactive=False):
        """Plot all recorded flux data

        :return: None

        """
        self.fluxdata = postprocess.load_flux_data(self.get_latest_ouput_file_name())

        #if 'fcen' in self.__dict__ and 'df' in self.__dict__:
        #    y_range = (self.fcen - self.df / 2.0, self.fcen + self.df / 2.0)
        #else:
        #    y_range = None
    
        if interactive:
            fig_filename = None
        else:
            fig_filename = path.join(self.workingdir, self.jobname + '_fluxes_raw.png')
            log.info('saving flux diagram to file %s' % fig_filename)

        fig = plt.figure(figsize=(14, 5))
        for i in range(self.fluxdata.shape[1] - 1):
            plt.plot(self.fluxdata[:, 0], self.fluxdata[:, i + 1], label='flux plane %i' % (i+1))

        plt.legend()
        #plt.xlim(0.43, 0.45)
        #plt.ylim(1.5, 2.5)

        plt.tight_layout()
        if fig_filename is not None:
            fig.savefig(
                fig_filename, transparent=False,
                bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

        return

    def plot_fluxes(self, interactive=False, only_fluxplanes=None, norm_with_influx=True, gap=None, xlim=None, ylim=None):

        if interactive:
            fig_filename = None
        else:
            fig_filename = path.join(self.workingdir, self.jobname + '_fluxes.png')
            log.info('saving flux diagram to file %s' % fig_filename)

        self.fluxdata = postprocess.load_flux_data(self.get_latest_ouput_file_name())

        fig = plt.figure(figsize=(13.8,6))
        X = self.fluxdata[:, 0]
        #X = range(len(fluxes[:, 0]))
        if only_fluxplanes is None:
            # plot all:
            only_fluxplanes = range(1, self.fluxdata.shape[1])

        refl = self.fluxdata[:, 1]
        trans = self.fluxdata[:, 2]
        influx = trans-refl
        #plt.plot(X, influx)
        for i in only_fluxplanes:
            if norm_with_influx:
                plt.plot(X, self.fluxdata[:, i]/influx, label=i)
            else:
                plt.plot(X, self.fluxdata[:, i], label=i)

        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)

        ymin, ymax = plt.ylim()
        if gap is not None:
            plt.fill([gap[0], gap[0], gap[1], gap[1]], [ymin, ymax, ymax, ymin], 'b', alpha=0.15)
        plt.ylim(ymin, ymax)

        plt.legend()
        plt.tight_layout()
        if fig_filename is not None:
            fig.savefig(
                fig_filename, transparent=False,
                bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

        return

    def post_process_bands(self, interactive=False, gap=None, maxq=None, coldataindex=1):
        """Make csv files for all band information.

        :return: None

        """

        # load all frequencies, even if two harminv outputs present:
        self.kdata, self.hdata = postprocess.load_bands_data(
            self.get_latest_ouput_file_name(), phase_tolerance=100, freq_tolerance=-1)  # 0.01)

        #        x_axis_formatter = axis_formatter.KVectorAxisFormatter(5)
        #        # add hover data:
        #        if x_axis_formatter._hover_func_is_default:
        #            x_axis_formatter.set_hover_data(self.kdata)
        #
        #        plotter = BandPlotter(figure_size=defaults.fig_size)
        #        print (self.kdata.shape, self.hdata.shape)
        #        plotter.plot_bands(
        #            self.hdata[:, :, 0], self.kdata,
        #            formatstr='o'
        #            x_axis_formatter=x_axis_formatter)
        #        plotter.show()


        if 'fcen' in self.__dict__ and 'df' in self.__dict__:
            y_range = (self.fcen - self.df / 2.0, self.fcen + self.df / 2.0)
        else:
            y_range = None

        if interactive:
            fig_filename = None
            self.patts_simulated = self.find_modes_with_calculated_patterns()
            try:
                self.patts_to_sim = np.loadtxt(path.join(self.workingdir, 'patterns_to_simulate'), ndmin=2)
            except IOError:
                self.patts_to_sim = None
        else:
            fig_filename = path.join(self.workingdir, self.jobname + '_bands.png')
            log.info('saving band diagram to file %s' % fig_filename)
            self.patts_simulated = None
            self.patts_to_sim = None
        postprocess.plot_bands(
            self.kdata, self.hdata,
            draw_light_line=True,
            maxq=maxq,
            filename=fig_filename,
            onpick=self.onclick,
            modes_with_calculated_patterns=self.patts_simulated,
            mode_patterns_to_be_calculated=self.patts_to_sim,
            y_range=y_range,
            gap=gap,
            coldataindex=coldataindex)

        return

    def find_modes_with_calculated_patterns(self):
        # make list of all field pattern folders:
        filenames = glob1(self.workingdir, "pattern_*")
        if not filenames:
            return np.zeros((0, 0))

        result = []

        # The regular expression pattern for parsing file names:
        retest = re.compile('^pattern_k(?P<knum>\d+)_f(?P<fraw>[p\d]+)$')

        # Analyze folders and make dictionary with data for each
        # destination file:
        dst_dict = dict()
        for fname in filenames:
            m = retest.match(fname)
            if m is None:
                # wrong format, probably another file/folder
                continue
            redict = m.groupdict()
            try:
                freq = float(redict['fraw'].replace('p', '.'))
                knum = float(redict['knum'])
            except (KeyError, ValueError):
                # wrong format, probably another file/folder
                continue
            result.append(np.array([knum, freq]))
        return np.array(result)

    def onclick(self, event):
        """This is the function called if the bands are plotted with a
        picker supplied and the user clicks on a vertex in the plot. It then
        prints some information about the vertex(ices) clicked on to stdout,
        including the mode, the k-vector and -index and the frequency(ies).

        The k-index, the frequency and a default bandwidth (should be adjusted
        manually later) is added to a file ('patterns_to_simulate') which can
        be used later to selectively excite a single mode and save the mode
        pattern in another simulation.
        On the other hand, if the mode pattern was already simulated in this
        way, there should exist a subfolder with the name
        'pattern_k{0:03.0f}_f{1:.4f}'.format(kindex, frequency).replace('.', 'p').
        Then a new figure is displayed with all pngs found in this subfolder.

        """
        try:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print('event.ind:', ind, thisline.indexes.shape)
            #xaxisformatter = event.mouseevent.inaxes.xaxis.major.formatter
        except AttributeError:
            print('error getting event data')
            return

        print()
        for i in ind:
            kindex = thisline.indexes[0, i]
            bandindex = thisline.indexes[1, i]
            kvec = thisline.kdata[kindex]
            freq = thisline.hdata[kindex, bandindex, 0]
            #xaxispos = xdata[i]
            #freq = ydata[i]
            s = 'picker_index={0}, band_index={1}, k_index={2:.0f}, k_vec={3}, freq={4}'.format(
                i, bandindex, kindex, kvec, freq)
            print(s + '; ')

        ## display mode pattern if it was exported;
        patterndir = path.join(
            self.workingdir,
            'pattern_k{0:03.0f}_f{1:.4f}'.format(kindex, freq).replace('.', 'p'))

        ## calculate it if not exported yet:        
        if not path.exists(patterndir):
            log.info('Folder "%s" not found; added to list of modes to be simulated.' % patterndir)
            with open(path.join(self.workingdir, 'patterns_to_simulate'), 'a') as f:
                f.write("{0:.0f}\t{1:.4f}\t{2:.4f}\n".format(kindex, freq, defaults.mode_pattern_sim_df))
            # mark the mode in the plot:
            s = plt.scatter([kindex], [freq], facecolors='none', edgecolors='r', s=100)
            plt.show()

        if path.exists(patterndir):
            # display all pngs in folder   
            pngs = glob1(patterndir, '*.png')
            cnt = len(pngs)
            if not cnt:
                return
            print('displaying all pngs in folder: %s' % patterndir)
            # print the frequencies found in the simulation to stdout, to make sure only one mode was excited:
            with open(path.join(patterndir, glob1(patterndir, '*.out')[0])) as f:
                for line in f:
                    if line.startswith('harminv'):
                        print(line.rstrip())

            # Start interactive mode:
            plt.ion()
            # create a new popup figure:
            fig, axes = plt.subplots(ncols=cnt, num='mode pattern', figsize=(min(16, cnt*2), 2), sharey=True) 
            plt.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0)
            
            maxx = 0
            maxy = 0
            for i, fname in enumerate(pngs):
                img = mpimg.imread(path.join(patterndir, fname))
                axes[i].imshow(
                    img,)
                    #origin='upper',
                    #extent=(xl, xr, yb, yt),
                    #interpolation='none')
                maxx = max(maxx, img.shape[1])
                maxy = max(maxy, img.shape[0])
                axes[i].set_title(fname, {'fontsize' : 6})
                # remove ticks:
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                
            print(maxx, maxy)
            for ax in axes:
                ax.set_xlim(0, maxx)
                ax.set_ylim(0, maxy)
            #plt.tight_layout()
