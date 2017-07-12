# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------
# Copyright 2017 Juergen Probst
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with This program.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------

from __future__ import division

import sys
from os import path

import numpy as np
#import matplotlib.pyplot as plt

from pymeep import Simulation, log, defaults

g_num_processors = 16
template = r'''
(set! filename-prefix #f)

(define a1 (vector3 (/ (sqrt 3) 2)  0.5  0))
(define a2 (vector3 (/ (sqrt 3) 2) -0.5  0))
(define-param xtiles %(nx)s)
(define-param sx (* (sqrt 3) xtiles))
(define-param sy 1)
(define-param sz %(sz)s) 
(define-param dpml %(dpml)s) ; pml thickness in z

(set! geometry-lattice (make lattice (size sx sy sz)))
(set! symmetries (list (make mirror-sym (direction Z) (phase 1))))

(define-param r %(radius)s)
(define-param slabt %(thickness)s); phc slab thickness
(define-param row1rad %(wg_row1rad)s) ; 1st neighboring row's hole radius
(define-param row2rad %(wg_row2rad)s) ; 2nd neighboring row's hole radius
(define-param nair 1.0)
(define-param nb 2.0085)

(define slabmat (make dielectric (epsilon 1)
	(E-susceptibilities 
            (make lorentzian-susceptibility
                (frequency 1.933128) (gamma 0) (sigma 2.8939))
        ))
)
;(define slabmat (make dielectric (index nb)))
(define airrod (make dielectric (index nair)))
(define c0 (vector3 0 0 0))
(define c1 (vector3  (/ (sqrt 3) -2)   0.5  0))
(define c2 (vector3  (/ (sqrt 3)  2)   0.5  0))
(define c3 (vector3  (/ (sqrt 3) -2)  -0.5  0))
(define c4 (vector3  (/ (sqrt 3)  2)  -0.5  0))

(set! geometry (append
    (list
        (make block (center 0 0 0) (size infinity infinity slabt) (material slabmat))
    )
    ; add 1st and 2nd neigboring row's holes:
    (list
        (make cylinder (center c1) (radius row1rad) (height infinity) (material airrod))
        (make cylinder (center c2) (radius row1rad) (height infinity) (material airrod))
        (make cylinder (center c3) (radius row1rad) (height infinity) (material airrod))
        (make cylinder (center c4) (radius row1rad) (height infinity) (material airrod))
        (make cylinder (center (vector3 (sqrt 3) 0 0)) (radius row2rad) (height infinity) (material airrod))
        (make cylinder (center (vector3 (- (sqrt 3)) 0 0)) (radius row2rad) (height infinity) (material airrod))
    )

    ; add holes to the right:
    (geometric-objects-duplicates (vector3 (sqrt 3) 0 0) 1 (/ xtiles 2)  
        (list
            (make cylinder (center c2) (radius r) (height infinity) (material airrod))
            (make cylinder (center c4) (radius r) (height infinity) (material airrod))
            (make cylinder (center (vector3 (sqrt 3) 0 0)) (radius r) (height infinity) (material airrod))
        )
    )
    ; add holes to the left:
    (geometric-objects-duplicates (vector3 (- (sqrt 3)) 0 0) 1 (/ xtiles 2)
        (list
            (make cylinder (center c1) (radius r) (height infinity) (material airrod))
            (make cylinder (center c3) (radius r) (height infinity) (material airrod))
            (make cylinder (center (vector3 (- (sqrt 3)) 0 0)) (radius r) (height infinity) (material airrod))
        )
    )
))


(set-param! resolution %(resolution)s)
(set! pml-layers (list (make pml (direction Z) (thickness dpml)))) 

(define-param k-interp %(k_interp)s)
(define Gamma (vector3 0 0 0))
;(define M     (vector3 (/ 0.5 (sqrt 3)) -0.5 0))
;(define K     (vector3 0.0 (/ -2 3) 0))
(define k-points (list Gamma (vector3 0 0.5 0)))
(set! k-points (interpolate k-interp k-points))

(define-param fcen %(fcen)s)
(define-param df %(df)s)
(define s1 (vector3 -0.666 -0.35 0))
(define s2 (vector3+ s1 a1))
(define gauss (make gaussian-src (frequency fcen) (fwidth df)))
(define-param T %(harminv_time_steps)s)
(set! harminv-Q-thresh 1.0)

(run-until 0 (at-beginning output-epsilon))

(define dtime 0)
(define m1 1)
(define m2 0)
(define R (vector3* (* 2 pi) (vector3+ (vector3* m1 a1) (vector3* m2 a2)) ))

(define n 0)
(define rfreqs (list ))
(define ifreqs (list ))
(map (lambda (kvec)
        (restart-fields)
        (change-k-point! kvec)

        (set! dtime (vector3-dot kvec R))
        (change-sources! (list
            (make source (src gauss) (component Hz) (center s1))
            (make source (src gauss) (component Hz) (center s2)
                         (amplitude (exp (* 0+1i dtime))))
        ))
        (run-sources+  T
            (after-sources (harminv Hz s1 fcen df)
                           (harminv Hz s2 fcen df))
        )

        (set! rfreqs (map  harminv-freq-re  harminv-results))
        (print "freqs:, " (+ n 1) ", " (vector3-x kvec) ", " (vector3-y kvec)
               ", " (vector3-z kvec))
        (map  (lambda (f) (print ", " f))  rfreqs)
        (print "\n")
        (set! ifreqs (map  harminv-freq-im  harminv-results))
        (print "ifreqs:, " (+ n 1) ", " (vector3-x kvec) ", " (vector3-y kvec)
               ", " (vector3-z kvec))
        (map  (lambda (f) (print ", " f))  ifreqs)
        (print "\n")

        (set! n (+ n 1))
        (meep-fields-print-times fields)
     )
     k-points
)
'''


def main():
    ownname = path.splitext(path.basename(sys.argv[0]))[0].capitalize()
    if len(sys.argv) > 1:
        runmode = sys.argv[1][0]
    else:
        print('please provide mode: "sim" / "s" or "display" / "d" or "post-process" / "p"')
        return

    if len(sys.argv) > 2:
        onlystep = float(sys.argv[2])
    else:
        onlystep = None

    radius = 0.38
    thickness = 0.8
    resolution = 12
    sz = 4
    dpml = 1
    containing_folder = './'
    T = 200
    fcen = 0.5
    df = 0.3
    nx = 7
    k_interp = 15

    minstep = 0.20
    stepsize = 0.02 
    maxstep = 0.48 #minstep + stepsize * 6
    numsteps = int((maxstep - minstep) / stepsize + 1.5)
    steps = np.linspace(minstep, maxstep, num=numsteps, endpoint=True)
    #steps = [0.8]

    if runmode == 'd' and numsteps > 1 and onlystep is None:
        print('will only plot first step: %f. Alternatively, specify step as 2nd parameter' % minstep)
        steps = [minstep]
    elif onlystep is not None:
        steps = [onlystep]

    numsteps = len(steps)

    # save previous step's data, needed for comparison with current data:
    #prev_step_data = None

    for i, step1 in enumerate(steps):
      for j, step2 in enumerate(steps[1::2]):
        log.info(
            "running simulation with "
            "{0:n} first row radius steps:\n{1}".format(
                numsteps, steps) +
            "{0:n} second row radius steps:\n{1}".format(
                numsteps, steps) +
            '\n  ### current step: row1: #{0} ({1}); row2: #{2} ({3}) ###\n'.format(
                i + 1, step1, j + 1, step2))

        r1 = step1
        r2 = step2

        ### create and run simulation ###

        jobname = ownname + '_r{0:03.0f}_t{1:03.0f}_res{2:03.0f}'.format(
                    radius * 1000, thickness * 100, resolution)
        jobname_suffix = '_1r{0:03.0f}_2r{1:03.0f}'.format(r1 * 1000, r2 * 1000)

        sim = Simulation(
            jobname=jobname + jobname_suffix,
            ctl_template=template,
            resolution=resolution,
            work_in_subfolder=path.join(
                containing_folder, jobname + jobname_suffix),
            clear_subfolder=runmode.startswith('s'),
            radius=radius,
            thickness=thickness,
            sz=sz,
            dpml=dpml,
            harminv_time_steps=T,
            fcen=fcen,
            df=df,
            nx=nx,
            k_interp=k_interp,
            wg_row1rad=r1,
            wg_row2rad=r2
        )

        if runmode == 's':
            error = sim.run_simulation(num_processors=g_num_processors)
            if error:
                log.error('an error occured during simulation. See the .out file')
                return
        if runmode in ['s', 'd', 'p']:
            sim.post_process_bands(interactive = runmode == 'd', gap=[0.44090, 0.52120])

        log.info(' ##### step={0}-{1} - success! #####\n\n'.format(step1, step2))

        # reset logger; the next stuff logged is going to next step's file:
        log.reset_logger()
        
if __name__ == '__main__':
    main()
