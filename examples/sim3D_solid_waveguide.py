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
from pymeep.utility import export_gds_elements_locations_and_radii

g_num_processors = 16
template = r'''
(set! filename-prefix #f)

(define-param sx %(sx)s)
(define-param sy %(sy)s)
(define-param sz %(sz)s) 
(define-param dpml %(dpml)s) ; pml thickness in x, y and z
(define-param swgwidth (sqrt 3))
(define-param slabt %(thickness)s); phc slab thickness

(define-param fcen %(fcen)s)
(define-param df %(df)s)
(define-param src-x (+ (/ sx -2) dpml 1)) ; source position along x
(define-param nfreq 501) ; number of frequencies at which to compute flux
;(define-param tflux-x (+ src-x 2))
(define-param rflux-x (+ (/ sx -2) dpml))

(set-param! resolution %(resolution)s)

(set! geometry-lattice (make lattice (size sx sy sz)))
(set! symmetries (list (make mirror-sym (direction Z) (phase 1))))

(define-param nair 1.0)
(define-param nb 2.0085)
(define slabmat (make dielectric (index nb)))
(define airrod (make dielectric (index nair)))

; ****************************************************************
; some utilities

; print a line of comma-separated values:
(define (print-csv-line . ds) (print (string-join (map (lambda (s) (if (number? s) (number->string s) s)) ds) ", ") "\n"))

; ****************************************************************

(define c0 (vector3 0 0 0))
;(define c1 (vector3  (/ (sqrt 3) -2)   0.5  0))
;(define c2 (vector3  (/ (sqrt 3)  2)   0.5  0))
;(define c3 (vector3  (/ (sqrt 3) -2)  -0.5  0))
;(define c4 (vector3  (/ (sqrt 3)  2)  -0.5  0))

(set! geometry (list
    (make block (center 0 0 0) (size infinity swgwidth slabt) (material slabmat)) 
))

(set! pml-layers (list (make pml (thickness dpml)))) 

(define gauss (make gaussian-src (frequency fcen) (fwidth df)))
(print "source:, x-position, y-position, width/sqrt(3) (in y), depth (in z)\n")
(print-csv-line "source:" src-x 0 (/ (* swgwidth 3) (sqrt 3)) (* slabt 3))
(set! sources (list
    (make eigenmode-source
        (src gauss)
        (eig-parity TE)
        (center src-x 0 0)
        (size 0 (* 3 swgwidth) (* 3 slabt))
        (component ALL-COMPONENTS)
    )
))

; ****************************************************************
; define flux planes

(define fpwidth swgwidth)

(print "fluxplane:, 0, x-position, y-position, width/sqrt(3) (in y), depth (in z)\n")   

; add couple of transmission flux planes 
(define n 1)
(define xpos 0)
(define x0 rflux-x)
(define fluxplanes 
    (map (lambda (x) (begin
        (set! xpos (+ x0 x) )
        (print-csv-line "fluxplane:" n xpos 0 (/ fpwidth (sqrt 3)) slabt)
        (set! n (+ n 1))
        (add-flux fcen df nfreq
            (make flux-region (center xpos 0 0) (size 0 fpwidth (* slabt 1)) ) 
        )
    ))
    (arith-sequence 0 1 (- (/ sx 2) x0 1))

    ) ; append
); define fluxplanes

; finally add some flux planes lining the pml:
(define ypos 0)
(define zpos 0)
(define sizex (- sx (* dpml 2)) )
(define sizey (- sy (* dpml 2)) )
(define sizez (- sz (* dpml 2)) )
(print "fluxplane:, pml-lining, x-pos, y-pos, z-pos, x-size, y-size, z-size\n")
(define fluxplanes (append
    fluxplanes
    (list 
        ; +Z:
        (begin
            (set! xpos 0 )
            (set! ypos 0 )
            (set! zpos (- (/ sz 2) dpml) )
            (print-csv-line "fluxplane:" n xpos ypos zpos sizex sizey 0)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex sizey 0) ) 
            ) 
        )
        ; -Z:
        (begin
            (set! zpos (- dpml (/ sz 2)) )
            (print-csv-line "fluxplane:" n xpos ypos zpos sizex sizey 0)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex sizey 0) ) 
            ) 
        )
        ; +X:
        (begin
            (set! xpos (- (/ sx 2) dpml) )
            (set! ypos 0 )
            (set! zpos 0 )
            (print-csv-line "fluxplane:" n xpos ypos zpos 0 sizey sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size 0 sizey sizez) ) 
            ) 
        )
        ; -X:
        (begin
            (set! xpos (- dpml (/ sx 2)) )
            (print-csv-line "fluxplane:" n xpos ypos zpos 0 sizey sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size 0 sizey sizez) ) 
            ) 
        )
        ; +Y:
        (begin
            (set! xpos 0 )
            (set! ypos (- (/ sy 2) dpml) )
            (set! zpos 0 )
            (print-csv-line "fluxplane:" n xpos ypos zpos sizex 0 sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex 0 sizez) ) 
            ) 
        )
        ; -Y:
        (begin
            (set! ypos (- dpml (/ sy 2)) )
            (print-csv-line "fluxplane:" n xpos ypos zpos sizex 0 sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex 0 sizez) ) 
            ) 
        )
    )
    
    ) ; append
); define fluxplanes

; ****************************************************************
; run             

(run-sources+ (stop-when-fields-decayed 50 Hz (vector3 (- (/ sx 2) dpml) 0 0) 1e-6)
;(run-until 30
    (at-beginning (in-volume (volume (center 0 0 0) (size sx sy 0)) output-epsilon))
;    (to-appended "hz" (at-every 0.6 (in-volume (volume (center 0 0 0) (size sx sy 0)) output-hfield-z) ))
)

; ****************************************************************
; print flux data

;(define fluxes (append (list refl trans) fluxplanes))
(define fluxes  fluxplanes)
(apply display-csv (append 
    (list "flux"
        (get-flux-freqs (car fluxes))
    )
    (map get-fluxes fluxes)
))

'''


def main():
    ownname = path.splitext(path.basename(sys.argv[0]))[0].capitalize()
    if len(sys.argv) > 1:
        runmode=sys.argv[1][0]
    else:
        print('please provide mode: "sim" / "s" or "display" / "d" or "post-process" / "p"')
        return

    if len(sys.argv) > 2:
        onlystep = float(sys.argv[2])
    else:
        onlystep = None

    resolution = 12
    fcen = 0.5
    df = 0.5
#    dpml = 3
#    sx = 22
#    sy = 12
#    sz = 10
    thickness = 0.8
    containing_folder = './'

    minstep = 1
    stepsize = 1 
    maxstep = 8.0 #minstep + stepsize * 6
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
    prev_step_data = None

    for i, step in enumerate(steps):
        dpml = step
        sx = 16 + 2 * dpml
        sy = 6 + 2 * dpml
        sz = 4 + 2 * dpml
        log.info("running simulation with {0:n} steps:\n{1}".format(
            numsteps, steps) +
            '\n  ### step: #{0} ({1}) ###\n'.format(i+1, step))


        ### create and run simulation ###

        jobname = ownname + '_res{0:03.0f}'.format(resolution)
        jobname_suffix = '_dpml{0:03.0f}'.format(dpml * 10)

        sim = Simulation(
            jobname=jobname + jobname_suffix,
            ctl_template=template,
            resolution=resolution,
            work_in_subfolder=path.join(
                containing_folder, jobname + jobname_suffix),
            clear_subfolder=runmode.startswith('s'),
            sx=sx,
            sy=sy,
            sz=sz,
            thickness=thickness,
            dpml=dpml,
            fcen=fcen,
            df=df,
        )

        if runmode == 's':
            error = sim.run_simulation(num_processors=g_num_processors)
            if error:
                log.error('an error occured during simulation. See the .out file')
                return
        if runmode in ['s', 'd', 'p']:
            sim.plot_fluxes(interactive = runmode == 'd', only_fluxplanes=None)

        log.info(' ##### step={0} - success! #####\n\n'.format(step))

        # reset logger; the next stuff logged is going to next step's file:
        log.reset_logger()
        
if __name__ == '__main__':
    main()
