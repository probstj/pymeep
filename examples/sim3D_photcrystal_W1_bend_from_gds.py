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
(define-param pointsfilename "%(pointsfilename)s")
(define-param swg-y-pos %(swg_y_pos)s)
(define-param dpml %(dpml)s) ; pml thickness in -x and +-z
(define-param dpmlthick %(dpmlthick)s) ; pml thickness in +x and +-y (where the holes are)
(define-param swgwidth %(swgwidth)s) ; width of solid waveguide
(define-param swglength %(swglength)s) ; length of solid waveguide
(define-param holerad %(holerad)s) ; radius of holes
(define-param phcterm %(phcterm)s) ; termination of phc at incoupling
(define-param slabt %(thickness)s); phc slab thickness

(define-param fcen %(fcen)s)
(define-param df %(df)s)
(define-param src-x (+ (/ sx -2) dpml 1)) ; source position along x
(define-param nfreq 501) ; number of frequencies at which to compute flux
(define-param rflux-x (+ (/ sx -2) dpml))
(define-param tflux-x (+ src-x 1))
; needed for correct placement of flux planes. Depends on gds:
(define-param input-arm-length 10.5)

(set-param! resolution %(resolution)s)

(set! geometry-lattice (make lattice (size sx sy sz)))
(set! symmetries (list (make mirror-sym (direction Z) (phase 1))))

(define-param nb 2.0085)

(define slabmat (make dielectric (index nb)))

(define x0 (- swglength (/ sx 2) phcterm)) ; x-position of first hole
(define y0 swg-y-pos)
; exit-x-pos must have an integer distance from x0, to ensure that it lies on a grid position:
(define exit-x-pos (+ x0 (floor (- (/ sx 2) dpmlthick x0))) )
; TODO: center-x-pos broken if a phcterm is defined!!!
(define center-x-pos (+ x0 (/ (- sx swglength) 2) ) )
;(define num-holes-x (ceiling (+ (- sx swglength) phcterm)))
;(define num-holes-y (ceiling (/ sy 2 (sqrt 3))) )
(print "x-position of interface: " (- swglength (/ sx 2)) "\n")
(print "y-position of interface: " swg-y-pos "\n")
(print "x-position of first hole: " x0 "\n")
(print "x-position of center of phc: " center-x-pos "\n")
(print "lattice-length of input arm: " input-arm-length "\n")
(print "x-position of exit: " exit-x-pos "\n")
;(print "number of holes in x: " num-holes-x "\n")
;(print "number of hole rows above W1: " (/ sy (sqrt 3)) "\n")

; ****************************************************************
; some utilities

; print a line of comma-separated values:
(define (print-csv-line . ds) (print (string-join (map (lambda (s) (if (number? s) (number->string s) s)) ds) ", ") "\n"))

; Read list from file:
(define (read-data-from-file filename)
    (call-with-input-file filename
      (lambda (p)
        (let f ((x (read p)))
          (if (eof-object? x)
              '()
              (cons x (f (read p))))))))

; ****************************************************************
; prepare holes from data file

; Read coords from file:
; Each line must be a tuple "(x y r)" with x and y coordinates and r radius of hole.
(define data (read-data-from-file pointsfilename))

(define holes
    (map (lambda (xyr) ; xyr is a tuple '(x y r)
        (let ((x (car xyr)) (y (cadr xyr)) (r (caddr xyr)))
            (make cylinder
                (center (+ x0 x) (+ swg-y-pos y) 0)
                (radius holerad) ; Note: override radius from file
                (height infinity)
                (material air) )
        )
    ) data) 
)

(define swg-exclude-block-top-width (- (/ (- sy swgwidth) 2) swg-y-pos) )
(define swg-exclude-block-bottom-width ( + (/ (- sy swgwidth) 2) swg-y-pos) )
(define swg-exclude-block-center-x (/ (- swglength sx) 2))
(define swg-exclude-block-top-center-y (/ (- sy swg-exclude-block-top-width) 2))
(define swg-exclude-block-bottom-center-y (/ (- swg-exclude-block-bottom-width sy) 2))

(set! geometry (append
    (list
        (make block (center 0 0 0) (size infinity infinity slabt) (material slabmat)) 
        (make block 
            (center swg-exclude-block-center-x swg-exclude-block-top-center-y 0) 
            (size swglength swg-exclude-block-top-width infinity) (material air))
        (make block 
            (center swg-exclude-block-center-x swg-exclude-block-bottom-center-y 0) 
            (size swglength swg-exclude-block-bottom-width infinity) (material air))
    )
    holes
))

; ****************************************************************

(set! pml-layers (list 
    (make pml (direction Z) (thickness dpml))
    (make pml (direction X) (side Low) (thickness dpml))
    (make pml (direction X) (side High) (thickness dpmlthick))
    (make pml (direction Y) (thickness (* dpmlthick (sqrt 3) 0.5)) )
))

(define gauss (make gaussian-src (frequency fcen) (fwidth df)))
(print "source:, x-position, (y-pos - swg-y-pos)/sqrt(3), height/sqrt(3) (in y), depth (in z)\n")
(print-csv-line "source:" src-x (/ (- y0 swg-y-pos) (sqrt 3)) (/ (* swgwidth 3) (sqrt 3)) (* slabt 3))
(set! sources (list
    (make eigenmode-source
        (src gauss)
        (eig-parity TE)
        (center src-x y0 0)
        (size 0 (* 3 swgwidth) (* 3 slabt))
        (component ALL-COMPONENTS)
    )
))

; ****************************************************************
; define flux planes

(define-param num-flux-planes-per-arm 13)
(define-param num-flux-planes-center-arm 8)
(define fpwidth swgwidth)

(define refl ; reflected flux                                                
    (add-flux fcen df nfreq
        (make flux-region (center rflux-x y0 0) (size 0 fpwidth slabt) )
))
(print "fluxplane:, 0, x-position, (y-pos - swg-y-pos)/sqrt(3), height/sqrt(3) (in y), depth (in z)\n")   
(print-csv-line "fluxplane:" 1 rflux-x (/ (- y0 swg-y-pos) (sqrt 3)) (/ fpwidth (sqrt 3)) slabt)
(define trans ; flux just behind src                                                 
    (add-flux fcen df nfreq
        (make flux-region (center tflux-x y0 0) (size 0 fpwidth slabt) )
))
(print-csv-line "fluxplane:" 2 tflux-x (/ (- y0 swg-y-pos) (sqrt 3)) (/ fpwidth (sqrt 3)) slabt)

; add couple of transmission flux planes inside phc
(define fpwidth (* (sqrt 3) 4) )
(define xpos 0)
(define ypos 0)
(define n 3)
(define fluxplanes (append
    (map (lambda (x) (begin
        (set! xpos (+ x0 x) )
        (set! ypos y0)
        (print-csv-line "fluxplane:" n xpos (/ (- ypos swg-y-pos) (sqrt 3)) (/ fpwidth (sqrt 3)) slabt)
        (set! n (+ n 1))
        (add-flux fcen df nfreq
            (make flux-region (center xpos ypos 0) (size 0 fpwidth slabt) ) 
        ) ; add-flux
    )) ; begin ; lambda
    (arith-sequence 0 1 num-flux-planes-per-arm)) ; map
    
    (map (lambda (x) (begin
        ;(set! xpos (+ (- center-x-pos (/ num-flux-planes-center-arm 4)) 0.25 (/ x 2)) )
        ;(set! ypos (* (+ (- (/ num-flux-planes-center-arm 2)) 0.5 x) (sqrt 3) 0.5) )
	(set! xpos (+ x0 input-arm-length 0.5 (/ x 2)) )
	(set! ypos (+ y0 (* (+ x 1) (sqrt 3) 0.5)) )
        (print-csv-line "fluxplane:" n xpos (/ (- ypos swg-y-pos) (sqrt 3)) 0 slabt "(x-width: 2)")
        (set! n (+ n 1))
        (add-flux fcen df nfreq
            (make flux-region (center xpos ypos 0) (size 2 0 slabt) ) 
        ) ; add-flux
    )) ; begin ; lambda
    (arith-sequence 0 1 num-flux-planes-center-arm)) ; map
    
    (map (lambda (x) (begin
        (set! xpos (+ exit-x-pos dpmlthick (- num-flux-planes-per-arm) x) )
        (set! ypos (- y0) )
        (print-csv-line "fluxplane:" n xpos (/ (- ypos swg-y-pos) (sqrt 3)) (/ fpwidth (sqrt 3)) slabt )
        (set! n (+ n 1))
        (add-flux fcen df nfreq
            (make flux-region (center xpos ypos 0) (size 0 fpwidth slabt) ) 
        ) ; add-flux
    )) ; begin ; lambda
    (arith-sequence 0 1 num-flux-planes-per-arm)) ; map
    
    ) ; append
); define fluxplanes

; finally add some flux planes lining the pml:
(define zpos 0)
(define sizex (- sx dpml dpmlthick) )
(define sizey (- sy (* dpmlthick (sqrt 3))) )
(define sizez (- sz (* dpml 2)) )
(print "fluxplane:, pml-lining, x-pos, y-pos/sqrt(3), z-pos, x-size, y-size/sqrt(3), z-size\n")
(define fluxplanes (append
    fluxplanes
    (list 
        ; +Z:
        (begin
            (set! xpos (/ (- dpml dpmlthick) 2) )
            (set! ypos 0 )
            (set! zpos (- (/ sz 2) dpml) )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos sizex (/ sizey (sqrt 3)) 0)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex sizey 0) ) 
            ) 
        )
        ; -Z:
        (begin
            (set! zpos (- dpml (/ sz 2)) )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos sizex (/ sizey (sqrt 3)) 0)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex sizey 0) ) 
            )
        )
        ; +X:
        (begin
            (set! xpos (- (/ sx 2) dpmlthick) )
            (set! ypos 0 )
            (set! zpos 0 )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos 0 (/ sizey (sqrt 3)) sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size 0 sizey sizez) ) 
            ) 
        )
        ; -X:
        (begin
            (set! xpos (- dpml (/ sx 2)) )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos 0 (/ sizey (sqrt 3)) sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size 0 sizey sizez) ) 
            ) 
        )
        ; +Y:
        (begin
            (set! xpos (/ (- dpml dpmlthick) 2) )
            (set! ypos (- (/ sy 2) (* dpmlthick (sqrt 3) 0.5)) )
            (set! zpos 0 )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos sizex 0 sizez)
            (set! n (+ n 1))
            (add-flux fcen df nfreq
                (make flux-region (center xpos ypos zpos) (size sizex 0 sizez) ) 
            ) 
        )
        ; -Y:
        (begin
            (set! ypos (- (* dpmlthick (sqrt 3) 0.5) (/ sy 2)) )
            (print-csv-line "fluxplane:" n xpos (/ ypos (sqrt 3)) zpos sizex 0 sizez)
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

(run-sources+ (stop-when-fields-decayed 50 Hz (vector3 exit-x-pos (- y0) 0) 1e-6)
;(run-until 0
    (at-beginning (in-volume (volume (center 0 0 0) (size sx sy 0)) output-epsilon))
    (to-appended "hz" (at-every 0.6 (in-volume (volume (center 0 0 0) (size sx sy 0)) output-hfield-z ) ))
)

; ****************************************************************
; print flux data

(define fluxes (append (list refl trans) fluxplanes))
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
        runmode = sys.argv[1][0]
    else:
        print('please provide mode: "sim" / "s" or "display" / "d" or "post-process" / "p"')
        return

    if len(sys.argv) > 2:
        onlystep = float(sys.argv[2])
    else:
        onlystep = None

    gdsfile = './design.gds'
    gdscell = 'bend_optS'
    gdsunitlength = 0.27e-6
    resolution = 12
    fcen = 0.5
    df = 0.5
    dpml = 3
    dpml_thick = 6 # thicker pml where phc protrudes into pml
    swgwidth = '(sqrt 3)'
    swglength = 4 + dpml
    holerad = 0.32
    phcterm = 0
    sx = np.ceil((swglength + dpml_thick + 9) / 2.0) * 2
    sy = np.ceil((5 + dpml_thick) * np.sqrt(3) / 2.0) * 2
    sz = 4 + 2 * dpml
    thickness = 0.8
    containing_folder = './'

    minstep = 0
    stepsize = 0.1 
    maxstep =0 #minstep + stepsize * 6
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
        log.info("running simulation with {0:n} steps:\n{1}".format(
            numsteps, steps) +
            '\n  ### step: #{0} ({1}) ###\n'.format(i+1, step))


        ### create and run simulation ###

        gdsfilebase = path.splitext(path.basename(gdsfile))[0]
        pointsfilename = gdsfilebase + '_' + gdscell + '.dat'
        jobname = ownname + '_r{0:03.0f}_res{1:03.0f}'.format(holerad * 1000, resolution)
        jobname_suffix = '_opt0' #_term{0:03.0f}'.format(step*1000)

        sim = Simulation(
            jobname=jobname + jobname_suffix,
            ctl_template=template,
            resolution=resolution,
            work_in_subfolder=path.join(
                containing_folder, jobname + jobname_suffix),
            clear_subfolder=runmode.startswith('s'),
            sx=None, # will be set later
            sy=None, # will be set later
            sz=sz,
            thickness=thickness,
            pointsfilename=pointsfilename,
            dpml=dpml,
            dpmlthick=dpml_thick,
            swgwidth=swgwidth,
            swglength=swglength,
            swg_y_pos=None,  # will be set later
            holerad=holerad,
            phcterm=phcterm,
            fcen=fcen,
            df=df,
        )

        # prepare circles list:
        bb = export_gds_elements_locations_and_radii(
            gdsfile, gdscell,
            path.join(sim.workingdir, pointsfilename),
            unit=gdsunitlength,
            layers=[1])

        # late setting of important grid size:
        sim.sx = np.ceil((swglength + bb[1, 0] - bb[0, 0]) * resolution / 2) / resolution * 2.0
        sim.sy = np.ceil((bb[1, 1] - bb[0, 1]) * resolution / 2) / resolution * 2.0
        sim.swg_y_pos = (bb[0, 1] + bb[1, 1]) / -2.0

        if runmode == 's':
            error = sim.run_simulation(num_processors=g_num_processors)
            if error:
                log.error('an error occured during simulation. See the .out file')
                return
        if runmode in ['s', 'd', 'p']:
            sim.plot_fluxes(
                interactive = runmode == 'd', only_fluxplanes=None, gap=[0.411115, 0.473015],
                )#xlim=(0.4, 0.5), ylim=(0.3, 1))

        log.info(' ##### step={0} - success! #####\n\n'.format(step))

        # reset logger; the next stuff logged is going to next step's file:
        log.reset_logger()
        
if __name__ == '__main__':
    main()
