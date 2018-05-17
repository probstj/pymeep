# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------
# Copyright 2017 Juergen Probst
#
# This file is part of pyMEEP.
#
# pyMEEP is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyMEEP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyMEEP.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------

from __future__ import division, print_function
import gdspy
import numpy as np
from pymeep import log


def _get_polygon_bounding_box(polygon):
    if len(polygon) == 0:
        bb = np.array(((0, 0), (0, 0)))
    else:
        bb = np.array(
            ((polygon[:, 0].min(), polygon[:, 1].min()),
             (polygon[:, 0].max(), polygon[:, 1].max())))
    return bb


class GDSReader(object):
    def __init__(self, filename, unit=None):
        """
        filename : file or string
            GDSII stream file (or path) to be imported.  It must be opened for
            reading in binary format.
        unit : number
            Unit (in *meters*) to use for the imported structures.  If
            ``None``, the units used to create the GDSII file will be used.

        """
        self.filename = filename
        self.gdslib = gdspy.GdsLibrary()
        self.gdslib.read_gds(filename, unit)
        self.celldict = self.gdslib.cell_dict

    def get_cell_names(self):
        return self.celldict.keys()

    def get_element_locations_and_radii(self, cellname, layers=None, center_structure=False, shift_granularity=0):
        """Return a list of (x pos, y pos, radius)-tuples for each element in cell.

        layers is a list of layers to be exported. If None, all layers will be exported.

        If *center_structure*, the structure will be shifted such that the center is at zero.
        This shift will be a multiple of *shift_granularity*, if it is specified.
        If *shift_granularity* equals 1/resolution, the origin of the gds
        stays on a grid pixel boundary.

        """
        if cellname not in self.get_cell_names():
            log.exception('File does not contain cell "%s".' % cellname)
        cell = self.celldict[cellname]
        cbb = cell.get_bounding_box()
        if center_structure:
            # calc shift to center the structure:
            dx = - (cbb[0, 0] + cbb[1, 0]) / 2.0
            dy = - (cbb[0, 1] + cbb[1, 1]) / 2.0
            if shift_granularity:
                dx = np.round(dx / shift_granularity) * shift_granularity
                dy = np.round(dy / shift_granularity) * shift_granularity
        else:
            dx = 0
            dy = 0
        if layers is None:
            polys = cell.get_polygons(by_spec=False)
        else:
            byspec = cell.get_polygons(by_spec=True)
            polys = []
            for key, val in byspec.items():
                if key[0] in layers:
                    polys.extend(val)
        count = len(polys)
        result = np.zeros((count, 3))
        for i, polygon in enumerate(polys):
            bb = _get_polygon_bounding_box(polygon)
            x = (bb[0, 0] + bb[1, 0]) / 2.0
            y = (bb[0, 1] + bb[1, 1]) / 2.0
            # just a very cheep way of getting the radius, assuming all elements are circles:
            r = max(abs(bb[1, 0] - bb[0, 0]), abs(bb[1, 1] - bb[0, 1])) / 2.0
            result[i, :2] = x + dx, y + dy
            result[i, 2] = r
        return result
