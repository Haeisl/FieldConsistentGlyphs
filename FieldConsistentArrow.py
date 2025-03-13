import os
import sys

sys.path.append(os.path.dirname(__file__))

import time
from functools import partial

import numpy as np
from paraview.util.vtkAlgorithm import smproperty, smproxy
from vtk import vtkCellArray, vtkPoints, vtkPolyLine
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.vtkCommonDataModel import (vtkDataObject, vtkImageData,
                                           vtkPolyData)

import Utilities
from Colors import (group_color_values, group_colors, group_map,
                    orientation_color_values, orientation_colors,
                    orientation_map)
from Metrics import print_metrics
from SharedData import shared_data


@smproxy.filter(name="consistentArrow", label="Field Consistent Arrow Glyph")
@smproperty.input(name="Input")
@smproperty.xml("""<OutputPort name="PolyOutput" index="0" id="port0"/>""")
class consistentArrow(VTKPythonAlgorithmBase):
    def __init__(self):
        # ---default properties---
        self._mode = "Runge-Kutta 4"
        self._shape = "Smooth"
        self._normalize = True
        self._center = [1., 1.]
        self._grid_dims = [0, 0]
        # ---advanced properties---
        self._stepsize = 0.5
        self._scaling = 1.
        self._length = 5.
        self._thickness = 1.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, nOutputPorts=1)


    @smproperty.xml("""
        <StringVectorProperty name="Integration Method"
                              command="SetMethod"
                              number_of_elements="1"
                              default_values="Runge-Kutta 4">
            <StringListDomain name="list">
                <String value="Runge-Kutta 4"/>
                <String value="Euler"/>
            </StringListDomain>
            <Documentation>
                Integration Method used in tracing streamlines.
            </Documentation>
        </StringVectorProperty>
    """)
    def SetMethod(self, value):
        self._mode = value
        self.Modified()


    @smproperty.xml("""
        <StringVectorProperty name="Arrow Tip Shape"
                              command="SetShape"
                              number_of_elements="1"
                              default_values="Smooth">
            <StringListDomain name="list">
                <String value="Smooth"/>
                <String value="Jagged"/>
                <String value="Bezier Spline"/>
            </StringListDomain>
            <Documentation>
                Type of arrow tip shape to be used.
            </Documentation>
        </StringVectorProperty>
    """)
    def SetShape(self, value):
        self._shape = value
        self.Modified()


    @smproperty.xml("""
        <IntVectorProperty name="Normalize"
                           command="SetNormalize"
                           number_of_elements="1"
                           default_values="1">
            <BooleanDomain name="bool" />
            <Documentation>
                Whether the steps taken during integration should be normalized. Recommended.
            </Documentation>
        </IntVectorProperty>
    """)
    def SetNormalize(self, val):
        self._normalize = val
        self.Modified()


    @smproperty.xml("""
        <IntVectorProperty name="Only Show Glyphs On Grid"
                           command="SetOnlyGrid"
                           number_of_elements="1"
                           default_values="1">
            <BooleanDomain name="bool" />
            <Documentation>
                Dismisses the Additional Glyph option if checked.
            </Documentation>
        </IntVectorProperty>
    """)
    def SetOnlyGrid(self, val):
        self._only_grid = val
        self.Modified()


    @smproperty.xml("""
        <DoubleVectorProperty name="Additional Glyph"
                              command="SetStartPoint"
                              number_of_elements="2"
                              default_values="0.0 0.0">
            <DoubleRangeDomain name="range"/>
            <Documentation>
                Generates a glyph evolving from the specified point.
            </Documentation>
        </DoubleVectorProperty>
    """)
    def SetStartPoint(self, x, y):
        self._center = [x, y]
        self.Modified()


    @smproperty.xml("""
        <IntVectorProperty name="Grid [rows] | [cols]"
                           command="SetGridDims"
                           number_of_elements="2"
                           default_values="4 4">
            <Documentation>
                Draws glyphs in a regular grid. Leaving zeroes yields no grid.
            </Documentation>
        </IntVectorProperty>
    """)
    def SetGridDims(self, rows, cols):
        self._grid_dims = [rows, cols]
        self.Modified()


    #---advanced---
    @smproperty.xml("""
        <DoubleVectorProperty name="Stepsize"
                              command="SetStepSize"
                              number_of_elements="1"
                              default_values="0.2"
                              panel_visibility="advanced">
            <Documentation>
                Controls the stepsize used during integration.
            </Documentation>
        </DoubleVectorProperty>
    """)
    def SetStepSize(self, d):
        self._stepsize = d
        self.Modified()


    @smproperty.xml("""
        <DoubleVectorProperty name="Glyph scaling"
                              command="SetScaling"
                              number_of_elements="1"
                              default_values="1."
                              panel_visibility="advanced">
            <DoubleRangeDomain name="range" min="0.0" max="2.0" />
            <Documentation>
                Additional size scaling for the glyphs. Effectively a factor to the stepsize.
            </Documentation>
        </DoubleVectorProperty>
    """)
    def SetScaling(self, s):
        self._scaling = s
        self.Modified()


    @smproperty.xml("""
        <DoubleVectorProperty name="Glyph Length"
                              command="SetLength"
                              number_of_elements="1"
                              default_values="3."
                              panel_visibility="advanced">
            <Documentation>
                Unit length of the glyph along its major streamlet.
            </Documentation>
        </DoubleVectorProperty>
    """)
    def SetLength(self, l):
        self._length = l
        self.Modified()


    @smproperty.xml("""
        <DoubleVectorProperty name="Glyph Width"
                              command="SetThickness"
                              number_of_elements="1"
                              default_values="1."
                              panel_visibility="advanced">
            <Documentation>
                Unit width of the glyph at its bottom arc.
            </Documentation>
        </DoubleVectorProperty>
    """)
    def SetThickness(self, d):
        self._thickness = d
        self.Modified()


    def FillOutputPortInformation(self, port, info):
        info.Set(vtkDataObject.DATA_TYPE_NAME(), 'vtkPolyData')
        return 1


    def get_step_function(self):
        step_funcs = {
            "Runge-Kutta 4": Utilities.rk4_step,
            "Euler": Utilities.euler_step,
        }
        if self._mode in step_funcs:
            return step_funcs[self._mode]
        else:
            raise ValueError(f"Unexpected value for {self._mode=}")


    def integrate_standard(self, cur, units, backward, single=False):
        steps = 1 if single else int(np.ceil(units / self._stepsize))

        points = []
        for _ in range(steps):
            next, mag = self._step_function(cur, backward)
            if mag < self._stepsize:
                break
            points.append(next)
            cur = next

        return points


    def integrate_orthogonal(self, cur, units, left, single=False):
        steps = 1 if single else int(np.ceil(units / self._stepsize))

        points = []
        for _ in range(steps):
            next, mag = self._step_function(cur, left, ortho=True)
            if mag < self._stepsize:
                break
            points.append(next)
            cur = next

        return points


    def jagged_tip(self, start, end, parallel_func, orthogonal_func):
        points = []

        cur = start
        delta_v = parallel_func(cur=cur, units=1)[0] - cur
        delta_v = Utilities.normalize_if_requested(delta_v)
        next = cur + delta_v * self._stepsize * self._scaling
        points.append(next)
        cur = next

        consecutive_parallel_steps = 1
        consecutive_orthogonal_steps = 0
        min_dist = Utilities.magnitude(end - cur)

        for _ in range(1000):
            step_parallel = parallel_func(cur=cur, units=1)[0] - cur
            step_orthogonal = orthogonal_func(cur=cur, units=1)[0] - cur
            point_parallel = cur + Utilities.normalize_if_requested(step_parallel) * self._stepsize * self._scaling
            point_orthogonal = cur + Utilities.normalize_if_requested(step_orthogonal) * self._stepsize * self._scaling

            dist_parallel = Utilities.magnitude(end - point_parallel)
            dist_orthogonal = Utilities.magnitude(end - point_orthogonal)

            if dist_parallel > min_dist and dist_orthogonal > min_dist:
                break

            if consecutive_parallel_steps >= 2:
                next = point_orthogonal
                min_dist = dist_orthogonal
                consecutive_orthogonal_steps += 1
                consecutive_parallel_steps = 0
            elif consecutive_orthogonal_steps >= 2:
                next = point_parallel
                min_dist = dist_parallel
                consecutive_parallel_steps += 1
                consecutive_orthogonal_steps = 0
            else:
                if dist_parallel <= dist_orthogonal:
                    next = point_parallel
                    min_dist = dist_parallel
                    consecutive_parallel_steps += 1
                    consecutive_orthogonal_steps = 0
                else:
                    next = point_orthogonal
                    min_dist = dist_orthogonal
                    consecutive_orthogonal_steps += 1
                    consecutive_parallel_steps = 0
            points.append(next)

            if np.allclose(next, end, atol=(self._stepsize * self._scaling) / 2):
                break

            cur = next

        return points


    def smooth_tip(self, start, end, parallel_func, orthogonal_func):
        # next = cur + a*vec_parallel + b*vec_orthogonal
        # a + b = 1
        # first iteration a = 0, b = 1
        # second iteration a = 1, b = 0
        # third to n-th iteration a_n = (a_{n-1} + a_{n-2}) / 2
        factor_a_0, factor_a_1 = 0., 1.
        dist_0, dist_1 = np.inf, np.inf
        for i in range(20):
            if i == 0:
                factor_a = factor_a_0
            elif i == 1:
                factor_a = factor_a_1
            else:
                factor_a = (factor_a_0 + factor_a_1) / 2.

            points = []
            factor_b = 1 - factor_a
            min_dist = np.inf
            cur = start

            for _ in range(1000):
                parallel_vec = parallel_func(cur=cur, units=1)[0] - cur
                orthogonal_vec = orthogonal_func(cur=cur, units=1)[0] - cur
                delta_v = factor_a * parallel_vec + factor_b * orthogonal_vec
                delta_v = Utilities.normalize_if_requested(delta_v)
                next = cur + delta_v * self._stepsize * self._scaling

                dist = Utilities.magnitude(end - next)

                if dist < self._stepsize:
                    points.append(next)
                    return points

                if dist < min_dist:
                    points.append(next)
                    min_dist = dist
                    cur = next
                else:
                    break

            if i == 0:
                dist_0 = min_dist
            elif i == 1:
                dist_1 = min_dist
            elif abs(dist_0 - min_dist) <= 1e-4 or abs(dist_1 - min_dist) <= 1e-4:
                return points
            else:
                distance_factors = [(dist_0, factor_a_0),(dist_1, factor_a_1),(min_dist, factor_a)]
                distance_factors.sort(key=lambda x: x[0])
                (dist_0, factor_a_0), (dist_1, factor_a_1) = distance_factors[:2]
        return points


    def bezier_tip(self, start, end, parallel_func, orthogonal_func):
        # orthogonal_func is neither needed nor used, but passed to ensure same function head as the other methods
        start_velocity = parallel_func(cur=start, units=1)[0]
        start_velocity_norm = Utilities.normalize_if_requested(start_velocity)
        start_control_point = start + start_velocity_norm

        end_velocity = orthogonal_func(cur=end, units=1)[0]
        end_velocity_norm = Utilities.normalize_if_requested(end_velocity)
        end_control_point = end - end_velocity_norm

        t_intermediary = np.linspace(0, 1, 1000)
        curve_points = Utilities.bezier_points(t_intermediary, start, start_control_point, end, end_control_point)

        arc_length = Utilities.calculate_arc_length(curve_points)
        num_points = int(arc_length // (self._stepsize * self._scaling)) + 1
        t_final = np.linspace(0, 1, num_points)

        return Utilities.bezier_points(t_final, start, start_control_point, end, end_control_point)


    def form_segment(self, *point_lists):
        segment_length = sum(len(points) for points in point_lists)
        segment_points = []
        for points in point_lists:
            segment_points.extend(points)
        return (segment_length, segment_points)


    def get_tip_shape_function(self):
        shape_funcs = {
            'Smooth': self.smooth_tip,
            'Jagged': self.jagged_tip,
            'Bezier Spline': self.bezier_tip,
        }
        if self._shape in shape_funcs:
            return shape_funcs[self._shape]
        else:
            raise ValueError(f"Unexpected value for {self._shape=}")


    def get_last_or_fallback(self, my_list, fallback_value):
        return my_list[-1] if my_list else fallback_value


    def compute_line_segments(self, origin):
        # Arrow tip shape
        arrowtip_shape = self.get_tip_shape_function()

        # Partial integration functions for the arrow tip
        arrowhead_standard = partial(self.integrate_standard, backward=False, single=True)
        arrowhead_orthogonal_l = partial(self.integrate_orthogonal, left=True, single=True)
        arrowhead_orthogonal_r = partial(self.integrate_orthogonal, left=False, single=True)

        # Compute center lines
        center_line_backwards = self.integrate_standard(cur=origin, backward=True, units=self._length/2)
        center_line_forwards = self.integrate_standard(cur=origin, backward=False, units=self._length/2)
        center_line_start = self.get_last_or_fallback(center_line_backwards, origin)
        center_line_end = self.get_last_or_fallback(center_line_forwards, origin)

        # Compute bottom arcs
        bottom_arc_left = self.integrate_orthogonal(cur=center_line_start, left=True, units=self._thickness/2)
        bottom_arc_right = self.integrate_orthogonal(cur=center_line_start, left=False, units=self._thickness/2)
        bottom_left_end = self.get_last_or_fallback(bottom_arc_left, center_line_start)
        bottom_right_end = self.get_last_or_fallback(bottom_arc_right, center_line_start)

        # Compute side lines
        side_line_length = (len(center_line_backwards) + len(center_line_forwards)) / 2 * self._stepsize
        side_line_left = self.integrate_standard(cur=bottom_left_end, backward=False, units=side_line_length)
        side_line_right = self.integrate_standard(cur=bottom_right_end, backward=False, units=side_line_length)
        side_left_end = self.get_last_or_fallback(side_line_left, bottom_left_end)
        side_right_end = self.get_last_or_fallback(side_line_right, bottom_right_end)

        # Compute arrow bases
        arrowbase_length = (len(bottom_arc_left) + len(bottom_arc_right)) / 2 * self._stepsize
        arrowbase_left = self.integrate_orthogonal(cur=side_left_end, left=True, units=arrowbase_length)
        arrowbase_right = self.integrate_orthogonal(cur=side_right_end, left=False, units=arrowbase_length)
        arrowbase_left_end = self.get_last_or_fallback(arrowbase_left, side_left_end)
        arrowbase_right_end = self.get_last_or_fallback(arrowbase_right, side_right_end)

        # Compute arrow heads
        arrowhead_left = arrowtip_shape(
            start=arrowbase_left_end,
            end=center_line_end,
            parallel_func=arrowhead_standard,
            orthogonal_func=arrowhead_orthogonal_r
        )

        arrowhead_right = arrowtip_shape(
            start=arrowbase_right_end,
            end=center_line_end,
            parallel_func=arrowhead_standard,
            orthogonal_func=arrowhead_orthogonal_l
        )

        segments = [
            self.form_segment(center_line_backwards[::-1], [origin], center_line_forwards),
            self.form_segment(bottom_arc_left[::-1], [center_line_start], bottom_arc_right),
            self.form_segment([bottom_left_end], side_line_left),
            self.form_segment([bottom_right_end], side_line_right),
            self.form_segment([side_left_end], arrowbase_left),
            self.form_segment([side_right_end], arrowbase_right),
            self.form_segment([arrowbase_left_end], arrowhead_left),
            self.form_segment([arrowbase_right_end], arrowhead_right),
        ]

        return segments


    def construct_glyph_segment(self, segment, points: vtkPoints, lines:vtkCellArray):
        start_index = points.GetNumberOfPoints()

        length = segment[0]
        segment_points = segment[1]
        for p in segment_points:
            points.InsertNextPoint(p)

        polyline = vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(length)
        for i in range(length):
            polyline.GetPointIds().SetId(i, start_index + i)

        lines.InsertNextCell(polyline)
        start_index += length


    def RequestData(self, request, inInfo, outInfo):
        start_time = time.time()

        # get first input and the output
        image_data = vtkImageData.GetData(inInfo[0])
        bounds = image_data.GetBounds()
        shared_data.set("image_data", image_data)
        shared_data.set("bounds", bounds)
        shared_data.set("stepsize", self._stepsize)
        shared_data.set("scaling", self._scaling)
        shared_data.set("normalize", self._normalize)

        self._step_function = self.get_step_function()

        poly_output = vtkPolyData.GetData(outInfo)
        poly_data = vtkPolyData()

        # points object to hold all the line points
        points = vtkPoints()
        # cell array to store the lines' connectivity
        lines = vtkCellArray()

        origins = np.array([]).reshape((0,3))
        if not self._only_grid:
            origins = np.array([self._center[0], self._center[1], 0.0]).reshape((1,3))

        grid_points = Utilities.generate_grid_points(bounds, self._grid_dims)
        origins = np.concatenate((origins, grid_points), axis=0)

        glyph_creation_times = []

        glyph_start_time = time.time()
        for glyph_number, origin in enumerate(origins):
            try:
                segments = self.compute_line_segments(origin=origin)
            except Exception as e:
                glyph_creation_times.append((glyph_number, origin, "CRITICAL POINT  "))
                continue
            for index, segment in enumerate(segments):
                self.construct_glyph_segment(segment, points, lines)

                segment_orientation_color = orientation_color_values[orientation_map[index]]
                orientation_colors.InsertNextTypedTuple(segment_orientation_color)

                segment_group_color = group_color_values[group_map[index]]
                group_colors.InsertNextTypedTuple(segment_group_color)

            glyph_creation_times.append((glyph_number, origin, time.time()))


        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        poly_data.GetCellData().AddArray(group_colors)
        poly_data.GetCellData().AddArray(orientation_colors)

        poly_output.ShallowCopy(poly_data)

        end_time = time.time()

        print_metrics(
            start_time,
            end_time,
            glyph_start_time,
            glyph_creation_times,
            points,
            lines,
        )

        return 1
