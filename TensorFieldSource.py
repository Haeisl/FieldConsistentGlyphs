from paraview.util.vtkAlgorithm import *
import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.numpy_interface import dataset_adapter as dsa

@smproxy.source(name="TensorFieldSource", label="Tensor Field Source")
@smhint.xml('<InputProperty name="TimeSteps" number_of_elements="1" default_values="0"/>')
class TensorFieldSource(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(
            self, nInputPorts=0, nOutputPorts=1, outputType="vtkImageData"
        )
        self._dims = [10, 10, 10]
        self._origin = [0.0, 0.0, 0.0]
        self._spacing = [1.0, 1.0, 1.0]

    @smproperty.intvector(name="Dimensions", default_values=[10, 10, 10])
    @smdomain.intrange(min=1, max=100)
    def SetDimensions(self, x, y, z):
        self._dims = [x, y, z]
        self.Modified()

    @smproperty.doublevector(name="Origin", default_values=[0.0, 0.0, 0.0])
    def SetOrigin(self, x, y, z):
        self._origin = [x, y, z]
        self.Modified()

    @smproperty.doublevector(name="Spacing", default_values=[1.0, 1.0, 1.0])
    def SetSpacing(self, x, y, z):
        self._spacing = [x, y, z]
        self.Modified()

    def RequestData(self, request, inInfo, outInfo):
        output = vtkImageData.GetData(outInfo)
        output.SetDimensions(self._dims)
        output.SetOrigin(self._origin)
        output.SetSpacing(self._spacing)

        num_points = np.prod(self._dims)
        tensor_field = np.zeros((num_points, 3, 3))

        # Generate a sample tensor field (this can be customized as needed)
        for i in range(num_points):
            tensor_field[i] = np.array([
                [i, 0, 0],
                [0, i, 0],
                [0, 0, i]
            ])

        # Convert the numpy array to a VTK array
        vtk_tensor_array = vtkDoubleArray()
        vtk_tensor_array.SetNumberOfComponents(9)
        vtk_tensor_array.SetNumberOfTuples(num_points)
        vtk_tensor_array.SetName("TensorField")

        # Flatten the tensor array and insert it into the VTK array
        vtk_tensor_array.SetArray(tensor_field.flatten(), num_points * 9, 1)

        output.GetPointData().SetTensors(vtk_tensor_array)

        return 1