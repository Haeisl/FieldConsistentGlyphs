from vtk import vtkNamedColors, vtkUnsignedCharArray


def create_color_array(name):
    color_array = vtkUnsignedCharArray()
    color_array.SetNumberOfComponents(3)
    color_array.SetName(name)
    return color_array


colors = vtkNamedColors()

group_map = {
    0: "center",
    1: "bottom",
    2: "side",
    3: "side",
    4: "arrowbase",
    5: "arrowbase",
    6: "arrowhead",
    7: "arrowhead",
}
group_color_values = {
    "center": colors.GetColor3ub("Gold"),
    "bottom": colors.GetColor3ub("Orchid"),
    "side": colors.GetColor3ub("SkyBlue"),
    "arrowbase": colors.GetColor3ub("Coral"),
    "arrowhead": colors.GetColor3ub("SpringGreen"),
}

orientation_map = {
    0: "parallel",
    1: "orthogonal",
    2: "parallel",
    3: "parallel",
    4: "orthogonal",
    5: "orthogonal",
    6: "combined",
    7: "combined",
}
orientation_color_values = {
    "parallel": colors.GetColor3ub("Tomato"),
    "orthogonal": colors.GetColor3ub("DodgerBlue"),
    "combined": colors.GetColor3ub("LimeGreen"),
}


group_colors = create_color_array("Group Colors")
orientation_colors = create_color_array("Orientation Colors")