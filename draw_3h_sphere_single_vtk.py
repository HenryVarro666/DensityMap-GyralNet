import vtk
import os
import shutil

def create_colored_sphere(point, radius):
    # Create a sphere source
    sphere_source = vtk.vtkSphereSource()
    sphere_source.SetCenter(point)
    sphere_source.SetRadius(radius)
    sphere_source.Update()

    # Create a color array
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    
    # Add the color green to the entire sphere
    num_points = sphere_source.GetOutput().GetNumberOfPoints()
    for i in range(num_points):
        colors.InsertNextTuple3(0, 0, 255)  # RGB for green

    sphere_source.GetOutput().GetPointData().SetScalars(colors)
    return sphere_source.GetOutput()

def main(vtk_file_path, txt_file_path, output_file_path, sphere_radius):
    # Read the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()

    polydata = reader.GetOutput()
    points = polydata.GetPoints()

    # Read the txt file to get point IDs
    with open(txt_file_path, 'r') as f:
        point_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

    append_filter = vtk.vtkAppendPolyData()

    # Create and append a colored sphere for each selected point
    for point_id in point_ids:
        if point_id < points.GetNumberOfPoints():
            point = points.GetPoint(point_id)
            sphere_polydata = create_colored_sphere(point, sphere_radius)
            append_filter.AddInputData(sphere_polydata)

    # Check if there are any inputs added to the append filter
    if append_filter.GetNumberOfInputConnections(0) > 0:
        append_filter.Update()
        # Write the combined spheres to a single VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(output_file_path)
        writer.SetInputData(append_filter.GetOutput())
        writer.Write()
    else:
        print(f"No valid spheres to write for {vtk_file_path}")

if __name__ == '__main__':
    root = '.'
    solidcolor = 'Green'
    sphere_radius = 2

    for subject in ['100206']:
        print(f"Processing subject: {subject}")
        h = ['lh', 'rh']
        for hemi in h:
            vtk_file_path = os.path.join(root, subject, f'{hemi}.withGrad.164k_fsaverage.flip.Inner.vtk')
            txt_file_path = os.path.join(root, subject, 'gyralnet_island_164k_flip', f'{hemi}_3hinge_ids.txt')
            output_file_path = os.path.join(root, subject, f'{hemi}_3hinge_spheres_{solidcolor}_radius_{sphere_radius}.vtk')
            main(vtk_file_path, txt_file_path, output_file_path, sphere_radius)

    # for subject in os.listdir(root):
    #     if not subject.startswith('.') and os.path.isdir(os.path.join(root, subject)) and subject.isdigit():
    #         print(f"Processing subject: {subject}")
    #         h = ['lh', 'rh']
    #         for hemi in h:
    #             vtk_file_path = os.path.join(root, subject, f'{hemi}.withGrad.164k_fsaverage.flip.rescale.Inner.vtk')
    #             txt_file_path = os.path.join(root, subject, 'gyralnet_island', f'{hemi}_3hinge_ids.txt')
    #             output_file_path = os.path.join(root, subject, f'{hemi}_3hinge_spheres_{solidcolor}.vtk')
    #             main(vtk_file_path, txt_file_path, output_file_path, sphere_radius)