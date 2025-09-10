// sphere-snappy.geo

// 1) Use the OpenCASCADE kernel
SetFactory("OpenCASCADE");

// 2) Enable curvature‐based sizing (20 elements per full curvature)
Mesh.MeshSizeFromCurvature = 20;

// 3) Outer sphere
Sphere(1) = {0, 0, 0, 10.0};

// 4) Surface‐only mesh
Mesh 2;

// 5) Export to STL
Save "sphere.stl";
