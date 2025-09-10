// ellipsoid_in_sphere.geo

// 1) Use the OpenCASCADE kernel
SetFactory("OpenCASCADE");

// 2) Enable curvature‐based sizing:
//    Mesh.MeshSizeFromCurvature = 20 means Gmsh will place roughly 20 elements around a full 2π turn 
//    on any circle of curvature. In other words, at each point the local curvature radius R yields 
//    a target edge length h ≃ (2π·R)/20.
Mesh.MeshSizeFromCurvature = 30;

// 3) Create the outer sphere (tag = 1) of radius 10
Sphere(1) = {0, 0, 0, 10.0};

// 4) Create a unit‐sphere (tag = 2), then immediately Dilate it into an ellipsoid of semi‐axes (2,2,1)
Sphere(2) = {0, 0, 0, 1.0};
Dilate {{0, 0, 0}, {2.0, 2.0, 1.0}} {
  Volume{ 2 };
}

// 5) Subtract the ellipsoid (tag 2) from the outer sphere (tag 1), deleting both originals
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }

// 6) Finally, generate the 3D mesh of the hollow region
Mesh 3;
