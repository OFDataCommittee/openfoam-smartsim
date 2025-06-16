// ellipsoid-snappy.geo

// 1) Use the OpenCASCADE kernel
SetFactory("OpenCASCADE");

// 2) Enable curvature‐based sizing (50 elements per full curvature)
Mesh.MeshSizeFromCurvature = 100;

// 3) Unit sphere → ellipsoid
Sphere(1) = {0, 0, 0, 1.0};
Dilate {{0, 0, 0}, {2.0, 2.0, 1.0}} {
  Volume {1};
}

// 4) Surface‐only mesh
Mesh 2;

// 5) Export to STL
Save "ellipsoid.stl";
