SetFactory("OpenCASCADE");

// Parameters
r = 1.0;
h = 1.0;
lc = 0.05;

// Force uniform global mesh size
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Create open cylinder (no caps)
Cylinder(1) = {0, 0, 0, 0, 0, h, r};

// Export only lateral surface (Surface ID 1)
Physical Surface("inner_cylinder") = {1};

// Mesh and export
Mesh 2;
Save "inner_cylinder.stl";

