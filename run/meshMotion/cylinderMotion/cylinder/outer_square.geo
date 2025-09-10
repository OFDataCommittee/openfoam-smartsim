
SetFactory("OpenCASCADE");

// Parameters
L = 3.0;      // Side length of square
h = 1.0;      // Height of extrusion
lc = 0.05;    // Mesh size

// Global mesh size control
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Half side length
s = L / 2.0;

// Define square in XY plane centered at (0,0)
Point(1) = {-s, -s, 0, lc};
Point(2) = { s, -s, 0, lc};
Point(3) = { s,  s, 0, lc};
Point(4) = {-s,  s, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(10) = {1, 2, 3, 4};
//Plane Surface(10) = {10};

// Extrude square in Z direction by height h
out[] = Extrude {0, 0, h} {
  Surface{10};
  Layers{1};
  Recombine;
};

// Export lateral surfaces only (side walls)
Physical Surface("outer_square") = {out}; 

// Mesh and export
Mesh 2;
Save "outer_square.stl";
