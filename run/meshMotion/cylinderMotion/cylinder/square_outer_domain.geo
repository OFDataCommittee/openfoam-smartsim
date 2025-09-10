
SetFactory("OpenCASCADE");

// Parameters
L = 3.0;     // Side length of square
h = 1.0;     // Extrusion height
lc = 0.01;   // Mesh size

// Global mesh size
Mesh.CharacteristicLengthMin = lc;
Mesh.CharacteristicLengthMax = lc;

// Half length
s = L / 2.0;

// Bottom wire (z = 0)
Point(1) = {-s, -s, 0, lc};
Point(2) = { s, -s, 0, lc};
Point(3) = { s,  s, 0, lc};
Point(4) = {-s,  s, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Top wire (z = h)
Point(5) = {-s, -s, h, lc};
Point(6) = { s, -s, h, lc};
Point(7) = { s,  s, h, lc};
Point(8) = {-s,  s, h, lc};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Connect bottom and top wires into ruled surfaces
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Create ruled surfaces for each side
Line Loop(21) = {1, 10, -5, -9};
Ruled Surface(21) = {21};

Line Loop(22) = {2, 11, -6, -10};
Ruled Surface(22) = {22};

Line Loop(23) = {3, 12, -7, -11};
Ruled Surface(23) = {23};

Line Loop(24) = {4, 9, -8, -12};
Ruled Surface(24) = {24};

// Group all sides
Physical Surface("outer_box_sides") = {21, 22, 23, 24};

// Mesh and export
Mesh 2;
Save "square_outer_domain.stl";
