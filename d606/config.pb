language: PYTHON
name:     "optimize"

variable {
 name: "n_comp"
 type: INT
 size: 1
 min:  2
 max:  15
}

variable {
 name: "C"
 type: FLOAT
 size: 1
 min:  0.1
 max:  1
}

variable {
 name: "kernel"
 type: ENUM
 size: 1
 options: "linear"
 options: "poly"
 options: "rbf"
}

variable {
 name: "band_list"
 type: ENUM
 size: 1
 options: "[[8, 12], [12, 16], [16, 20], [20, 24]]"
 options: "[[8, 12], [12, 16], [16, 20], [20, 24], [24, 28]]"
 options: "[[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32]]"
}

variable {
 name: "oacl_ranges"
 type: ENUM
 size: 1
 options: "((2, 7), (7, 15))"
 options: "((1, 2), (2, 3))"
 options: "((4, 6), (7, 9))"
}
