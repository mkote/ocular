language: PYTHON
name:     "optimize"

variable {
 name: "n_comp"
 type: INT
 size: 1
 min:  2
 max:  8
}

variable {
 name: "X"
 type: FLOAT
 size: 1
 min:  0.1
 max:  1
}

# Enumeration example
# 
# variable {
#  name: "Z"
#  type: ENUM
#  size: 3
#  options: "foo"
#  options: "bar"
#  options: "baz"
# }


