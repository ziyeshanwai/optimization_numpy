import sympy
sympy.init_printing()

p00, p01, p02, p03, p10, p11, p12, p13, p20, p21, p22, p23, x, y, z, u, v, tmp = \
    sympy.symbols("p_{00}, p_{01}, p_{02}, p_{03}, p_{10}, p_{11}, p_{12}, "
                  "p_{13}, p_{20}, p_{21}, p_{22}, p_{23}, x, y, z, u, v, tmp")

P = sympy.Matrix([[p00, p01, p02, p03], [p10, p11, p12, p13], [p20, p21, p22, p23]])
X = sympy.Matrix([[x], [y], [z], [1]])
b = sympy.Matrix([[u], [v], [tmp]])
result = sympy.solve(P*X-b, [x, y, z])
print(result[x])
