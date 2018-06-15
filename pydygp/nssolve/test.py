import numpy as np
from scipy.linalg import block_diag
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
A0 = np.random.normal(size=4).reshape(2, 2)
A0 = np.zeros((2, 2))
A1 = np.random.normal(size=4).reshape(2, 2)
A2 = np.random.normal(size=4).reshape(2, 2)

g1 = np.random.normal(size=3)
g2 = np.random.normal(size=3)

F0 = A0 + A1*g1[0] + A2*g2[0]
F1 = A0 + A1*g1[1] + A2*g2[1]
F2 = A0 + A1*g1[2] + A2*g2[2]

I = np.identity(2)

FF = np.column_stack([F0, F1, F2])

b = np.column_stack([A0]*3)

At = np.row_stack([A1.T, A2.T])
A = np.column_stack([A1, A2])
_g = np.array([[g1[0], g2[0]],
               [g1[1], g2[1]],
               [g1[2], g2[2]]])

gg = np.concatenate((g1, g2))
vecF = FF.T.ravel()
print(vecF)
l1 = A1.T.ravel()
ell1 = block_diag(*[l1[:, None], l1[:, None], l1[:, None]])
l2 = A2.T.ravel()
ell2 = block_diag(*[l2[:, None], l2[:, None], l2[:, None]])
ell = np.column_stack((ell1, ell2))
print(np.dot(ell, gg))
#gI = np.kron(_g, I)
#print(A)
#vecA = A.T.ravel()
#print(vecA)
#print(np.dot(gI, At).T)
#print('-------------------')
#print(A1*g1[0] + A2*g2[0])
#print(A1*g1[1] + A2*g2[1])
#print('-------------------')
#expr2 = np.dot(np.kron(A, np.ones((1, 1))),
#               np.kron(_g.T, I))
#print(expr2)
#print('-------------------')
#M1 = A
#M2 = np.kron(_g.T, I)
#print(np.dot(M1, M2))
"""
X = np.random.normal(size=6).reshape(3, 2)

tt = np.linspace(0., 1., 3)
gg = np.random.normal(size=5)
L0 = A0 + A1*gg[0]
L1 = A0 + A1*gg[1]
L2 = A0 + A2*gg[2]

LL = np.row_stack([L0.T, L1.T, L2.T])

wt0 = (tt[1] - tt[0])/2
wt1 = (tt[1] - tt[0])/2
ws0 = (tt[2] - tt[0])/6
ws1 = 4*(tt[2] - tt[0])/6
ws2 = (tt[2] - tt[0])/6

b = np.row_stack([X[0, ]]*3)

W1 = np.zeros((3, 6))
W1[1, :2] = wt0*X[0, ]
W1[1, 2:4] = wt1*X[1, ]
W1[2, :2] = ws0*X[0, ]
W1[2, 2:4] = ws1*X[1, ]
W1[2, 4:] = ws2*X[2, ]

res = np.dot(W1, LL) + b

res2 = X[0, ] + ws0*np.dot(L0, X[0, ]) + ws1*np.dot(L1, X[1, ]) + ws2*np.dot(L2, X[2, ])
"""
