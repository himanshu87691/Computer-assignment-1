import numpy as np

# Define the size of the matrix
n = 100

# Initialize the main diagonal, super-diagonal, and sub-diagonal
main_diag = np.full(n, 2.0004)
super_diag = np.full(n-1, -1)
sub_diag = np.full(n-1, -1)

# Adjust the last sub-diagonal for boundary conditions
sub_diag[-1] = -2

# Initialize the RHS vector
rhs = np.zeros(n)
rhs[0] = 1
#labelling for easy reading purpose
a=sub_diag
b=main_diag
c=super_diag
d=rhs

## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) code
def TDMAsolver(a, b, c, d):

    nf = len(d) # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d)) # copy arrays
    for it in range(1, nf):
        mc = ac[it-1]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
        	    
    xc = bc
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    return xc

print (TDMAsolver(a, b, c, d))