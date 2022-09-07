'''

linear6d.py

'''

import numpy as np
from utils.mpfit import mpfit
import copy
import pickle

#### Begin Functions


####
# Function to calculate new x/y positions given a 6D parameter solution
####
def linear1(x, par):

  imod = int(len(x)/2.0)
  model = np.zeros((len(x)))

  for a in range(imod):
    model[a] = par[0] + par[1]*x[a] + par[2]*x[a+imod]
    model[a+imod] = par[3] + par[4]*x[a] + par[5]*x[a+imod]


  return model

####

####
# Function to interact with mpfit
####
def linear1_lin(par, fjac=None, x=None, y=None, err=None):
  # Parameter values are passed in "par"
  # If fjac==None then partial derivatives should not be
  # computed.  It will always be None if MPFIT is called with default
  # flag.
  model = linear1(x, par)

  # Non-negative status value means MPFIT should continue, negative means
  # stop the calculation.
  status = 0
  return [status, (y-model)/err]

####

####
# Now write main program
####

####
# Need eight 1D arrays: the x and y coordinates of the master frame,
# the x and y coordinates of the frame to be transformed, already matched
# to the master frame, the weights for the x and y positions to be used
# in solving for the coefficients, and finally the x/y positions of all
# sources to be transformed.
def test_linear(matchx, matchy, masterx, mastery, weix, weiy, allx, ally):

  master = np.hstack((masterx, mastery))
  match = np.hstack((matchx, matchy))
  error = np.hstack((weix, weiy))

  allvals = np.hstack((allx, ally))

  #print(matchx.shape, matchy.shape, allx.shape, ally.shape, match.shape, allvals.shape)

  xx = 6 	# number of parameters

  p0 = np.zeros((xx))
  p0.fill(1e-06)


  parbase = {'value':0., 'fixed':0, 'limited':[0,0], 'limits':[0., 0.]}

  parinfo = []

  for i in range(xx):
    parinfo.append(copy.deepcopy(parbase))

  for i in range(xx):
    parinfo[i]['value'] = p0[i]


  fa = {'x':match, 'y':master, 'err':error}

  m = mpfit(linear1_lin, p0, parinfo=parinfo, functkw=fa, ftol=1e-15, xtol=1e-15, \
            gtol=1e-15)


  # calculate the degrees of freedom for the problem
  dof = len(match) - len(m.params)


  # calculate the parameter correlation matrix pcor from the calculated 
  # covariance matrix cov
  cov = m.covar
  pcor = cov*0.

  for i in range(len(m.params)):
    for j in range(len(m.params)):
      pcor[i][j] = cov[i][j]/np.sqrt(cov[i][i] * cov[j][j])


  # save out the summed squared residuals for the returned parameters
  bestnorm = m.fnorm


  # create array to store the results of the run
  res = np.zeros((xx+3, xx))

  res[0][0] = bestnorm
  res[0][1] = dof
  res[0][2] = m.status

  # save out the converged parameters
  res[1] = m.params

  # save out the [i,i] series of the pcovar matrix
  for i in range(xx):
    res[2][i] = np.sqrt(pcor[i][i])

  # save out the full covariance matrix
  res[3:xx+3] = pcor


  # move the converged parameters into a separate array and pass into the other function
  par = m.params


  # calculate the new positions
  ar = linear1(match, par)

  match_new = np.zeros((int(len(ar)/2), 2))

  match_new[:,0] = ar[0:int(len(ar)/2)]
  match_new[:,1] = ar[int(len(ar)/2):]
  
  #np.savetxt("image1_it.dat", ar, fmt="%1.5f")

  ar2 = linear1(allvals, par)

  all_new = np.zeros((int(len(ar2)/2), 2))

  all_new[:,0] = ar2[0:int(len(ar2)/2)]
  all_new[:,1] = ar2[int(len(ar2)/2):]

  #np.savetxt("image1_iat.dat", ar2, fmt="%1.5f")
  with open("pcor.sav", "wb") as f:
    _ = pickle.dump(pcor, f)

  return match_new, all_new

####

if __name__ == "__main__":
  
  vals = np.loadtxt("image2_i.dat")	# Load in the data
  allvals = np.loadtxt("image2_ia.dat") # Load in all positions

  ix, iy, ierr = 0, 1, 2		# Set the column references

  x_valsx = vals[0:int(len(vals)/2),ix]
  x_valsy = vals[int(len(vals)/2):, ix]

  print(len(x_valsx), len(x_valsy))

  y_valsx = vals[0:int(len(vals)/2),iy]
  y_valsy = vals[int(len(vals)/2):, iy]

  err_x = vals[0:int(len(vals)/2),ierr]
  err_y = vals[int(len(vals)/2):, ierr]

  x_allx = allvals[0:int(len(allvals)/2),ix]
  x_ally = allvals[int(len(allvals)/2):, ix]

  vals_new, all_new = test_linear(x_valsx, x_valsy, y_valsx, y_valsy, err_x, err_y, x_allx, x_ally)

