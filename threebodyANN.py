import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Function f(x)
def mFactor(X):
    x = X[0]
    y = X[1]
    return x*y*(x-3)*(y-3)

# Derivative f(x)
def dermFactor(X,var):
    x = X[0]
    y = X[1]
    if var == 'x':
        return y*(y-3)*(2*x-3)
    elif var == 'y':
        return x*(x-3)*(2*y-3)

# Second derivative f(x)
def der2mFactor(X,var):
    x = X[0]
    y = X[1]
    if var == 'x':
        return 2*y*(y-3)
    elif var == 'y':
        return 2*x*(x-3)

# Apply the function into the tensor
def mapfun(fun,tensor,var=None):
    result = torch.ones([tensor.size()[0],1])
    if fun == mFactor:
        for i in range(tensor.size()[0]):
            result[i] *= fun(tensor[i])
    else:
        for i in range(tensor.size()[0]):
            result[i] *= fun(tensor[i],var=var)
    return result

# Fixes the seed
# seed = 0
# torch.manual_seed(seed)

# Physical parameters
avar = (0,0) # (lx,ly)
mu_x = 1
mu_y = 2
omega_x = 3
omega_y = 1

# Parameters
Nin = 2 # Inputs
Nout = 1 # Outputs
Nh = 5 # Hidden layer size

Nmesh = 50  # Intervals! (not points) (same for x and y)
xlim = (0,3) # x interval
ylim = (0,3) # y interval
h = (xlim[1]-xlim[0])/(Nmesh)

x_train = torch.linspace(xlim[0],xlim[1],Nmesh+1)
y_train = torch.linspace(ylim[0],ylim[1],Nmesh+1)

# Creates the grid
grid = torch.cartesian_prod(x_train,y_train)

# Derivatives of the mFactor
func = mapfun(mFactor,grid)
fxder = mapfun(dermFactor,grid,var='x')
fyder = mapfun(dermFactor,grid,var='y')
fx2der = mapfun(der2mFactor,grid,var='x')
fy2der = mapfun(der2mFactor,grid,var='y')

grid.requires_grad = True

# NEURAL NETWORK
class ANNv4(nn.Module):
    def __init__(self):
        super(ANNv4, self).__init__()
        self.linear1 = nn.Linear(Nin,Nh,bias = True)
        self.softplus = nn.Softplus()
        self.linear2 = nn.Linear(Nh,Nout,bias = False)

    def forward(self, x):
        out = self.linear1(x)
        out = self.softplus(out)
        out = self.linear2(out)

        return out

# 1st derivative numerical
def der1num(x, var):
    mat = torch.reshape(x,(Nmesh+1,Nmesh+1))

    if var == 'y':
        mat = torch.transpose(mat,0,1)
    elif var == 'x':
        pass

    dif = torch.diff(mat,dim = 0)
    der = (dif[1:]+dif[:-1])/(2*h)

    # Extremes
    adv = (mat[1]-mat[0])/h
    ret = (mat[-1]-mat[-2])/h
    adv = torch.unsqueeze(adv,0)
    ret = torch.unsqueeze(ret,0)

    fmat = torch.cat((adv,der,ret))

    if var == 'y':
        fmat = torch.transpose(fmat,0,1)
    elif var == 'x':
        pass

    return torch.reshape(fmat,(-1,1))

# 2nd derivative numerical
def der2num(x,var):
    mat = torch.reshape(x,(Nmesh+1,Nmesh+1))

    if var == 'y':
        mat = torch.transpose(mat,0,1)
    elif var == 'x':
        pass

    der = torch.diff(torch.diff(mat,dim = 0),dim = 0)/h**2

    # Extremes
    adv = (mat[2]-2*mat[1]+mat[0])/h**2
    ret = (mat[-3]-2*mat[-2]+mat[-1])/h**2
    adv = torch.unsqueeze(adv,0)
    ret = torch.unsqueeze(ret,0)

    fmat = torch.cat((adv,der,ret))

    if var == 'y':
        fmat = torch.transpose(fmat,0,1)
    elif var == 'x':
        pass

    return torch.reshape(fmat,(-1,1))

# Inialitze the integration weigts (Simpson --> 1 2 4 2 ... 4 1)
def ini_weights():
    global W

    Nx = Nmesh+1
    Ny = Nmesh+1

    # Weight matrices:
    Wx = []
    Wx.append(1)
    for i in range(Nx-2):
        if i%2==0:
            Wx.append(4)
        else:
            Wx.append(2)
    Wx.append(1)
    Wx = torch.tensor(Wx,dtype=torch.float32)
    Wx = Wx.reshape(1,Nx)

    Wy = []
    Wy.append(1)
    for i in range(Ny-2):
        if i%2==0:
            Wy.append(4)
        else:
            Wy.append(2)
    Wy.append(1)
    Wy = torch.tensor(Wy,dtype=torch.float32)
    Wy = Wy.reshape(Ny,1)

    W = torch.matmul(Wy,Wx)
    W = W.reshape((-1,1))

# Loss/cost function
def cost(): # Energy
    lx,ly = avar #avar --> angular variables (lx,ly)
    G = grid.clone()

    Z = model(G)

    # DERIVATIVES
    
    # Autograd
    # first_der, = torch.autograd.grad(outputs=Z, grad_outputs=torch.ones_like(Z), inputs=grid, create_graph=True)
    # second_der, = torch.autograd.grad(outputs=first_der, grad_outputs=torch.ones_like(first_der), inputs=grid, create_graph=True)
    
    # No autograd
    d1x = der1num(Z,var = 'x')
    d1y = der1num(Z,var = 'y')
    d2x = der2num(Z,var = 'x')
    d2y = der2num(Z,var = 'y')

    # Chain rule
    dx = fx2der*Z + 2*fxder*d1x + func*d2x
    dy = fy2der*Z + 2*fyder*d1y + func*d2y

    # Column vectors
    x_v, y_v = G[:,0], G[:,1]
    x_v = torch.reshape(x_v,(-1,1))
    y_v = torch.reshape(y_v,(-1,1))

    # If derivatives are calculated via autograd:
    # dx, dy = second_der[:,0],second_der[:,1]
    # dx = torch.reshape(dx,(-1,1))
    # dy = torch.reshape(dy,(-1,1))

    N = h**2*torch.sum(W*func*Z*func*Z)/9
    V = h**2*torch.sum(W*func*Z*(lx*(lx+1)+ly*(ly+1)+mu_x*omega_x**2*x_v**2/2+mu_y*omega_y**2*y_v**2/2)*func*Z)/(N*9)
    K = h**2*torch.sum(W*func*Z*(-dx/(2*mu_x)-dy/(2*mu_y)))/(N*9)
    E = V + K
    phi = (func*Z)/torch.sqrt(N)

    return E, V, K, phi

ini_weights()

# TRAINING
epochs = 1000 # Iterations

model = ANNv4()

# LOAD TRAINED MODEL
# lPATH = 'v4/models/parabola100.pth'
# model.load_state_dict(torch.load(lPATH))
# model.eval()

# File:
NOM = 'prova'
file = open('v4/txt/'+NOM+'.txt', 'w')

learningrate = 1e-2
#optimizer = torch.optim.RMSprop(model.parameters(),lr=learningrate)
optimizer = torch.optim.Adam(model.parameters(),lr=learningrate)

tot_loss = []

# TRAINING
for i in range(epochs):
    loss, U, K, phi = cost()

    optimizer.zero_grad() # initialize gradients
    loss.backward()       # computation of the gradients
    optimizer.step()      # update of the parameters

    tot_loss.append(loss.detach())

    with torch.no_grad():
        file.write(f'{i+1}\t{loss:.8f}\t{U:.8f}\t{K:.8f}\n')

    if i%1000 == 0:
        print(f'Epoch: {i+1}, loss: {loss}')

file.close()


# Save model
sPATH = 'v4/models/'+NOM+'.pth'
torch.save(model.state_dict(),sPATH)

file = open('v4/wavefn/'+NOM+'_wf.txt','w')
for i in phi:
    file.write(str(i.item())+'\n')
file.close()

#Plot energy vs iterations
# plt.plot(torch.linspace(1, epochs, epochs).numpy(), tot_loss,label='$E$')
# plt.show()

print(f'Minimum energy: {min(tot_loss):.8f}')
