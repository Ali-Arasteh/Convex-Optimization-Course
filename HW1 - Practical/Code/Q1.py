import cvxpy as cp
import math

x = cp.Variable()
y = cp.Variable()
obj = cp.Minimize(2*cp.exp(x+y)+(2*cp.exp(x)+math.pi*cp.exp(y)))
constraints = [cp.log(300)-(x+y)<=0, y-x<=0, x-y-cp.log(2)<=0, cp.log(10)-y<=0, y-cp.log(20)<=0, cp.log(20)-x<=0, x-cp.log(30)<=0]
prob = cp.Problem(obj, constraints)
prob.solve()
l = math.exp(x.value)
w = math.exp(y.value)
print(l, w)