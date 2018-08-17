import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N, d_in, h, d_out = 64, 1000, 100, 10

x = Variable(torch.randn( (N, d_in) ).type(dtype), requires_grad=False)
y = Variable(torch.randn( N, d_out ).type(dtype), requires_grad=False)

w1 = Variable(torch.randn( d_in, h ).type(dtype), requires_grad=True)
w2 = Variable(torch.randn( h, d_out ).type(dtype), requires_grad=True)

learning_rate = 1e-6

print("auto grade")
for _ in range(500):
    #定义并计算数据流
    h1 = x.mm(w1)
    h_relu = h1.clamp(min=0)
    y_pred = h_relu.mm(w2)

    #计算经验风险函数
    loss = ( y_pred - y ).pow(2).sum()
    print(loss)

    #自动进行back propagation导数计算
    loss.backward()

    #迭代一步
    w1.data -= learning_rate*w1.grad.data
    w2.data -= learning_rate*w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()

#============================================================================
"""
x = torch.randn( (N, d_in) ).type(dtype)
y = torch.randn( N, d_out ).type(dtype)

w1 = torch.randn( d_in, h ).type(dtype)
b1 = torch.randn( N, h ).type(dtype)
w2 = torch.randn( h, d_out ).type(dtype)
b2 = torch.randn( d_out ).type(dtype)

print("using relu activation function")
for _ in range(500):
    #定义并计算数据流
    h1 = x.mm(w1) + b1
    h_relu = h1.clamp(min=0)
    y_pred = h_relu.mm(w2) + b2

    #计算经验风险函数
    loss = ( y_pred - y ).pow(2).sum()
    print(loss)

    #back propagation导数计算
    grad_y_pred = 2.0*(y_pred-y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h1 < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    grad_b1 = grad_h

    #迭代一步
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
    b1 -= learning_rate*grad_b1
    b2 -= learning_rate


#=====================================================================
print("using sigmod activation function")
#sigmod
w1 = torch.randn(d_in, h).type(dtype)
w2 = torch.randn(h, d_out).type(dtype)
for _ in range(500):
    #定义并计算数据流
    h1 = x.mm(w1) + b1
    h_sigmod = 1/(1+torch.exp(-h1))
    y_pred = h_sigmod.mm(w2) + b2

    #计算经验风险函数
    loss = ( y_pred - y ).pow(2).sum()
    print(loss)

    #back propagation导数计算
    grad_y_pred = 2.0*(y_pred-y)
    grad_w2 = h_sigmod.t().mm(grad_y_pred)
    grad_h_sigmod = grad_y_pred.mm(w2.t())
    grad_h = grad_h_sigmod.mm(h_sigmod.t().mm(1-h_sigmod)) #sigmod'=sigmod(1-sigmod)
    grad_w1 = x.t().mm(grad_h)
    grad_b1 = grad_h

    #迭代一步
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2
    b1 -= learning_rate*grad_b1
    b2 -= learning_rate
"""