import os
import numpy as np
import matplotlib.pyplot as plt

#创建Result文件夹
result_dir = "Result"
os.makedirs(result_dir, exist_ok=True)

hidden_layer=64
input_size=1024
output_size=1
lr_rate=0.05
epochs=8192
boundary=np.pi
lr_down=0.95
class ReLULayer:
    def __init__(self):
        self.mask = None  # 用于记录前向传播中<=0的位置

    def forward(self, x):
        self.mask = (x <= 0)  # 记录哪些元素小于等于0
        return np.maximum(0, x)  # ReLU计算

    def backward(self, dout):
        dout[self.mask] = 0  # 对前向传播中<=0的位置，梯度置0
        return dout

class AffineLayer:
    def __init__(self,input_dim,output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim) #权重的初始化
        self.b=np.zeros(output_dim)  #偏置的初始化
        self.out=None
        self.grads={}
        self.x=None

    def forward(self,x):
        self.out=np.dot(x,self.W)+self.b
        self.x=x
        return self.out

    def backward(self,dout):
        #计算输入值的梯度
        dx=np.dot(dout,self.W.T)
        #计算权重的梯度
        dW=np.dot(self.x.T,dout)
        #计算偏置的梯度
        db=np.sum(dout,axis=0)
        self.grads['W']=dW
        self.grads['b']=db
        return dx
    def upgrade(self,lr_rate):
        self.W-=lr_rate*self.grads['W']
        self.b-=lr_rate*self.grads['b']

def generate_data(input):
    np.random.seed(1217)#固定种子重复实验
    x=np.linspace(-boundary,boundary,input).reshape(-1,1)
    #要拟合的函数
    y = np.sin(0.5 * np.pi * x) + np.sin(np.pi * x)
    x_input = (x - x.min()) / (x.max() - x.min())  # 归一化
    return x_input,y.reshape(-1,1)

class Loss:
    def __init__(self):
        self.y_pred=None
        self.y_true=None
        self.loss=None

    def forward(self,y_pred,y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.loss=np.mean((y_true-y_pred)**2)
        return self.loss

    def backward(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]

def generate_plot_data():
    x_plot = np.linspace(-boundary, boundary, input_size).reshape(-1, 1)
    x_plot_input = (x_plot - (-boundary)) / (boundary - (-boundary))  # 与训练数据相同的归一化
    y_plot = np.sin(0.5 * np.pi * x_plot) + np.sin(np.pi * x_plot)
    return x_plot, x_plot_input, y_plot

# 生成训练数据和绘图数据
x_train, y_train = generate_data(input_size)
x_plot_original, x_plot_input, y_plot_true = generate_plot_data()

#生成数据
x,y=generate_data(input_size)


Layer1=AffineLayer(1,hidden_layer)  #输入层到隐藏层之间的线性层
Relu1=ReLULayer()
Layer2=AffineLayer(hidden_layer,hidden_layer)
Relu2=ReLULayer()
Layer3=AffineLayer(hidden_layer,hidden_layer)
Relu3=ReLULayer()
Layer4=AffineLayer(hidden_layer,hidden_layer)
Relu4=ReLULayer()
Layer5=AffineLayer(hidden_layer,output_size)#隐藏层到输出层的线性层
Loss=Loss()

#训练循环
train_losses=[]
for epoch in range(epochs):
    a1=Layer1.forward(x)
    z1=Relu1.forward(a1)
    a2=Layer2.forward(z1)
    z2=Relu2.forward(a2)
    a3=Layer3.forward(z2)
    z3=Relu3.forward(a3)
    a4=Layer4.forward(z3)
    z4=Relu4.forward(a4)
    a5=Layer5.forward(z4)
    y_pred=a5


    loss=Loss.forward(y_pred,y)
    train_losses.append(loss)

    # 反向传播
    dout = Loss.backward()
    dout = Layer5.backward(dout)
    dout = Relu4.backward(dout)
    dout = Layer4.backward(dout)
    dout = Relu3.backward(dout)
    dout = Layer3.backward(dout)
    dout = Relu2.backward(dout)
    dout = Layer2.backward(dout)
    dout = Relu1.backward(dout)
    _ = Layer1.backward(dout)

    #更新数据
    Layer1.upgrade(lr_rate*lr_down**(epoch//1000))
    Layer2.upgrade(lr_rate*lr_down**(epoch//1000))
    Layer3.upgrade(lr_rate*lr_down**(epoch//1000))
    Layer4.upgrade(lr_rate*lr_down**(epoch//1000))
    Layer5.upgrade(lr_rate*lr_down**(epoch//1000))

    if epoch % 100 == 0:
        print("epoch:{}, loss:{:.4f}".format(epoch, loss))

    if epoch % 1000 == 0 or epoch == epochs - 1:
        # 预测绘图数据
        a1_p = Layer1.forward(x_plot_input)
        z1_p = Relu1.forward(a1_p)
        a2_p = Layer2.forward(z1_p)
        z2_p = Relu2.forward(a2_p)
        a3_p = Layer3.forward(z2_p)
        z3_p = Relu3.forward(a3_p)
        a4_p = Layer4.forward(z3_p)
        z4_p = Relu4.forward(a4_p)
        y_plot_pred = Layer5.forward(z4_p)

        # 创建可视化图像
        plt.figure(figsize=(12, 6))
        plt.plot(x_plot_original, y_plot_true, 'r--', linewidth=2, label='True Function')
        plt.plot(x_plot_original, y_plot_pred, 'b-', linewidth=1.5, label='Prediction')
        plt.title(f'Epoch: {epoch}  Loss: {loss:.4f}')
        plt.xlabel('x'), plt.ylabel('y')
        plt.legend(), plt.grid(True)
        plot_path = os.path.join(result_dir, f'Fitting_Progress_Epoch_{epoch}.png')
        plt.savefig(plot_path)
        plt.close()

# 绘制Loss曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel("Epoch"), plt.ylabel("Loss")
plt.title("Loss Curve")
final_loss_path = os.path.join(result_dir, 'Final_Loss_Curve.png')
plt.savefig(final_loss_path)
plt.show()
