# Regression：linear model

> 這裡用的是 Adagrad ，接下來的課程會再細講，這裡只是想顯示 gradient descent 實作起來沒有想像的那麼簡單，還有很多小技巧要注意

這裡採用最簡單的 linear model：**y_data=b+w\*x_data**

我們要用 gradient descent 把 b 和 w 找出來

當然這個問題有 closed-form solution，這個 b 和 w 有更簡單的方法可以找出來；那我們假裝不知道這件事，我們練習用 gradient descent 把 b 和 w 找出來

#### 數據準備：

```python
# 假設x_data和y_data都有10筆，分別代表寶可夢進化前後的cp值
x_data=[338.,333.,328.,207.,226.,25.,179.,60.,208.,606.]
y_data=[640.,633.,619.,393.,428.,27.,193.,66.,226.,1591.]
# 這裡採用最簡單的linear model：y_data=b+w*x_data
# 我們要用gradient descent把b和w找出來
```

#### 計算梯度微分的函數 getGrad()

```python
# 計算梯度微分的函數getGrad()
def getGrad(b,w):
    # initial b_grad and w_grad
    b_grad=0.0
    w_grad=0.0
    for i in range(10):
        b_grad+=(-2.0)*(y_data[i]-(b+w*x_data[i]))
        w_grad+=(-2.0*x_data[i])*(y_data[i]-(b+w*x_data[i]))
    return (b_grad,w_grad)
```

### 1、自己寫的版本

當兩個微分值 b_grad 和 w_grad 都為 0 時，gradient descent 停止，b 和 w 的值就是我們要找的最終參數

```python
# 這是我自己寫的版本，事實證明結果很糟糕。。。
# y_data=b+w*x_data
# 首先，這裡沒有用到高次項，僅是一個簡單的linear model，因此不需要regularization版本的loss function
# 我們只需要隨機初始化一個b和w，然後用b_grad和w_grad記錄下每一次iteration的微分值；不斷循環更新b和w直至兩個微分值b_grad和w_grad都為0，此時gradient descent停止，b和w的值就是我們要找的最終參數

b=-120 # initial b
w=-4 # initial w
lr=0.00001 # learning rate
b_grad=0.0
w_grad=0.0
(b_grad,w_grad)=getGrad(b,w)

while(abs(b_grad)>0.00001 or abs(w_grad)>0.00001):
    #print("b: "+str(b)+"\t\t\t w: "+str(w)+"\n"+"b_grad: "+str(b_grad)+"\t\t\t w_grad: "+str(w_grad)+"\n")
    b-=lr*b_grad
    w-=lr*w_grad
    (b_grad,w_grad)=getGrad(b,w)

print("the function will be y_data="+str(b)+"+"+str(w)+"*x_data")

error=0.0
for i in range(10):
    error+=abs(y_data[i]-(b+w*x_data[i]))
average_error=error/10
print("the average error is "+str(average_error))
```

    the function will be y_data=-inf+nan*x_data
    the average error is nan

上面的數據輸出處於隱藏狀態，點擊即可顯示

### 2、這裡使用李宏毅老師的 demo 嘗試

#### 引入需要的庫

```python
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
%matplotlib inline
import random as random
import numpy as np
import csv
```

#### 準備好 b、w、loss 的圖像數據

```python
# 生成一組b和w的數據圖，方便給gradient descent的過程做標記
x = np.arange(-200,-100,1) # bias
y = np.arange(-5,5,0.1) # weight
Z = np.zeros((len(x),len(y))) # color
X,Y = np.meshgrid(x,y)
for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]

        # Z[j][i]存儲的是loss
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - (b + w * x_data[n]))**2
        Z[j][i] = Z[j][i]/len(x_data)
```

#### 規定迭代次數和 learning rate，進行第一次嘗試

距離最優解還有一段距離

```python
# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.0000001 # learning rate
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)

# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]

# iterations
for i in range(iteration):

    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)

    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad

    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()
```

<center><img src="https://img-blog.csdnimg.cn/20200123150335527.png" width="60%;"/></center>

#### 把 learning rate 增大 10 倍嘗試

發現經過 100000 次的 update 以後，我們的參數相比之前與最終目標更接近了，但是這裡有一個劇烈的震蕩現象發生

```python
# 上圖中，gradient descent最終停止的地方里最優解還差很遠，
# 由於我們是規定了iteration次數的，因此原因應該是learning rate不夠大，這裡把它放大10倍

# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.000001 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)

# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]

# iterations
for i in range(iteration):

    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)

    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad

    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()
```

<center><img src="https://img-blog.csdnimg.cn/20200123150716524.png" width="60%;"/></center>

#### 把 learning rate 再增大 10 倍

發現此時 learning rate 太大了，參數一 update，就遠遠超出圖中標注的範圍了

所以我們會發現一個很嚴重的問題，如果 learning rate 變小一點，他距離最佳解還是會具有一段距離；但是如果 learning rate 放大，它就會直接超出範圍了

```python
# 上圖中，gradient descent最終停止的地方里最優解還是有一點遠，
# 由於我們是規定了iteration次數的，因此原因應該是learning rate還是不夠大，這裡再把它放大10倍

# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 0.00001 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)

# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]

# iterations
for i in range(iteration):

    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)

    # update b and w
    b -= lr * b_grad
    w -= lr * w_grad

    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)

# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()
```

<center><img src="https://img-blog.csdnimg.cn/2020012315075713.png" width="60%;"/></center>

這個問題明明很簡單，可是只有兩個參數 b 和 w，gradient descent 搞半天都搞不定，那以後做 neural network 有數百萬個參數的時候，要怎麼辦呢

這個就是**一室不治何以國家為**的概念

#### 解決方案：Adagrad

我們給 b 和 w 訂制化的 learning rate，讓它們兩個的 learning rate 不一樣

```python
# 這裡給b和w不同的learning rate

# y_data = b + w * x_data
b = -120 # initial b
w = -4 # initial w
lr = 1 # learning rate 放大10倍
iteration = 100000 # 這裡直接規定了迭代次數，而不是一直運行到b_grad和w_grad都為0(事實證明這樣做不太可行)

# store initial values for plotting，我們想要最終把數據描繪在圖上，因此存儲過程數據
b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

# iterations
for i in range(iteration):

    # get new b_grad and w_grad
    b_grad,w_grad=getGrad(b,w)

    # get the different learning rate for b and w
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    # 這一招叫做adagrad，之後會詳加解釋
    # update b and w with new learning rate
    b -= lr / np.sqrt(lr_b) * b_grad
    w -= lr / np.sqrt(lr_w) * w_grad

    #store parameters for plotting
    b_history.append(b)
    w_history.append(w)

    # output the b w b_grad w_grad
    # print("b: "+str(b)+"\t\t\t w: "+str(w)+"\n"+"b_grad: "+str(b_grad)+"\t\t w_grad: "+str(w_grad)+"\n")

# output the final function and its error
print("the function will be y_data="+str(b)+"+"+str(w)+"*x_data")
error=0.0
for i in range(10):
    print("error "+str(i)+" is: "+str(np.abs(y_data[i]-(b+w*x_data[i])))+" ")
    error+=np.abs(y_data[i]-(b+w*x_data[i]))
average_error=error/10
print("the average error is "+str(average_error))

# plot the figure
plt.contourf(x,y,Z,50,alpha=0.5,cmap=plt.get_cmap('jet'))
plt.plot([-188.4],[2.67],'x',ms=12,markeredgewidth=3,color='orange')
plt.plot(b_history,w_history,'o-',ms=3,lw=1.5,color='black')
plt.xlim(-200,-100)
plt.ylim(-5,5)
plt.xlabel(r'$b$',fontsize=16)
plt.ylabel(r'$w$',fontsize=16)
plt.show()
```

    the function will be y_data=-188.3668387495323+2.6692640713379903*x_data
    error 0 is: 73.84441736270833
    error 1 is: 67.4980970060185
    error 2 is: 68.15177664932844
    error 3 is: 28.8291759825683
    error 4 is: 13.113158627146447
    error 5 is: 148.63523696608252
    error 6 is: 96.43143001996799
    error 7 is: 94.21099446925288
    error 8 is: 140.84008808876973
    error 9 is: 161.7928115187101
    the average error is 89.33471866905532

<center><img src="https://img-blog.csdnimg.cn/20200123150828598.png" width="60%;"/></center>

**有了新的 learning rate 以後，從初始值到終點，我們在 100000 次 iteration 之內就可以順利地完成了**
