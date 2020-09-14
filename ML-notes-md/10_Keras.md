# Keras2.0

#### Why Keras

你可能會問，為什麼不學 TensorFlow 呢？明明 tensorflow 才是目前最流行的 machine learning 庫之一啊。其實，它並沒有那麼好用，tensorflow 和另外一個功能相近的 toolkit theano，它們是非常 flexible 的，你甚至可以把它想成是一個微分器，它完全可以做 deep learning 以外的事情，因為它的作用就是幫你算微分，拿到微分之後呢，你就可以去算 gradient descent 之類，而這麼 flexible 的 toolkit 學起來是有一定的難度的，你沒有辦法在半個小時之內精通這個 toolkit

但是另一個 toolkit——Keras，你是可以在數十分鐘內就熟悉並精通它的，然後用它來 implement 一個自己的 deep learning，Keras 其實是 tensorflow 和 theano 的 interface，所以用 Keras 就等於在用 tensorflow，只是有人幫你把操縱 tensorflow 這件事情先幫你寫好

所以 Keras 是比較容易去學習和使用的，並且它也有足夠的彈性，除非你自己想要做 deep learning 的研究，去設計一個自己的 network，否則多數你可以想到的 network，在 Keras 里都有現成的 function 可以拿來使用；因為它背後就是 tensorflow or theano，所以如果你想要精進自己的能力的話，你永遠可以去改 Keras 背後的 tensorflow 的 code，然後做更厲害的事情

而且，現在 Keras 已經成為了 Tensorflow 官方的 API，它像搭積木一樣簡單

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras.png" width="50%;" /></center>
接下來我們用手寫數字識別的demo來介紹一下"Hello world" of deep learning

#### prepare data

使用的 data 是 MNIST 的 Data：http://yann.lecun.com/exdb/mnist/

Keras 提供了自動下載 MNIST data 的 function：http://keras.io/datasets/

#### process

首先要先導入 keras 包：`from keras.models import Sequential`

##### step 1：define a set of function——neural network

先用`Sequential()`宣告建立一個 model

```python
model = Sequential()
```

然後開始疊一個 neural network：它有兩個 hidden layer，每個 hidden layer 都有 500 個 neuron

- 加一個**Fully connected**的 layer——用**Dense**來表示，當然你也可以加別的 layer，比如 convolution 的 layer

  之前我們說過，input layer 比較特殊，它並不是真正意義上的 layer，因為它沒有所謂的"neuron"，於是 Keras 在 model 裡面加的第一層 layer 會有一些特殊，要求同時輸入`input_dim`和`units`，分別代表第一層 hidden layer 輸入維數(也就是 input layer 的 dimension)和第一層 hidden layer 的 neuron 個數

  `input_dim=28*28`表示一個 28\*28=784 長度的 vector，代表 image；`units=500`表示該層 hidden layer 要有 500 個 neuron；`activation=‘sigmoid’`表示激活函數使用 sigmoid function

  ```python
  model.add(Dense(input_dim=28 * 28, units=500, activation='sigmoid'))
  ```

  加完 layer 之後，還需要設定該層 hidden layer 所使用的 activation function，這裡直接就用 sigmoid function

  在 Keras 里還可以選別的 activation function，比如 softplus、softsign、relu、tanh、hard_sigmoid、linear 等等，如果你要加上自己的 activation function，其實也蠻容易的，只要在 Keras 裡面找到寫 activation function 的地方，自己再加一個進去就好了

- 從第二層 hidden layer 開始，如果要在 model 里再加一個 layer，就用 model.add 增加一個 Dense 全連接層，包括`units`和`activation`參數

  這邊就不需要再 redefine `input_dim`是多少了，因為新增 layer 的 input 就等於前一個 layer 的 output，Keras 自己是知道這件事情的，所以你就直接告訴它說，新加的 layer 有 500 個 neuron 就好了

  這裡同樣把 activation function 設置為 sigmoid function

  ```python
  model.add(Dense(units=500, activation='sigmoid'))
  ```

- 最後，由於是分 10 個數字，所以 output 是 10 維，如果把 output layer 當做一個 Multi-class classifier 的話，那 activation function 就用 softmax(這樣可以讓 output 每一維的幾率之和為 1，表現得更像一個概率分布)，當然你也可以選擇別的

  ```python
  model.add(Dense(units=10, activation='softmax'))
  ```

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step1.png" width="60%;" /></center>
注：上圖中寫的是Keras1.0的語法，在筆記中給出的則是Keras2.0的語法，應當使用後者

##### Step 2：goodness of function——cross entropy

evaluate 一個 function 的好壞，你要做的事情是用 model.compile 去定義你的 loss function 是什麼

比如說你要用**cross entropy**的話，那你的 loss 參數就是**categorical_crossentropy**(Keras 里的寫法)

```python
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step2.png" width="60%;" /></center>
##### Step 3：pick the best function

###### Configuration

在 training 之前，你要先下一些**configuration**告訴它 training 的時候，你打算要怎麼做

你要定義的第一個東西是 optimizer，也就是說，你要用什麼樣的方式來找最好的 function，雖然 optimizer 後面可以接不同的方式，但是這些不同的方式，其實都是 gradient descent 類似的方法

有一些方法 machine 會自動地，empirically(根據經驗地)決定 learning rate 的值應該是多少，所以這些方法是不需要給它 learning rate 的，Keras 裡面有諸如：SGD(gradient descent)、RMSprop、Adagrad、Adadelta、Adam、Adamax、Nadam 之類的尋找最優參數的方法，它們都是 gradient descent 的方式

```python
model.compile(loss='categorical crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
```

###### Training

決定好怎麼做 gradient descent 之後，就是實際去做訓練了，去跑 gradient descent 找最優參數了

這裡使用的是`model.fit`方法，要給它 4 給 input(假設我們給了 10000 張 image 作 Training data)

- 第一個 input 是 Training data——`x_train`

  在這個 case 里，Training data 就是一張一張的 image，需要把它存放到 numpy array 裡面，這個 numpy array 是 two-dimension 的 matrix，每張 image 存為 numpy array 的一個行向量(它把 image 中 28\*28 個像素值拉成一個行向量)，總共有 10000 行，它的列數就是每張 image 的像素點個數，即 28\*28=784 列

- 第二個 input 是每一個 Training data 對應的 label——`y_train`

  在這個 case 里，就是標誌著這張 image 對應的是 0~9 的那一個數字，同樣也是 two-dimension 的 numpy array，每張 image 的 label 存為 numpy array 的一個行向量，用來表示 0~9 這 10 個數字中的某一個數，所以是 10 列，用的是 one-hot 編碼，10 個數字中對了對應 image 的那個數字為 1 之外其餘都是 0

- 第三個 input 是`batch_size`，告訴 Keras 我們的 batch 要有多大

  在這個 case 里，batch_size=100，表示我們要把 100 張隨機選擇的 image 放到一個 batch 裡面，然後把所有的 image 分成一個個不同的 batch，Keras 會自動幫你完成隨機選擇 image 的過程，不需要自己去 code

- 第四個 input 是`nb_epoch`，表示對所有 batch 的訓練要做多少次

  在這個 case 里，nb_epoch=20，表示要對所有的 batch 進行 20 遍 gradient descent 的訓練，每看到一個 batch 就 update 一次參賽，假設現在每一個 epoch 裡面有 100 個 batch，就對應著 update 100 次參數，20 個 epoch 就是 update 2000 次參數

注：如果 batch_size 設為 1，就是**Stochastic Gradient Descent**(隨機梯度下降法)，這個我們之前在討論 gradient descent 的時候有提到，就是每次拿到一個樣本點就 update 一次參數，而不是每次拿到一批樣本點的 error 之後才去 update 參數，因此 stochastic gradient descent 的好處是它的速度比較快，雖然每次 update 參數的方向是不穩定的，但是**天下武功，唯快不破**，在別人出一拳的時候，它就已經出了 100 拳了，所以它是比較強的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-step3.png" width="60%;" /></center>
#### Mini-batch

這裡有一個秘密，就是我們在做 deep learning 的 gradient descent 的時候，並不會真的去 minimize total loss，那我們做的是什麼呢？我們會把 Training data 分成一個一個的 batch，比如說你的 Training data 一共有 1w 張 image，每次 random 選 100 張 image 作為一個 batch(我的理解是，先將原來的 image 分布隨機打亂，然後再按順序每次挑出 batch_size 張 image 組成一個 batch，這樣才能保證所有的 data 都有被用到，且不同的 batch 里不會出現重復的 data)

- 像 gradient descent 一樣，先隨機 initialize network 的參數

- 選第一個 batch 出來，然後計算這個 batch 裡面的所有 element 的 total loss，$L'=l^1+l^{31}+...$，接下來根據$L'$去 update 參數，也就是計算$L'$對所有參數的偏微分，然後 update 參數

  注意：不是全部 data 的 total loss

- 再選擇第二個 batch，現在這個 batch 的 total loss 是$L''=l^2+l^{16}+...$，接下來計算$L''$對所有參數的偏微分，然後 update 參數

- 反復做這個 process，直到把所有的 batch 通通選過一次，所以假設你有 100 個 batch 的話，你就把這個參數 update 100 次，把所有 batch 看過一次，就叫做一個 epoch

- 重復 epoch 的過程，所以你在 train network 的時候，你會需要好幾十個 epoch，而不是只有一個 epoch

整個訓練的過程類似於 stochastic gradient descent，不是將所有數據讀完才開始做 gradient descent 的，而是拿到一部分數據就做一次 gradient descent

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mini-batch.png" width="50%;" /></center>
#### Batch size and Training Speed

##### batch size 太小會導致不穩定，速度上也沒有優勢

前面已經提到了，stochastic gradient descent 速度快，表現好，既然如此，為什麼我們還要用 Mini-batch 呢？這就涉及到了一些實際操作上的問題，讓我們必須去用 Mini-batch

舉例來說，我們現在有 50000 個 examples，如果我們把 batch size 設置為 1，就是 stochastic gradient descent，那在一個 epoch 裡面，就會 update 50000 次參數；如果我們把 batch size 設置為 10，在一個 epoch 裡面，就會 update 5000 次參數

看上去 stochastic gradient descent 的速度貌似是比較快的，它一個 epoch 更新參數的次數比 batch size 等於 10 的情況下要快了 10 倍，但是！我們好像忽略了一個問題，我們之前一直都是下意識地認為不同 batch size 的情況下運行一個 epoch 的時間應該是相等的，然後我們才去比較每個 epoch 所能夠 update 參數的次數，可是它們又怎麼可能會是相等的呢？

實際上，當你 batch size 設置不一樣的時候，一個 epoch 需要的時間是不一樣的，以 GTX 980 為例，下圖是對總數為 50000 筆的 Training data 設置不同的 batch size 時，每一個 epoch 所需要花費的時間

- case1：如果 batch size 設為 1，也就是 stochastic gradient descent，一個 epoch 要花費 166 秒，接近 3 分鐘

- case2：如果 batch size 設為 10，那一個 epoch 是 17 秒

也就是說，當 stochastic gradient descent 算了一個 epoch 的時候，batch size 為 10 的情況已經算了近 10 個 epoch 了；所以 case1 跑一個 epoch，做了 50000 次 update 參數的同時，case2 跑了十個 epoch，做了近 5000\*10=50000 次 update 參數；你會發現 batch size 設 1 和設 10，update 參數的次數幾乎是一樣的

如果不同 batch size 的情況，update 參數的次數幾乎是一樣的，你其實會想要選 batch size 更大的情況，就像在本例中，相較於 batch size=1，你會更傾向於選 batch size=10，因為 batch size=10 的時候，是會比較穩定的，因為**由更大的數據集計算的梯度能夠更好的代表樣本總體，從而更準確的朝向極值所在的方向**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/batch-size-speed.png" width="50%;" /></center>
我們之前把gradient descent換成stochastic gradient descent，是因為後者速度比較快，update次數比較多，可是現在如果你用stochastic gradient descent並沒有見得有多快，那你為什麼不選一個update次數差不多，又比較穩定的方法呢？

##### batch size 會受到 GPU 平行加速的限制，太大可能導致在 train 的時候卡住

上面例子的現象產生的原因是我們用了 GPU，用了平行運算，所以 batch size=10 的時候，這 10 個 example 其實是同時運算的，所以你在一個 batch 里算 10 個 example 的時間跟算 1 個 example 的時間幾乎可以是一樣的

那你可能會問，既然 batch size 越大，它會越穩定，而且還可以平行運算，那為什麼不把 batch size 變得超級大呢？這裡有兩個 claim(聲明)：

- 第一個 claim 就是，如果你把 batch size 開到很大，最終 GPU 會沒有辦法進行平行運算，它終究是有自己的極限的，也就是說它同時考慮 10 個 example 和 1 個 example 的時間是一樣的，但當它考慮 10000 個 example 的時候，時間就不可能還是跟一個 example 一樣，因為 batch size 考慮到**硬件限制**，是沒有辦法無窮盡地增長的

- 第二個 claim 是說，如果把 batch size 設的很大，在 train gradient descent 的時候，可能跑兩下你的 network 就卡住了，就陷到 saddle point 或者 local minima 裡面去了

  因為在 neural network 的 error surface 上面，如果你把 loss 的圖像可視化出來的話，它並不是一個 convex 的 optimization problem，不會像理想中那麼平滑，實際上它會有很多的坑坑洞洞

  如果你用的 batch size 很大，甚至是 Full batch，那你走過的路徑會是比較平滑連續的，可能這一條平滑的曲線在走向最低點的過程中就會在坑洞或是緩坡上卡住了；但是，如果你的 batch size 沒有那麼大，意味著你走的路線沒有那麼的平滑，有些步伐走的是**隨機性**的，路徑是會有一些曲折和波動的

  可能在你走的過程中，它的曲折和波動剛好使得你「繞過」了那些 saddle point 或是 local minima 的地方；或者當你陷入不是很深的 local minima 或者沒有遇到特別麻煩的 saddle point 的時候，它步伐的隨機性就可以幫你跳出這個 gradient 接近於 0 的區域，於是你更有可能真的走向 global minima 的地方

  而對於 Full batch 的情況，它的路徑是沒有隨機性的，是穩定朝著目標下降的，因此在這個時候去 train neural network 其實是有問題的，可能 update 兩三次參數就會卡住，所以 mini batch 是有必要的

  下面是我手畫的圖例和注釋：

    <center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/batch-size.jpg" width="70%;" /></center>

##### 不同 batch size 在梯度下降上的表現

如下圖，左邊是 full batch(拿全部的 Training data 做一個 batch)的梯度下降效果，可以看到每一次迭代成本函數都呈現下降趨勢，這是好的現象，說明我們 w 和 b 的設定一直再減少誤差， 這樣一直迭代下去我們就可以找到最優解；右邊是 mini batch 的梯度下降效果，可以看到它是上下波動的，成本函數的值有時高有時低，但總體還是呈現下降的趨勢， 這個也是正常的，因為我們每一次梯度下降都是在 min batch 上跑的而不是在整個數據集上， 數據的差異可能會導致這樣的波動(可能某段數據效果特別好，某段數據效果不好)，但沒關係，因為它整體是呈下降趨勢的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-gd1.png" width="50%;" /></center>
把下面的圖看做是梯度下降空間：藍色部分是full batch而紫色部分是mini batch，就像上面所說的mini batch不是每次迭代損失函數都會減少，所以看上去好像走了很多彎路，不過整體還是朝著最優解迭代的，而且由於mini batch一個epoch就走了5000步(5000次梯度下降)，而full batch一個epoch只有一步，所以雖然mini batch走了彎路但還是會快很多

而且，就像之前提到的那樣，mini batch 在 update 的過程中，步伐具有隨機性，因此紫色的路徑可以在一定程度上繞過或跳出 saddle point、local minima 這些 gradient 趨近於 0 的地方；而藍色的路徑因為缺乏隨機性，只能按照既定的方式朝著目標前進，很有可能就在中途被卡住，永遠也跳不出來了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/keras-gd2.png" width="40%;" /></center>
當然，就像之前討論的一樣，如果batch size太小，會造成速度不僅沒有加快反而會導致下降的曲線更加不穩定的情況產生

==**因此 batch size 既不能太大，因為它會受到硬件 GPU 平行加速的限制，導致 update 次數過於緩慢，並且由於缺少隨機性而很容易在梯度下降的過程中卡在 saddle point 或是 local minima 的地方(極端情況是 Full batch)；而且 batch size 也不能太小，因為它會導致速度優勢不明顯的情況下，梯度下降曲線過於不穩定，算法可能永遠也不會收斂(極端情況是 Stochastic gradient descent)**==

##### GPU 是如何平行加速的

整個 network，不管是 Forward pass 還是 Backward pass，都可以看做是一連串的矩陣運算的結果

那今天我們就可以比較 batch size 等於 1(stochastic gradient descent)和 10(mini batch)的差別

如下圖所示，stochastic gradient descent 就是對每一個 input x 進行單獨運算；而 mini batch，則是把同一個 batch 裡面的 input 全部集合起來，假設現在我們的 batch size 是 2，那 mini batch 每一次運算的 input 就是把黃色的 vector 和綠色的 vector 拼接起來變成一個 matrix，再把這個 matrix 乘上$w_1$，你就可以直接得到$z^1$和$z^2$

這兩件事在理論上運算量是一樣多的，但是在實際操作上，對 GPU 來說，在矩陣裡面相乘的每一個 element 都是可以平行運算的，所以圖中 stochastic gradient descent 運算的時間反而會變成下面 mini batch 使用 GPU 運算速度的兩倍，這就是為什麼我們要使用 mini batch 的原因

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/matrix-speed.png" width="50%;" /></center>
所以，如果你買了GPU，但是沒有使用mini batch的話，其實就不會有多少加速的效果

#### Save and Load Models

Keras 是可以幫你 save 和 load model 的，你可以把 train 好的 model 存起來，以後再用另外一個程式讀出來，它也可以幫你做 testing

那怎麼用 neural network 去 testing 呢？有兩種 case：

- case 1 是**evaluation**，比如今天我有一組 testing set，testing set 的答案也是已知的，那 Keras 就可以幫你算現在的正確率有多少，這個`model.evaluate`函數有兩個 input，就是 testing 的 image 和 testing 的 label

  ```python
  score = model.evaluate(x_test,y_test)
  print('Total loss on Testing Set:',score[0])
  print('Accuracy of Testing Set:',score[1])
  ```

- case 2 是**prediction**，這個時候`model.predict`函數的 input 只有 image data 而沒有任何的 label data，output 就直接是分類的結果

  ```python
  result = model.predict(x_test)
  ```

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/save-load-model.png" width="60%;" /></center>
#### Appendix：手寫數字識別完整代碼(Keras2.0)

##### code

```python
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

# categorical_crossentropy
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    # x_test=np.random.normal(x_test)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    # load training data and testing data
    (x_train, y_train), (x_test, y_test) = load_data()

    # define network structure
    model = Sequential()

    model.add(Dense(input_dim=28*28, units=500, activation='sigmoid'))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))

    # set configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # train model
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    # evaluate the model and output the accuracy
    result = model.evaluate(x_test, y_test)
    print('Test Acc:', result[1])

```

##### result

```python
Epoch 1/20
10000/10000 [==============================] - 2s 214us/step - loss: 1.1724 - acc: 0.6558
Epoch 2/20
10000/10000 [==============================] - 1s 146us/step - loss: 0.3847 - acc: 0.8964
Epoch 3/20
10000/10000 [==============================] - 1s 132us/step - loss: 0.2968 - acc: 0.9119
Epoch 4/20
10000/10000 [==============================] - 1s 146us/step - loss: 0.2535 - acc: 0.9268
Epoch 5/20
10000/10000 [==============================] - 2s 185us/step - loss: 0.2284 - acc: 0.9332
Epoch 6/20
10000/10000 [==============================] - 1s 141us/step - loss: 0.2080 - acc: 0.9369
Epoch 7/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.1829 - acc: 0.9455
Epoch 8/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.1617 - acc: 0.9520
Epoch 9/20
10000/10000 [==============================] - 1s 136us/step - loss: 0.1470 - acc: 0.9563
Epoch 10/20
10000/10000 [==============================] - 1s 133us/step - loss: 0.1340 - acc: 0.9607
Epoch 11/20
10000/10000 [==============================] - 1s 141us/step - loss: 0.1189 - acc: 0.9651
Epoch 12/20
10000/10000 [==============================] - 1s 143us/step - loss: 0.1056 - acc: 0.9696
Epoch 13/20
10000/10000 [==============================] - 1s 140us/step - loss: 0.0944 - acc: 0.9728
Epoch 14/20
10000/10000 [==============================] - 2s 172us/step - loss: 0.0808 - acc: 0.9773
Epoch 15/20
10000/10000 [==============================] - 1s 145us/step - loss: 0.0750 - acc: 0.9800
Epoch 16/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0643 - acc: 0.9826
Epoch 17/20
10000/10000 [==============================] - 1s 132us/step - loss: 0.0568 - acc: 0.9850
Epoch 18/20
10000/10000 [==============================] - 1s 135us/step - loss: 0.0510 - acc: 0.9873
Epoch 19/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0434 - acc: 0.9898
Epoch 20/20
10000/10000 [==============================] - 1s 134us/step - loss: 0.0398 - acc: 0.9906
10000/10000 [==============================] - 1s 79us/step
Test Acc: 0.9439
```

可以發現每次做完一個 epoch 的 update 後，手寫數字識別的準確率都有上升，最終訓練好的 model 識別準確率等於 94.39%

注：把 activation function 從 sigmoid 換成 relu 可以使識別準確率更高，這裡不再重復試驗
