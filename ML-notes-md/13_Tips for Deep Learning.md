# Tips for Deep Learning

> 本文會順帶解決 CNN 部分的兩個問題：
> 1、max pooling 架構中用到的 max 無法微分，那在 gradient descent 的時候該如何處理？
> 2、L1 的 Regression 到底是什麼東西
>
> 本文的主要思路：針對 training set 和 testing set 上的 performance 分別提出針對性的解決方法
> 1、在 training set 上準確率不高：
> new activation function：ReLU、Maxout
> adaptive learning rate：Adagrad、RMSProp、Momentum、Adam
> 2、在 testing set 上準確率不高：Early Stopping、Regularization or Dropout

### Recipe of Deep Learning

#### three step of deep learning

Recipe，配方、秘訣，這裡指的是做 deep learning 的流程應該是什麼樣子

我們都已經知道了 deep learning 的三個步驟

- define the function set(network structure)
- goodness of function(loss function -- cross entropy)
- pick the best function(gradient descent -- optimization)

做完這些事情以後，你會得到一個更好的 neural network，那接下來你要做什麼事情呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/recipe-dl.png" width="60%;" /></center>
#### Good Results on Training Data？

你要做的第一件事是，**提高 model 在 training set 上的正確率**

先檢查 training set 的 performance 其實是 deep learning 一個非常 unique 的地方，如果今天你用的是 k-nearest neighbor 或 decision tree 這類非 deep learning 的方法，做完以後你其實會不太想檢查 training set 的結果，因為在 training set 上的 performance 正確率就是 100，沒有什麼好檢查的

有人說 deep learning 的 model 里這麼多參數，感覺一臉很容易 overfitting 的樣子，但實際上這個 deep learning 的方法，它才不容易 overfitting，我們說的**overfitting 就是在 training set 上 performance 很好，但在 testing set 上 performance 沒有那麼好**；只有像 k nearest neighbor，decision tree 這類方法，它們在 training set 上正確率都是 100，這才是非常容易 overfitting 的，而對 deep learning 來說，overfitting 往往不會是你遇到的第一個問題

因為你在 training 的時候，deep learning 並不是像 k nearest neighbor 這種方法一樣，一訓練就可以得到非常好的正確率，它有可能在 training set 上根本沒有辦法給你一個好的正確率，所以，這個時候你要回頭去檢查在前面的 step 裡面要做什麼樣的修改，好讓你在 training set 上可以得到比較高的正確率

#### Good Results on Testing Data？

接下來你要做的事是，**提高 model 在 testing set 上的正確率**

假設現在你已經在 training set 上得到好的 performance 了，那接下來就把 model apply 到 testing set 上，我們最後真正關心的，是 testing set 上的 performance，假如得到的結果不好，這個情況下發生的才是 Overfitting，也就是在 training set 上得到好的結果，卻在 testing set 上得到不好的結果

那你要回過頭去做一些事情，試著解決 overfitting，但有時候你加了新的 technique，想要 overcome overfitting 這個 problem 的時候，其實反而會讓 training set 上的結果變壞；所以你在做完這一步的修改以後，要先回頭去檢查新的 model 在 training set 上的結果，如果這個結果變壞的話，你就要從頭對 network training 的 process 做一些調整，那如果你同時在 training set 還有 testing set 上都得到好結果的話，你就成功了，最後就可以把你的系統真正用在 application 上面了

#### Do not always blame overfitting

不要看到所有不好的 performance 就歸責於 overfitting

先看右邊 testing data 的圖，橫坐標是 model 做 gradient descent 所 update 的次數，縱坐標則是 error rate(越低說明 model 表現得越好)，黃線表示的是 20 層的 neural network，紅色表示 56 層的 neural network

你會發現，這個 56 層 network 的 error rate 比較高，它的 performance 比較差，而 20 層 network 的 performance 則是比較好的，有些人看到這個圖，就會馬上得到一個結論：56 層的 network 參數太多了，56 層果然沒有必要，這個是 overfitting。但是，真的是這樣子嗎？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/blame-over.png" width="60%;" /></center>
你在說結果是overfitting之前，有檢查過training set上的performance嗎？對neural network來說，在training set上得到的結果很可能會像左邊training error的圖，也就是說，20層的network本來就要比56層的network表現得更好，所以testing set得到的結果並不能說明56層的case就是發生了overfitting

在做 neural network training 的時候，有太多太多的問題可以讓你的 training set 表現的不好，比如說我們有 local minimum 的問題，有 saddle point 的問題，有 plateau 的問題...所以這個 56 層的 neural network，有可能在 train 的時候就卡在了一個 local minimum 的地方，於是得到了一個差的參數，但這並不是 overfitting，而是在 training 的時候就沒有 train 好

有人認為這個問題叫做 underfitting，但我的理解上，**underfitting**的本意應該是指這個 model 的 complexity 不足，這個 model 的參數不夠多，所以它的能力不足以解出這個問題；但這個 56 層的 network，它的參數是比 20 層的 network 要來得多的，所以它明明有能力比 20 層的 network 要做的更好，卻沒有得到理想的結果，這種情況不應該被稱為 underfitting，其實就只是沒有 train 好而已

#### conclusion

當你在 deep learning 的文獻上看到某種方法的時候，永遠要想一下，這個方法是要解決什麼樣的問題，因為在 deep learning 裡面，有兩個問題：

- 在 training set 上的 performance 不夠好
- 在 testing set 上的 performance 不夠好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/different-methods.png" width="60%;" /></center>
當只有一個方法propose(提出)的時候，它往往只針對這兩個問題的其中一個來做處理，舉例來說，deep learning有一個很潮的方法叫做dropout，那很多人就會說，哦，這麼潮的方法，所以今天只要看到performance不好，我就去用dropout；但是，其實只有在testing的結果不好的時候，才可以去apply dropout，如果你今天的問題只是training的結果不好，那你去apply dropout，只會越train越差而已

所以，你**必須要先想清楚現在的問題到底是什麼，然後再根據這個問題去找針對性的方法**，而不是病急亂投醫，甚至是盲目診斷

下面我們分別從 Training data 和 Testing data 兩個問題出發，來講述一些針對性優化的方法

### Good Results on Training Data？

這一部分主要講述如何在 Training data 上得到更好的 performance，分為兩個模塊，New activation function 和 Adaptive Learning Rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/different-method.png" width="60%;" /></center>
#### New activation function

這個部分主要講述的是關於 Recipe of Deep Learning 中 New activation function 的一些理論

##### activation function

如果你今天的 training 結果不好，很有可能是因為你的 network 架構設計得不好。舉例來說，可能你用的 activation function 是對 training 比較不利的，那你就嘗試著換一些新的 activation function，也許可以帶來比較好的結果

在 1980 年代，比較常用的 activation function 是 sigmoid function，如果現在我們使用 sigmoid function，你會發現 deeper 不一定 imply better，下圖是在 MNIST 手寫數字識別上的結果，當 layer 越來越多的時候，accuracy 一開始持平，後來就掉下去了，在 layer 是 9 層、10 層的時候，整個結果就崩潰了；但注意！9 層、10 層的情況並不能被認為是因為參數太多而導致 overfitting，實際上這張圖就只是 training set 的結果，你都不知道 testing 的情況，又哪來的 overfitting 之說呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-not-ok.png" width="60%;" /></center>
##### Vanishing Gradient Problem

上面這個問題的原因不是 overfitting，而是 Vanishing Gradient(梯度消失)，解釋如下：

當你把 network 疊得很深的時候，在靠近 input 的地方，這些參數的 gradient(即對最後 loss function 的微分)是比較小的；而在比較靠近 output 的地方，它對 loss 的微分值會是比較大的

因此當你設定同樣 learning rate 的時候，靠近 input 的地方，它參數的 update 是很慢的；而靠近 output 的地方，它參數的 update 是比較快的

所以在靠近 input 的地方，參數幾乎還是 random 的時候，output 就已經根據這些 random 的結果找到了一個 local minima，然後就 converge(收斂)了

這個時候你會發現，參數的 loss 下降的速度變得很慢，你就會覺得 gradient 已經接近於 0 了，於是把程序停掉了，由於這個 converge，是幾乎 base on random 的參數，所以 model 的參數並沒有被訓練充分，那在 training data 上得到的結果肯定是很差的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vanishing.png" width="60%;" /></center>
為什麼會有這個現象發生呢？如果你自己把Backpropagation的式子寫出來的話，就可以很輕易地發現用sigmoid function會導致這件事情的發生；但是，我們今天不看Backpropagation的式子，其實從直覺上來想你也可以瞭解這件事情發生的原因

某一個參數$w$對 total cost $l$的偏微分，即 gradient $\frac{\partial l}{\partial w}$，它直覺的意思是說，當我今天把這個參數做小小的變化的時候，它對這個 cost 的影響有多大；那我們就把第一個 layer 里的某一個參數$w$加上$\Delta w$，看看對 network 的 output 和 target 之間的 loss 有什麼樣的影響

$\Delta w$通過 sigmoid function 之後，得到 output 是會變小的，改變某一個參數的 weight，會對某個 neuron 的 output 值產生影響，但是這個影響是會隨著層數的遞增而衰減的，sigmoid function 的形狀如下所示，它會把負無窮大到正無窮大之間的值都硬壓到 0~1 之間，把較大的 input 壓縮成較小的 output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/sigmoid-less.png" width="35%;" /></center>
因此即使$\Delta w$值很大，但每經過一個sigmoid function就會被縮小一次，所以network越深，$\Delta w$被衰減的次數就越多，直到最後，它對output的影響就是比較小的，相應的也導致input對loss的影響會比較小，於是靠近input的那些weight對loss的gradient $\frac{\partial l}{\partial w}$遠小於靠近output的gradient

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vanish.png" width="60%;" /></center>
那怎麼解決這個問題呢？比較早年的做法是去train RBM，它的精神就是，先把第一個layer train好，再去train第二個，然後再第三個...所以最後你在做Backpropagation的時候，儘管第一個layer幾乎沒有被train到，但一開始在做pre-train的時候就已經把它給train好了，這樣RBM就可以在一定程度上解決問題

但其實改一下 activation function 可能就可以 handle 這個問題了

##### ReLU

###### introduction

現在比較常用的 activation function 叫做 Rectified Linear Unit(整流線性單元函數，又稱修正線性單元)，它的縮寫是 ReLU，該函數形狀如下圖所示，z 為 input，a 為 output，如果 input>0 則 output = input，如果 input<0 則 output = 0

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/ReLU1.png" width="60%;" /></center>
選擇ReLU的理由如下：

- 跟 sigmoid function 比起來，ReLU 的運算快很多
- ReLU 的想法結合了生物上的觀察( Pengel 的 paper )
- 無窮多 bias 不同的 sigmoid function 疊加的結果會變成 ReLU
- ReLU 可以處理 Vanishing gradient 的問題( the most important thing )

###### handle Vanishing gradient problem

下圖是 ReLU 的 neural network，以 ReLU 作為 activation function 的 neuron，它的 output 要麼等於 0，要麼等於 input

當 output=input 的時候，這個 activation function 就是 linear 的；而 output=0 的 neuron 對整個 network 是沒有任何作用的，因此可以把它們從 network 中拿掉

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu1.png" width="60%;" /></center>
拿掉所有output為0的neuron後如下圖所示，此時整個network就變成了一個瘦長的**linear** network，linear的好處是，output=input，不會像sigmoid function一樣使input產生的影響逐層遞減

Q：這裡就會有一個問題，我們之所以使用 deep learning，就是因為想要一個 non-linear、比較複雜的 function，而使用 ReLU 不就會讓它變成一個 linear function 嗎？這樣得到的 function 不是會變得很弱嗎？

A：其實，使用 ReLU 之後的 network 整體來說還是 non-linear 的，如果你對 input 做小小的改變，不改變 neuron 的 operation region 的話，那 network 就是一個 linear function；但是，如果你對 input 做比較大的改變，導致 neuron 的 operation region 被改變的話，比如從 output=0 轉變到了 output=input，network 整體上就變成了 non-linear function

注：這裡的 region 是指 input z<0 和 input z>0 的兩個範圍

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu2.png" width="60%;" /></center>
Q：還有另外一個問題，我們對loss function做gradient descent，要求neural network是可以做微分的，但ReLU是一個分段函數，它是不能微分的(至少在z=0這個點是不可微的)，那該怎麼辦呢？

A：在實際操作上，當 region 的範圍處於 z>0 時，微分值 gradient 就是 1；當 region 的範圍處於 z<0 時，微分值 gradient 就是 0；當 z 為 0 時，就不要管它，相當於把它從 network 裡面拿掉

###### ReLU-variant

其實 ReLU 還存在一定的問題，比如當 input<0 的時候，output=0，此時微分值 gradient 也為 0，你就沒有辦法去 update 參數了，所以我們應該讓 input<0 的時候，微分後還能有一點點的值，比如令$a=0.01z$，這個東西就叫做**Leaky ReLU**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/relu-variant.png" width="60%;" /></center>
既然a可以等於0.01z，那這個z的系數可不可以是0.07、0.08之類呢？所以就有人提出了**Parametric ReLU**，也就是令$a=\alpha \cdot z$，其中$\alpha$並不是固定的值，而是network的一個參數，它可以通過training data學出來，甚至每個neuron都可以有不同的$\alpha$值

這個時候又有人想，為什麼一定要是 ReLU 這樣子呢，activation function 可不可以有別的樣子呢？所以後來有了一個更進階的想法，叫做**Maxout network**

##### Maxout

###### introduction

Maxout 的想法是，讓 network 自動去學習它的 activation function，那 Maxout network 就可以自動學出 ReLU，也可以學出其他的 activation function，這一切都是由 training data 來決定的

假設現在有 input $x_1,x_2$，它們乘上幾組不同的 weight 分別得到 5,7,-1,1，這些值本來是不同 neuron 的 input，它們要通過 activation function 變為 neuron 的 output；但在 Maxout network 里，我們事先決定好將某幾個「neuron」的 input 分為一個 group，比如 5,7 分為一個 group，然後在這個 group 里選取一個最大值 7 作為 output

這個過程就好像在一個 layer 上做 Max Pooling 一樣，它和原來的 network 不同之處在於，它把原來幾個「neuron」的 input 按一定規則組成了一個 group，然後並沒有使它們通過 activation function，而是選取其中的最大值當做這幾個「neuron」的 output

當然，實際上原來的」neuron「早就已經不存在了，這幾個被合併的「neuron」應當被看做是一個新的 neuron，這個新的 neuron 的 input 是原來幾個「neuron」的 input 組成的 vector，output 則取 input 的最大值，而並非由 activation function 產生

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout1.png" width="60%;" /></center>
在實際操作上，幾個element被分為一個group這件事情是由你自己決定的，它就是network structure里一個需要被調的參數，不一定要跟上圖一樣兩個分為一組

###### Maxout -> RELU

Maxout 是如何模仿出 ReLU 這個 activation function 的呢？

下圖左上角是一個 ReLU 的 neuron，它的 input x 會乘上 neuron 的 weight w，再加上 bias b，然後通過 activation function-ReLU，得到 output a

- neuron 的 input 為$z=wx+b$，為下圖左下角紫線
- neuron 的 output 為$a=z\ (z>0);\ a=0\ (z<0)$，為下圖左下角綠線

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout2.png" width="60%;" /></center>
如果我們使用的是上圖右上角所示的Maxout network，假設$z_1$的參數w和b與ReLU的參數一致，而$z_2$的參數w和b全部設為0，然後做Max Pooling，選取$z_1,z_2$較大值作為a

- neuron 的 input 為$\begin{bmatrix}z_1 \ z_2 \end{bmatrix}$
  - $z_1=wx+b$，為上圖右下角紫線
  - $z_2=0$，為上圖右下角紅線
- neuron 的 output 為$\max{\begin{bmatrix}z_1 \ z_2 \end{bmatrix}}$，為上圖右下角綠線

你會發現，此時 ReLU 和 Maxout 所得到的 output 是一模一樣的，它們是相同的 activation function

###### Maxout -> More than ReLU

除了 ReLU，Maxout 還可以實現更多不同的 activation function

比如$z_2$的參數 w 和 b 不是 0，而是$w',b'$，此時

- neuron 的 input 為$\begin{bmatrix}z_1 \ z_2 \end{bmatrix}$
  - $z_1=wx+b$，為下圖右下角紫線
  - $z_2=w'x+b'$，為下圖右下角紅線
- neuron 的 output 為$\max{\begin{bmatrix}z_1 \ z_2 \end{bmatrix}}$，為下圖右下角綠線

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout3.png" width="60%;" /></center>
這個時候你得到的activation function的形狀(綠線形狀)，是由network的參數$w,b,w',b'$決定的，因此它是一個**Learnable Activation Function**，具體的形狀可以根據training data去generate出來

###### property

Maxout 可以實現任何 piecewise linear convex activation function(分段線性凸激活函數)，其中這個 activation function 被分為多少段，取決於你把多少個 element z 放到一個 group 里，下圖分別是 2 個 element 一組和 3 個 element 一組的 activation function 的不同形狀

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout4.png" width="60%;" /></center>
###### How to train Maxout

接下來我們要面對的是，怎麼去 train 一個 Maxout network，如何解決 Max 不能微分的問題

假設在下面的 Maxout network 中，紅框圈起來的部分為每個 neuron 的 output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout-train.png" width="60%;" /></center>
其實Max operation就是linear的operation，只是它僅接在前面這個group里的某一個element上，因此我們可以把那些並沒有被Max連接到的element通通拿掉，從而得到一個比較細長的linear network

實際上我們真正訓練的並不是一個含有 max 函數的 network，而是一個化簡後如下圖所示的 linear network；當我們還沒有真正開始訓練模型的時候，此時這個 network 含有 max 函數無法微分，但是只要真的丟進去了一筆 data，network 就會馬上根據這筆 data 確定具體的形狀，此時 max 函數的問題已經被實際數據給解決了，所以我們完全可以根據這筆 training data 使用 Backpropagation 的方法去訓練被 network 留下來的參數

所以我們擔心的 max 函數無法微分，它只是理論上的問題；**在具體的實踐上，我們完全可以先根據 data 把 max 函數轉化為某個具體的函數，再對這個轉化後的 thiner linear network 進行微分**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/maxout-train2.png" width="60%;" /></center>
這個時候你也許會有一個問題，如果按照上面的做法，那豈不是只會train留在network裡面的那些參數，剩下的參數該怎麼辦？那些被拿掉的直線(weight)豈不是永遠也train不到了嗎？

其實這也只是個理論上的問題，在實際操作上，我們之前已經提到過，每個 linear network 的 structure 都是由 input 的那一筆 data 來決定的，當你 input 不同 data 的時候，得到的 network structure 是不同的，留在 network 裡面的參數也是不同的，**由於我們有很多很多筆 training data，所以 network 的 structure 在訓練中不斷地變換，實際上最後每一個 weight 參數都會被 train 到**

所以，我們回到 Max Pooling 的問題上來，由於 Max Pooling 跟 Maxout 是一模一樣的 operation，既然如何訓練 Maxout 的問題可以被解決，那訓練 Max Pooling 又有什麼困難呢？

**Max Pooling 有關 max 函數的微分問題採用跟 Maxout 一樣的方案即可解決**，至此我們已經解決了 CNN 部分的第一個問題

#### Adaptive learning rate

這個部分主要講述的是關於 Recipe of Deep Learning 中 Adaptive learning rate 的一些理論

##### Review - Adagrad

我們之前已經瞭解過 Adagrad 的做法，讓每一個 parameter 都要有不同的 learning rate

Adagrad 的精神是，假設我們考慮兩個參數$w_1,w_2$，如果在$w_1$這個方向上，平常的 gradient 都比較小，那它是比較平坦的，於是就給它比較大的 learning rate；反過來說，在$w_2$這個方向上，平常 gradient 都比較大，那它是比較陡峭的，於是給它比較小的 learning rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/review-adagrad.png" width="60%;" /></center>
但我們實際面對的問題，很有可能遠比Adagrad所能解決的問題要來的複雜，我們之前做Linear Regression的時候，我們做optimization的對象，也就是loss function，它是convex的形狀；但實際上我們在做deep learning的時候，這個loss function可以是任何形狀

##### RMSProp

###### learning rate

loss function 可以是任何形狀，對 convex loss function 來說，在每個方向上它會一直保持平坦或陡峭的狀態，所以你只需要針對平坦的情況設置較大的 learning rate，對陡峭的情況設置較小的 learning rate 即可

但是在下圖所示的情況中，即使是在同一個方向上(如 w1 方向)，loss function 也有可能一會兒平坦一會兒陡峭，所以你要隨時根據 gradient 的大小來快速地調整 learning rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rmsprop1.png" width="60%;" /></center>
所以真正要處理deep learning的問題，用Adagrad可能是不夠的，你需要更dynamic的調整learning rate的方法，所以產生了Adagrad的進階版——**RMSProp**

RMSprop 還是一個蠻神奇的方法，因為它並不是在 paper 里提出來的，而是 Hinton 在 mooc 的 course 裡面提出來的一個方法，所以需要 cite(引用)的時候，要去 cite Hinton 的課程鏈接

###### how to do RMSProp

RMSProp 的做法如下：

我們的 learning rate 依舊設置為一個固定的值 $\eta$ 除掉一個變化的值 $\sigma$，這個$\sigma$等於上一個$\sigma$和當前梯度$g$的加權方均根（特別的是，在第一個時間點，$\sigma^0$就是第一個算出來的 gradient 值$g^0$），即：

$$
w^{t+1}=w^t-\frac{\eta}{\sigma^t}g^t \\
\sigma^t=\sqrt{\alpha(\sigma^{t-1})^2+(1-\alpha)(g^t)^2}
$$

這裡的$\alpha$值是可以自由調整的，RMSProp 跟 Adagrad 不同之處在於，Adagrad 的分母是對過程中所有的 gradient 取平方和開根號，也就是說 Adagrad 考慮的是整個過程平均的 gradient 信息；而 RMSProp 雖然也是對所有的 gradient 進行平方和開根號，但是它**用一個$\alpha$來調整對不同 gradient 的使用程度**，比如你把 α 的值設的小一點，意思就是你更傾向於相信新的 gradient 所告訴你的 error surface 的平滑或陡峭程度，而比較無視於舊的 gradient 所提供給你的 information

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rmsprop2.png" width="60%;" /></center>
所以當你做RMSProp的時候，一樣是在算gradient的root mean square，但是你可以給現在已經看到的gradient比較大的weight，給過去看到的gradient比較小的weight，來調整對gradient信息的使用程度

##### Momentum

###### optimization - local minima？

除了 learning rate 的問題以外，在做 deep learning 的時候，也會出現卡在 local minimum、saddle point 或是 plateau 的地方，很多人都會擔心，deep learning 這麼複雜的 model，可能非常容易就會被卡住了

但其實 Yann LeCun 在 07 年的時候，就提出了一個蠻特別的說法，他說你不要太擔心 local minima 的問題，因為一旦出現 local minima，它就必須在每一個 dimension 都是下圖中這種山谷的低谷形狀，假設山谷的低谷出現的概率為 p，由於我們的 network 有非常非常多的參數，這裡假設有 1000 個參數，每一個參數都要位於山谷的低谷之處，這件事發生的概率為$p^{1000}$，當你的 network 越複雜，參數越多，這件事發生的概率就越低

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/optimal1.png" width="60%;" /></center>
所以在一個很大的neural network裡面，其實並沒有那麼多的local minima，搞不好它看起來其實是很平滑的，所以當你走到一個你覺得是local minima的地方被卡住了，那它八成就是global minima，或者是很接近global minima的地方

###### where is Momentum from

有一個 heuristic(啓發性)的方法可以稍微處理一下上面所說的「卡住」的問題，它的靈感來自於真實世界

假設在有一個球從左上角滾下來，它會滾到 plateau 的地方、local minima 的地方，但是由於慣性它還會繼續往前走一段路程，假設前面的坡沒有很陡，這個球就很有可能翻過山坡，走到比 local minima 還要好的地方

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/physical.png" width="60%;" /></center>
所以我們要做的，就是把**慣性**塞到gradient descent裡面，這件事情就叫做**Momentum**

###### how to do Momentum

當我們在 gradient descent 里加上 Momentum 的時候，每一次 update 的方向，不再只考慮 gradient 的方向，還要考慮上一次 update 的方向，那這裡我們就用一個變量$v$去記錄前一個時間點 update 的方向

隨機選一個初始值$\theta^0$，初始化$v^0=0$，接下來計算$\theta^0$處的 gradient，然後我們要移動的方向是由前一個時間點的移動方向$v^0$和 gradient 的反方向$\nabla L(\theta^0)$來決定的，即

$$
v^1=\lambda v^0-\eta \nabla L(\theta^0)
$$

注：這裡的$\lambda$也是一個手動調整的參數，它表示慣性對前進方向的影響有多大

接下來我們第二個時間點要走的方向$v^2$，它是由第一個時間點移動的方向$v^1$和 gradient 的反方向$\nabla L(\theta^1)$共同決定的；$\lambda v$是圖中的綠色虛線，它代表由於上一次的慣性想要繼續走的方向；$\eta \nabla L(\theta)$是圖中的紅色虛線，它代表這次 gradient 告訴你所要移動的方向；它們的矢量和就是這一次真實移動的方向，為藍色實線

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum1.png" width="60%;" /></center>
gradient告訴我們走紅色虛線的方向，慣性告訴我們走綠色虛線的方向，合起來就是走藍色的方向

我們還可以用另一種方法來理解 Momentum 這件事，其實你在每一個時間點移動的步伐$v^i$，包括大小和方向，就是過去所有 gradient 的加權和

具體推導如下圖所示，第一個時間點移動的步伐$v^1$是$\theta^0$處的 gradient 加權，第二個時間點移動的步伐$v^2$是$\theta^0$和$\theta^1$處的 gradient 加權和...以此類推；由於$\lambda$的值小於 1，因此該加權意味著越是之前的 gradient，它的權重就越小，也就是說，你更在意的是現在的 gradient，但是過去的所有 gradient 也要對你現在 update 的方向有一定程度的影響力，這就是 Momentum

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum2.png" width="60%;" /></center>
如果你對數學公式不太喜歡的話，那我們就從直覺上來看一下加入Momentum之後是怎麼運作的

在加入 Momentum 以後，每一次移動的方向，就是 negative 的 gradient 加上 Momentum 建議我們要走的方向，Momentum 其實就是上一個時間點的 movement

下圖中，紅色實線是 gradient 建議我們走的方向，直觀上看就是根據坡度要走的方向；綠色虛線是 Momentum 建議我們走的方向，實際上就是上一次移動的方向；藍色實線則是最終真正走的方向

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/momentum3.png" width="60%;" /></center>
如果我們今天走到local minimum的地方，此時gradient是0，紅色箭頭沒有指向，它就會告訴你就停在這裡吧，但是Momentum也就是綠色箭頭，它指向右側就是告訴你之前是要走向右邊的，所以你仍然應該要繼續往右走，所以最後你參數update的方向仍然會繼續向右；你甚至可以期待Momentum比較強，慣性的力量可以支撐著你走出這個谷底，去到loss更低的地方

##### Adam

其實**RMSProp 加上 Momentum，就可以得到 Adam**

根據下面的 paper 來快速描述一下 Adam 的 algorithm：

- 先初始化$m_0=0$，$m_0$就是 Momentum 中，前一個時間點的 movement

  再初始化$v_0=0$，$v_0$就是 RMSProp 里計算 gradient 的 root mean square 的$\sigma$

  最後初始化$t=0$，t 用來表示時間點

- 先算出 gradient $g_t$

  $$
  g_t=\nabla _{\theta}f_t(\theta_{t-1})
  $$

- 再根據過去要走的方向$m_{t-1}$和 gradient $g_t$，算出現在要走的方向 $m_t$——Momentum

  $$
  m_t=\beta_1 m_{t-1}+(1-\beta_1) g_t
  $$

- 然後根據前一個時間點的$v_{t-1}$和 gradient $g_t$的平方，算一下放在分母的$v_t$——RMSProp

  $$
  v_t=\beta_2 v_{t-1}+(1-\beta_2) g_t^2
  $$

- 接下來做了一個原來 RMSProp 和 Momentum 里沒有的東西，就是 bias correction，它使$m_t$和$v_t$都除上一個值，這個值本來比較小，後來會越來越接近於 1 (原理詳見 paper)

  $$
  \hat{m}_t=\frac{m_t}{1-\beta_1^t} \\ \hat{v}_t=\frac{v_t}{1-\beta_2^t}
  $$

- 最後做 update，把 Momentum 建議你的方向$\hat{m_t}$乘上 learning rate $\alpha$，再除掉 RMSProp normalize 後建議的 learning rate 分母，然後得到 update 的方向
  $$
  \theta_t=\theta_{t-1}-\frac{\alpha \cdot \hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
  $$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adam.png" width="80%;" /></center>
### Good Results on Testing Data？

這一部分主要講述如何在 Testing data 上得到更好的 performance，分為三個模塊，Early Stopping、Regularization 和 Dropout

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/result-test.png" width="60%;" /></center>
值得注意的是，Early Stopping和Regularization是很typical的做法，它們不是特別為deep learning所設計的；而Dropout是一個蠻有deep learning特色的做法

#### Early Stopping

假設你今天的 learning rate 調的比較好，那隨著訓練的進行，total loss 通常會越來越小，但是 Training set 和 Testing set 的情況並不是完全一樣的，很有可能當你在 Training set 上的 loss 逐漸減小的時候，在 Testing set 上的 loss 反而上升了

所以，理想上假如你知道 testing data 上的 loss 變化情況，你會在 testing set 的 loss 最小的時候停下來，而不是在 training set 的 loss 最小的時候停下來；但 testing set 實際上是未知的東西，所以我們需要用 validation set 來替代它去做這件事情

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/early-stop.png" width="60%;" /></center>
注：很多時候，我們所講的「testing set」並不是指代那個未知的數據集，而是一些已知的被你拿來做測試之用的數據集，比如kaggle上的public set，或者是你自己切出來的validation set

#### Regularization

regularization 就是在原來的 loss function 上額外增加幾個 term，比如我們要 minimize 的 loss function 原先應該是 square error 或 cross entropy，那在做 Regularization 的時候，就在後面加一個 Regularization 的 term

##### L2 regularization

regularization term 可以是參數的 L2 norm(L2 正規化)，所謂的 L2 norm，就是把 model 參數集$\theta$里的每一個參數都取平方然後求和，這件事被稱作 L2 regularization，即

$$
L2 \ regularization:||\theta||_2=(w_1)^2+(w_2)^2+...
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization1.png" width="60%;" /></center>
通常我們在做regularization的時候，新加的term里是不會考慮bias這一項的，因為加regularization的目的是為了讓我們的function更平滑，而bias通常是跟function的平滑程度沒有關係的

你會發現我們新加的 regularization term $\lambda \frac{1}{2}||\theta||_2$里有一個$\frac{1}{2}$，由於我們是要對 loss function 求微分的，而新加的 regularization term 是參數$w_i$的平方和，對平方求微分會多出來一個系數 2，我們的$\frac{1}{2}$就是用來和這個 2 相消的

L2 regularization 具體工作流程如下：

- 我們加上 regularization term 之後得到了一個新的 loss function：$L'(\theta)=L(\theta)+\lambda \frac{1}{2}||\theta||_2$
- 將這個 loss function 對參數$w_i$求微分：$\frac{\partial L'}{\partial w_i}=\frac{\partial L}{\partial w_i}+\lambda w_i$
- 然後 update 參數$w_i$：$w_i^{t+1}=w_i^t-\eta \frac{\partial L'}{\partial w_i}=w_i^t-\eta(\frac{\partial L}{\partial w_i}+\lambda w_i^t)=(1-\eta \lambda)w_i^t-\eta \frac{\partial L}{\partial w_i}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization2.png" width="60%;" /></center>
如果把這個推導出來的式子和原式作比較，你會發現參數$w_i$在每次update之前，都會乘上一個$(1-\eta \lambda)$，而$\eta$和$\lambda$通常會被設為一個很小的值，因此$(1-\eta \lambda)$通常是一個接近於1的值，比如0.99,；也就是說，regularization做的事情是，每次update參數$w_i$之前，不分青紅皂白就先對原來的$w_i$乘個0.99，這意味著，隨著update次數增加，參數$w_i$會越來越接近於0

Q：你可能會問，要是所有的參數都越來越靠近 0，那最後豈不是$w_i$通通變成 0，得到的 network 還有什麼用？

A：其實不會出現最後所有參數都變為 0 的情況，因為通過微分得到的$\eta \frac{\partial L}{\partial w_i}$這一項是會和前面$(1-\eta \lambda)w_i^t$這一項最後取得平衡的

使用 L2 regularization 可以讓 weight 每次都變得更小一點，這就叫做**Weight Decay**(權重衰減)

##### L1 regularization

除了 L2 regularization 中使用平方項作為 new term 之外，還可以使用 L1 regularization，把平方項換成每一個參數的絕對值，即

$$
||\theta||_1=|w_1|+|w_2|+...
$$

Q：你的第一個問題可能會是，絕對值不能微分啊，該怎麼處理呢？

A：實際上絕對值就是一個 V 字形的函數，在 V 的左邊微分值是-1，在 V 的右邊微分值是 1，只有在 0 的地方是不能微分的，那真的走到 0 的時候就胡亂給它一個值，比如 0，就 ok 了

如果 w 是正的，那微分出來就是+1，如果 w 是負的，那微分出來就是-1，所以這邊寫了一個 w 的 sign function，它的意思是說，如果 w 是正數的話，這個 function output 就是+1，w 是負數的話，這個 function output 就是-1

L1 regularization 的工作流程如下：

- 我們加上 regularization term 之後得到了一個新的 loss function：$L'(\theta)=L(\theta)+\lambda \frac{1}{2}||\theta||_1$
- 將這個 loss function 對參數$w_i$求微分：$\frac{\partial L'}{\partial w_i}=\frac{\partial L}{\partial w_i}+\lambda \ sgn(w_i)$
- 然後 update 參數$w_i$：$w_i^{t+1}=w_i^t-\eta \frac{\partial L'}{\partial w_i}=w_i^t-\eta(\frac{\partial L}{\partial w_i}+\lambda \ sgn(w_i^t))=w_i^t-\eta \frac{\partial L}{\partial w_i}-\eta \lambda \ sgn(w_i^t)$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization3.png" width="60%;" /></center>
這個式子告訴我們，每次update的時候，不管三七二十一都要減去一個$\eta \lambda \ sgn(w_i^t)$，如果w是正的，sgn是+1，就會變成減一個positive的值讓你的參數變小；如果w是負的，sgn是-1，就會變成加一個值讓你的參數變大；總之就是讓它們的絕對值減小至接近於0

##### L1 V.s. L2

我們來對比一下 L1 和 L2 的 update 過程：

$$
L1: w_i^{t+1}=w_i^t-\eta \frac{\partial L}{\partial w_i}-\eta \lambda \ sgn(w_i^t)\\
L2: w_i^{t+1}=(1-\eta \lambda)w_i^t-\eta \frac{\partial L}{\partial w_i}
$$

L1 和 L2，雖然它們同樣是讓參數的絕對值變小，但它們做的事情其實略有不同：

- L1 使參數絕對值變小的方式是每次 update**減掉一個固定的值**
- L2 使參數絕對值變小的方式是每次 update**乘上一個小於 1 的固定值**

因此，當參數 w 的絕對值比較大的時候，L2 會讓 w 下降得更快，而 L1 每次 update 只讓 w 減去一個固定的值，train 完以後可能還會有很多比較大的參數；當參數 w 的絕對值比較小的時候，L2 的下降速度就會變得很慢，train 出來的參數平均都是比較小的，而 L1 每次下降一個固定的 value，train 出來的參數是比較 sparse 的，這些參數有很多是接近 0 的值，也會有很大的值

在之前所講的 CNN 的 task 里，用 L1 做出來的效果是比較合適的，是比較 sparse 的

##### Weight Decay

之前提到了 Weight Decay，那實際上我們在人腦裡面也會做 Weight Decay

下圖分別描述了，剛出生的時候，嬰兒的神經是比較稀疏的；6 歲的時候，就會有很多很多的神經；但是到 14 歲的時候，神經間的連接又減少了，所以 neural network 也會跟我們人有一些很類似的事情，如果有一些 weight 你都沒有去 update 它，那它每次都會越來越小，最後就接近 0 然後不見了

這跟人腦的運作，是有異曲同工之妙

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization4.png" width="60%;" /></center>
#### some tips

ps：在 deep learning 裡面，regularization 雖然有幫助，但它的重要性往往沒有 SVM 這類方法來得高，因為我們在做 neural network 的時候，通常都是從一個很小的、接近於 0 的值開始初始參數的，而做 update 的時候，通常都是讓參數離 0 越來越遠，但是 regularization 要達到的目的，就是希望我們的參數不要離 0 太遠

如果你做的是 Early Stopping，它會減少 update 的次數，其實也會避免你的參數離 0 太遠，這跟 regularization 做的事情是很接近的

所以在 neural network 裡面，regularization 的作用並沒有 SVM 來的重要，SVM 其實是 explicitly 把 regularization 這件事情寫在了它的 objective function(目標函數)裡面，SVM 是要去解一個 convex optimization problem，因此它解的時候不一定會有 iteration 的過程，它不會有 Early Stopping 這件事，而是一步就可以走到那個最好的結果了，所以你沒有辦法用 Early Stopping 防止它離目標太遠，你必須要把 regularization explicitly 加到你的 loss function 裡面去

#### Dropout

這裡先講 dropout 是怎麼做的，然後再來解釋為什麼這樣做

##### How to do Dropout

Dropout 是怎麼做的呢？

###### Training

在 training 的時候，每次 update 參數之前，我們對每一個 neuron(也包括 input layer 的「neuron」)做 sampling(抽樣) ，每個 neuron 都有 p%的幾率會被丟掉，如果某個 neuron 被丟掉的話，跟它相連的 weight 也都要被丟掉

實際上就是每次 update 參數之前都通過抽樣只保留 network 中的一部分 neuron 來做訓練

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout1.png" width="60%;" /></center>
做完sampling以後，network structure就會變得比較細長了，然後你再去train這個細長的network

注：每次 update 參數之前都要做一遍 sampling，所以每次 update 參數的時候，拿來 training 的 network structure 都是不一樣的；你可能會覺得這個方法跟前面提到的 Maxout 會有一點像，但實際上，Maxout 是每一筆 data 對應的 network structure 不同，而 Dropout 是每一次 update 的 network structure 都是不同的(每一個 minibatch 對應著一次 update，而一個 minibatch 里含有很多筆 data)

當你在 training 的時候使用 dropout，得到的 performance 其實是會變差的，因為某些 neuron 在 training 的時候莫名其妙就會消失不見，但這並不是問題，因為：

==**Dropout 真正要做的事情，就是要讓你在 training set 上的結果變差，但是在 testing set 上的結果是變好的**==

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout2.png" width="60%;" /></center>
所以如果你今天遇到的問題是在training set上得到的performance不夠好，你再加dropout，就只會越做越差；這告訴我們，不同的problem需要用不同的方法去解決，而不是胡亂使用，dropout就是針對testing set的方法，當然不能夠拿來解決training set上的問題啦！

###### Testing

在使用 dropout 方法做 testing 的時候要注意兩件事情：

- testing 的時候不做 dropout，所有的 neuron 都要被用到
- 假設在 training 的時候，dropout rate 是 p%，從 training data 中被 learn 出來的所有 weight 都要乘上(1-p%)才能被當做 testing 的 weight 使用

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout3.png" width="60%;" /></center>
##### Why Dropout？

###### 為什麼 dropout 會有用？

直接的想法是這樣子：

在 training 的時候，會丟掉一些 neuron，就好像是你要練輕功的時候，會在腳上綁一些重物；然後，你在實際戰鬥的時候，就是實際 testing 的時候，是沒有 dropout 的，就相當於把重物拿下來，所以你就會變得很強

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout4.png" width="60%;" /></center>
另一個直覺的理由是這樣，neural network裡面的每一個neuron就是一個學生，那大家被連接在一起就是大家聽到說要組隊做final project，那在一個團隊裡總是有人會拖後腿，就是他會dropout，所以假設你覺得自己的隊友會dropout，這個時候你就會想要好好做，然後去carry這個隊友，這就是training的過程

那實際在 testing 的時候，其實大家都有好好做，沒有人需要被 carry，由於每個人都比一般情況下更努力，所以得到的結果會是更好的，這也就是 testing 的時候不做 dropout 的原因

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout5.png" width="60%;" /></center>
###### 為什麼training和testing使用的weight是不一樣的呢？

直覺的解釋是這樣的：

假設現在的 dropout rate 是 50%，那在 training 的時候，你總是期望每次 update 之前會丟掉一半的 neuron，就像下圖左側所示，在這種情況下你 learn 好了一組 weight 參數，然後拿去 testing

但是在 testing 的時候是沒有 dropout 的，所以如果 testing 使用的是和 training 同一組 weight，那左側得到的 output z 和右側得到的 output z‘，它們的值其實是會相差兩倍的，即$z'≈2z$，這樣會造成 testing 的結果與 training 的結果並不 match，最終的 performance 反而會變差

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout6.png" width="60%;" /></center>
那這個時候，你就需要把右側testing中所有的weight乘上0.5，然後做normalization，這樣z就會等於z'，使得testing的結果與training的結果是比較match的

##### Dropout is a kind of ensemble

在文獻上有很多不同的觀點來解釋為什麼 dropout 會 work，其中一種比較令人信服的解釋是：**dropout 是一種終極的 ensemble 的方法**

###### ensemble 精神的解釋

ensemble 的方法在比賽的時候經常用得到，它的意思是說，我們有一個很大的 training set，那你每次都只從這個 training set 裡面 sample 一部分的 data 出來，像下圖一樣，抽取了 set1,set2,set3,set4

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout7.png" width="60%;" /></center>
我們之前在講bias和variance的trade off的時候說過，打靶有兩種情況：

- 一種是因為 bias 大而導致打不准(參數過少)
- 另一種是因為 variance 大而導致打不准(參數過多)

假設我們今天有一個很複雜的 model，它往往是 bias 比較准，但 variance 很大的情況，如果你有很多個笨重複雜的 model，雖然它們的 variance 都很大，但最後平均起來，結果往往就會很准

所以 ensemble 做的事情，就是利用這個特性，我們從原來的 training data 裡面 sample 出很多 subset，然後 train 很多個 model，每一個 model 的 structure 甚至都可以不一樣；在 testing 的時候，丟了一筆 testing data 進來，使它通過所有的 model，得到一大堆的結果，然後把這些結果平均起來當做最後的 output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout8.png" width="60%;" /></center>
如果你的model很複雜，這一招往往是很有用的，那著名的random forest(隨機森林)也是實踐這個精神的一個方法，也就是如果你用一個decision tree，它就會很弱，也很容易overfitting，而如果採用random forest，它就沒有那麼容易overfitting

###### 為什麼 dropout 是一個終極的 ensemble 方法呢？

在 training network 的時候，每次拿一個 minibatch 出來就做一次 update，而根據 dropout 的特性，每次 update 之前都要對所有的 neuron 進行 sample，因此每一個 minibatch 所訓練的 network 都是不同的

假設我們有 M 個 neuron，每個 neuron 都有可能 drop 或不 drop，所以總共可能的 network 數量有$2^M$個；所以當你在做 dropout 的時候，相當於是在用很多個 minibatch 分別去訓練很多個 network(一個 minibatch 一般設置為 100 筆 data)，由於 update 次數是有限的，所以做了幾次 update，就相當於 train 了幾個不同的 network，最多可以訓練到$2^M$個 network

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout9.png" width="60%;" /></center>
每個network都只用一個minibatch的data來train，可能會讓人感到不安，一個batch才100筆data，怎麼train一個network呢？其實沒有關係，因為這些**不同的network之間的參數是shared**，也就是說，雖然一個network只能用一個minibatch來train，但同一個weight可以在不同的network里被不同的minibatch train，所以同一個weight實際上是被所有沒有丟掉它的network一起share的，它是拿所有這些network的minibatch合起來一起train的結果

###### 實際操作 ensemble 的做法

那按照 ensemble 這個方法的邏輯，在 testing 的時候，你把那 train 好的一大把 network 通通拿出來，然後把手上這一筆 testing data 丟到這把 network 裡面去，每個 network 都給你吐出一個結果來，然後你把所有的結果平均起來 ，就是最後的 output

但是在實際操作上，如下圖左側所示，這一把 network 實在太多了，你沒有辦法每一個 network 都丟一個 input 進去，再把它們的 output 平均起來，這樣運算量太大了

所以 dropout 最神奇的地方是，當你並沒有把這些 network 分開考慮，而是用一個完整的 network，這個 network 的 weight 是用之前那一把 network train 出來的對應 weight 乘上(1-p%)，然後再把手上這筆 testing data 丟進這個完整的 network，得到的 output 跟 network 分開考慮的 ensemble 的 output，是驚人的相近

也就是說下圖左側 ensemble 的做法和右側 dropout 的做法，得到的結果是 approximate(近似)的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout10.png" width="60%;" /></center>
###### 舉例說明dropout和ensemble的關係

這裡用一個例子來解釋：

我們 train 一個下圖右上角所示的簡單的 network，它只有一個 neuron，activation function 是 linear 的，並且不考慮 bias，這個 network 經過 dropout 訓練以後得到的參數分別為$w_1,w_2$，那給它 input $x_1,x_2$，得到的 output 就是$z=w_1 x_1+w_2 x_2$

如果我們今天要做 ensemble 的話，theoretically 就是像下圖這麼做，每一個 neuron 都有可能被 drop 或不 drop，這裡只有兩個 input 的 neuron，所以我們一共可以得到 2^2=4 種 network

我們手上這筆 testing data $x_1,x_2$丟到這四個 network 中，分別得到 4 個 output：$w_1x_1+w_2x_2,w_2x_2,w_1x_1,0$，然後根據 ensemble 的精神，把這四個 network 的 output 通通都 average 起來，得到的結果是$\frac{1}{2}(w_1x_1+w_2x_2)$

那根據 dropout 的想法，我們把從 training 中得到的參數$w_1,w_2$乘上(1-50%)，作為 testing network 里的參數，也就是$w'_1,w'_2=(1-50\%)(w_1,w_2)=0.5w_1,0.5w_2$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/dropout11.png" width="60%;" /></center>
這邊想要呈現的是，在這個最簡單的case裡面，用不同的network structure做ensemble這件事情，跟我們用一整個network，並且把weight乘上一個值而不做ensemble所得到的output，其實是一樣的

值得注意的是，**只有是 linear 的 network，才會得到上述的等價關係**，如果 network 是非 linear 的，ensemble 和 dropout 是不 equivalent 的；但是，dropout 最後一個很神奇的地方是，雖然在 non-linear 的情況下，它是跟 ensemble 不相等的，但最後的結果還是會 work

==**如果 network 很接近 linear 的話，dropout 所得到的 performance 會比較好，而 ReLU 和 Maxout 的 network 相對來說是比較接近於 linear 的，所以我們通常會把含有 ReLU 或 Maxout 的 network 與 Dropout 配合起來使用**==
