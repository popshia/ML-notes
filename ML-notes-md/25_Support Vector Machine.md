# Support Vector Machine

支持向量機(SVM)有兩個特點：SVM=鉸鏈損失(Hinge Loss)+核技巧(Kernel Method)

注：建議先看這篇[博客](https://juejin.im/post/5d273c34e51d45599e019e4d)瞭解 SVM 基礎知識後再看本文的分析

#### Hinge Loss

##### Binary Classification

先回顧一下二元分類的做法，為了方便後續推導，這裡定義 data 的標籤為-1 和+1

- 當$f(x)>0$時，$g(x)=1$，表示屬於第一類別；當$f(x)<0$時，$g(x)=-1$，表示屬於第二類別

- 原本用$\sum \delta(g(x^n)\ne \hat y^n)$，不匹配的樣本點個數，來描述 loss function，其中$\delta=1$表示$x$與$\hat y$相匹配，反之$\delta=0$，但這個式子不可微分，無法使用梯度下降法更新參數

  因此使用近似的可微分的$l(f(x^n),\hat y^n)$來表示損失函數

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-bc.png" width="60%"/></center>

下圖中，橫坐標為$\hat y^n f(x)$，我們希望橫坐標越大越好：

- 當$\hat y^n>0$時，希望$f(x)$越正越好
- 當$\hat y^n<0$時，希望$f(x)$越負越好

縱坐標是 loss，原則上，當橫坐標$\hat y^n f(x)$越大的時候，縱坐標 loss 要越小，橫坐標越小，縱坐標 loss 要越大

##### ideal loss

在$L(f)=\sum\limits_n \delta(g(x^n)\ne \hat y^n)$的理想情況下，如果$\hat y^n f(x)>0$，則 loss=0，如果$\hat y^n f(x)<0$，則 loss=1，如下圖中加粗的黑線所示，可以看出該曲線是無法微分的，因此我們要另一條近似的曲線來替代該損失函數

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-bc2.png" width="60%"/></center>

##### square loss

下圖中的紅色曲線代表了 square loss 的損失函數：$l(f(x^n),\hat y^n)=(\hat y^n f(x^n)-1)^2$

- 當$\hat y^n=1$時，$f(x)$與 1 越接近越好，此時損失函數化簡為$(f(x^n)-1)^2$
- 當$\hat y^n=-1$時，$f(x)$與-1 越接近越好，此時損失函數化簡為$(f(x^n)+1)^2$
- 但實際上整條曲線是不合理的，它會使得$\hat y^n f(x)$很大的時候有一個更大的 loss

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-bc3.png" width="60%"/></center>

##### sigmoid+square loss

此外藍線代表 sigmoid+square loss 的損失函數：$l(f(x^n),\hat y^n)=(\sigma(\hat y^n f(x^n))-1)^2$

- 當$\hat y^n=1$時，$\sigma (f(x))$與 1 越接近越好，此時損失函數化簡為$(\sigma(f(x))-1)^2$
- 當$\hat y^n=-1$時，$\sigma (f(x))$與 0 越接近越好，此時損失函數化簡為$(\sigma(f(x)))^2$
- 在邏輯回歸的時候實踐過，一般 square loss 的方法表現並不好，而是用 cross entropy 會更好

##### sigmoid+cross entropy

綠線則是代表了 sigmoid+cross entropy 的損失函數：$l(f(x^n),\hat y^n)=ln(1+e^{-\hat y^n f(x)})$

- $\sigma (f(x))$代表了一個分布，而 Ground Truth 則是真實分布，這兩個分布之間的交叉熵，就是我們要去 minimize 的 loss
- 當$\hat y^n f(x)$很大的時候，loss 接近於 0
- 當$\hat y^n f(x)$很小的時候，loss 特別大
- 下圖是把損失函數除以$ln2$的曲線，使之變成 ideal loss 的 upper bound，且不會對損失函數本身產生影響
- 我們雖然不能 minimize 理想的 loss 曲線，但我們可以 minimize 它的 upper bound，從而起到最小化 loss 的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-bc4.png" width="60%"/></center>

##### cross entropy VS square error

為什麼 cross entropy 要比 square error 要來的有效呢？

- 我們期望在極端情況下，比如$\hat y^n$與$f(x)$非常不匹配導致橫坐標非常負的時候，loss 的梯度要很大，這樣才能盡快地通過參數調整回到 loss 低的地方

- 對 sigmoid+square loss 來說，當橫坐標非常負的時候，loss 的曲線反而是平緩的，此時去調整參數值對最終 loss 的影響其實並不大，它並不能很快地降低

  形象來說就是，「沒有回報，不想努力」

- 而對 cross entropy 來說，當橫坐標非常負的時候，loss 的梯度很大，稍微調整參數就可以往 loss 小的地方走很大一段距離，這對訓練是友好的

  形象來說就是，「努力可以有回報""

##### Hinge Loss

紫線代表了 hinge loss 的損失函數：$l(f(x^n),\hat y^n)=\max(0,1-\hat y^n f(x))$

- 當$\hat y^n=1$，損失函數化簡為$\max(0,1-f(x))$
  - 此時只要$f(x)>1$，loss 就會等於 0
- 當$\hat y^n=-1$，損失函數化簡為$\max(0,1+f(x))$
  - 此時只要$f(x)<-1$，loss 就會等於 0
- 總結一下，如果 label 為 1，則當$f(x)>1$，機器就認為 loss 為 0；如果 label 為-1，則當$f(x)<-1$，機器就認為 loss 為 0，因此該函數並不需要$f(x)$有一個很大的值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-bc5.png" width="60%"/></center>

在紫線中，當$\hat y^n f(x)>1$，則已經實現目標，loss=0；當$\hat y^n f(x)>0$，表示已經得到了正確答案，但 Hinge Loss 認為這還不夠，它需要你繼續往 1 的地方前進

事實上，Hinge Loss 也是 Ideal loss 的 upper bound，但是當橫坐標$\hat y^n f(x)>1$時，它與 Ideal loss 近乎是完全貼近的

比較 Hinge loss 和 cross entropy，最大的區別在於他們對待已經做得好的樣本點的態度，在橫坐標$\hat y^n f(x)>1$的區間上，cross entropy 還想要往更大的地方走，而 Hinge loss 則已經停下來了，就像一個的目標是」還想要更好「，另一個的目標是」及格就好「

在實作上，兩者差距並不大，而 Hinge loss 的優勢在於它不怕 outlier，訓練出來的結果魯棒性(robust)比較強

#### Linear SVM

##### model description

在線性的 SVM 里，我們把$f(x)=\sum\limits_i w_i x_i+b=w^Tx$看做是向量$\left [\begin{matrix}w\\b \end{matrix}\right ]$和向量$\left [\begin{matrix}x\\1 \end{matrix}\right ]$的內積，也就是新的$w$和$x$，這麼做可以把 bias 項省略掉

在損失函數中，我們通常會加上一個正規項，即$L(f)=\sum\limits_n l(f(x^n),\hat y^n)+\lambda ||w||_2$

這是一個 convex 的損失函數，好處在於無論從哪個地方開始做梯度下降，最終得到的結果都會在最低處，曲線中一些折角處等不可微的點可以參考 NN 中 relu、maxout 等函數的微分處理

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-linear.png" width="60%"/></center>

對比 Logistic Regression 和 Linear SVM，兩者唯一的區別就是損失函數不同，前者用的是 cross entropy，後者用的是 Hinge loss

事實上，SVM 並不局限於 Linear，儘管 Linear 可以帶來很多好的特質，但我們完全可以在一個 Deep 的神經網絡中使用 Hinge loss 的損失函數，就成為了 Deep SVM，其實 Deep Learning、SVM 這些方法背後的精神都是相通的，並沒有那麼大的界限

##### gradient descent

儘管 SVM 大多不是用梯度下降訓練的，但使用該方法訓練確實是可行的，推導過程如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-gd.png" width="60%"/></center>

##### another formulation

前面列出的式子可能與你平常看到的 SVM 不大一樣，這裡將其做一下簡單的轉換

對$L(f)=\sum\limits_n \max(0,1-\hat y^n f(x))+\lambda ||w||_2$，用$L(f)=\sum\limits_n \epsilon^n+\lambda ||w||_2$來表示

其中$\epsilon^n=\max(0,1-\hat y^n f(x))$

對$\epsilon^n\geq0$、$\epsilon^n\geq1-\hat y^n f(x)$來說，它與上式原本是不同的，因為 max 是二選一，而$\geq$則取到等號的限制

但是當加上取 loss function $L(f)$最小化這個條件時，$\geq$就要取到等號，兩者就是等價的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-formulation.png" width="60%"/></center>

此時該表達式就和你熟知的 SVM 一樣了：

$L(f)=\sum\limits_n \epsilon^n+\lambda ||w||_2$，且$\hat y^n f(x)\geq 1-\epsilon^n$，其中$\hat y^n$和$f(x)$要同號，$\epsilon^n$要大於等於 0，這裡$\epsilon^n$的作用就是放寬 1 的 margin，也叫作鬆弛變量(slack variable)

這是一個 QP 問題(Quadradic programming problem)，可以用對應方法求解，當然前面提到的梯度下降法也可以解

#### Kernel Method

##### explain linear combination

你要先說服你自己一件事：實際上我們找出來的可以 minimize 損失函數的參數，其實就是 data 的線性組合

$$
w^*=\sum\limits_n \alpha^*_n x^n
$$

你可以通過拉格朗日乘數法去求解前面的式子來驗證，這裡試圖從梯度下降的角度來解釋：

觀察$w$的更新過程$w=w-\eta\sum\limits_n c^n(w)x^n$可知，如果$w$被初始化為 0，則每次更新的時候都是加上 data point $x$的線性組合，因此最終得到的$w$依舊會是$x$的 Linear Combination

而使用 Hinge loss 的時候，$c^n(w)$往往會是 0，不是所有的$x^n$都會被加到$w$里去，而被加到$w$里的那些$x^n$，就叫做**support vector**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-dual.png" width="60%"/></center>

SVM 解出來的$\alpha_n$是 sparse 的，因為有很多$x^n$的系數微分為 0，這意味著即使從數據集中把這些$x^n$的樣本點移除掉，對結果也是沒有影響的，這可以增強系統的魯棒性；而在傳統的 cross entropy 的做法里，每一筆 data 對結果都會有影響，因此魯棒性就沒有那麼好

##### redefine model and loss function

知道$w$是$x^n$的線性組合之後，我們就可以對原先的 SVM 函數進行改寫：

$$
w=\sum_n\alpha_nx^n=X\alpha \\
f(x)=w^Tx=\alpha^TX^Tx=\sum_n\alpha_n(x^n\cdot x)
$$

這裡的$x$表示新的 data，$x^n$表示數據集中已存在的所有 data，由於很多$\alpha_n$為 0，因此計算量並不是很大

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-dual2.png" width="60%"/></center>

接下來把$x^n$與$x$的內積改寫成 Kernel function 的形式：$x^n\cdot x=K(x^n,x)$

此時 model 就變成了$f(x)= \sum\limits_n\alpha_n K(x^n,x)$，未知的參數變成了$\alpha_n$

現在我們的目標是，找一組最好的$\alpha_n$，讓 loss 最小，此時損失函數改寫為：

$$
L(f)=\sum\limits_n l(\sum\limits_{n'} \alpha_{n'}K(x^{n'},x^n),\hat y^n)
$$

從中可以看出，我們並不需要真的知道$x$的 vector 是多少，需要知道的只是$x$跟$z$之間的內積值$K(x,z)$，也就是說，只要知道 Kernel function $K(x,z)$，就可以去對參數做優化了，這招就叫做**Kernel Trick**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-dual3.png" width="60%"/></center>

##### Kernel Trick

linear model 會有很多的限制，有時候需要對輸入的 feature 做一些轉換之後，才能用 linear model 來處理，假設現在我們的 data 是二維的，$x=\left[ \begin{matrix}x_1\\x_2 \end{matrix} \right]$，先要對它做 feature transform，然後再去應用 Linear SVM

如果要考慮特徵之間的關係，則把特徵轉換為$\phi(x)=\left[ \begin{matrix}x_1^2\\\sqrt{2}x_1x_2\\ x_2^2 \end{matrix} \right]$，此時 Kernel function 就變為：

$$
K(x,z)=\phi(x)\cdot \phi(z)=\left[ \begin{matrix}x_1^2\\\sqrt{2}x_1x_2\\ x_2^2 \end{matrix} \right] \cdot \left[ \begin{matrix}z_1^2\\\sqrt{2}z_1z_2\\ z_2^2 \end{matrix} \right]=(x_1z_1+x_2z_2)^2=(\left[ \begin{matrix}x_1\\x_2 \end{matrix} \right]\cdot \left[ \begin{matrix}z_1\\z_2 \end{matrix} \right])^2=(x\cdot z)^2
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-kernel.png" width="60%"/></center>

可見，我們對$x$和$z$做特徵轉換+內積，就等同於**在原先的空間上先做內積再平方**，在高維空間里，這種方式可以有更快的速度和更小的運算量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-kernel2.png" width="60%"/></center>

##### RBF Kernel

在 RBF Kernel 中，$K(x,z)=e^{-\frac{1}{2}||x-z||_2}$，實際上也可以表示為$\phi(x)\cdot \phi(z)$，只不過$\phi(*)$的維數是無窮大的，所以我們直接使用 Kernel trick 計算，其實就等同於在無窮多維的空間中計算兩個向量的內積

將 Kernel 展開成無窮維如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-kernel3.png" width="60%"/></center>

把與$x$相關的無窮多項串起來就是$\phi(x)$，把與$z$相關的無窮多項串起來就是$\phi(z)$，也就是說，當你使用 RBF Kernel 的時候，實際上就是在無窮多維的平面上做事情，當然這也意味著很容易過擬合

##### Sigmoid Kernel

Sigmoid Kernel：$K(x,z)=\tanh(x,z)$

如果使用的是 Sigmoid Kernel，那 model $f(x)$就可以被看作是只有一層 hidden layer 的神經網絡，其中$x^1$\~$x^n$可以被看作是 neuron 的 weight，變量$x$乘上這些 weight，再通過 tanh 激活函數，最後全部乘上$\alpha^1$\~$\alpha^n$做加權和，得到最後的$f(x)$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-kernel4.png" width="60%"/></center>

其中 neuron 的數目，由 support vector 的數量決定

##### Design Kernel Function

既然有了 Kernel Trick，其實就可以直接去設計 Kernel Function，它代表了投影到高維以後的內積，類似於相似度的概念

我們完全可以不去管$x$和$z$的特徵長什麼樣，因為用低維的$x$和$z$加上$K(x,z)$，就可以直接得到高維空間中$x$和$z$經過轉換後的內積，這樣就省去了轉換特徵這一步

當$x$是一個有結構的對象，比如不同長度的 sequence，它們其實不容易被表示成 vector，我們不知道$x$的樣子，就更不用說$\phi(x)$了，但是只要知道怎麼計算兩者之間的相似度，就有機會把這個 Similarity 當做 Kernel 來使用

我們隨便定義一個 Kernel Function，其實並不一定能夠拆成兩個向量內積的結果，但有 Mercer's theory 可以幫助你判斷當前的 function 是否可拆分

下圖是直接定義語音 vector 之間的相似度$K(x,z)$來做 Kernel Trick 的示例：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-kernel5.png" width="60%"/></center>

#### SVM vs Deep Learning

這裡簡單比較一下 SVM 和 Deep Learning 的差別：

- deep learning 的前幾層 layer 可以看成是在做 feature transform，而後幾層 layer 則是在做 linear classifier

- SVM 也類似，先用 Kernel Function 把 feature transform 到高維空間上，然後再使用 linear classifier

  在 SVM 里一般 Linear Classifier 都會採用 Hinge Loss

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/svm-dl.png" width="60%"/></center>
