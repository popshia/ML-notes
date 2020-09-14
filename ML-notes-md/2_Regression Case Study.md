# Regression：Case Study

> **回歸**-案例研究

#### 問題的導入：預測寶可夢的 CP 值

Estimating the Combat Power(CP) of a pokemon after evolution

我們期望根據已有的寶可夢進化前後的信息，來預測某只寶可夢進化後的 cp 值的大小

#### 確定 Senario、Task 和 Model

##### Senario

首先根據已有的 data 來確定 Senario，我們擁有寶可夢進化前後 cp 值的這樣一筆數據，input 是進化前的寶可夢(包括它的各種屬性)，output 是進化後的寶可夢的 cp 值；因此我們的 data 是 labeled，使用的 Senario 是**Supervised Learning**

##### Task

然後根據我們想要 function 的輸出類型來確定 Task，我們預期得到的是寶可夢進化後的 cp 值，是一個 scalar，因此使用的 Task 是**Regression**

##### Model

關於 Model，選擇很多，這裡採用的是**Non-linear Model**

#### 設定具體參數

$X$： 表示一隻寶可夢，用下標表示該寶可夢的某種屬性

$X_{cp}$：表示該寶可夢進化前的 cp 值

$X_s$： 表示該寶可夢是屬於哪一種物種，比如妙瓜種子、皮卡丘...

$X_{hp}$：表示該寶可夢的 hp 值即生命值是多少

$X_w$： 代表該寶可夢的重重量

$X_h$： 代表該寶可夢的高度

$f()$： 表示我們要找的 function

$y$： 表示 function 的 output，即寶可夢進化後的 cp 值，是一個 scalar

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pokeman-parameters.png" alt="pokeman-parameters" style="width:60%;"/></center>
#### Regression的具體過程

##### 回顧一下 machine Learning 的三個步驟：

- 定義一個 model 即 function set
- 定義一個 goodness of function 損失函數去評估該 function 的好壞
- 找一個最好的 function

##### Step1：Model (function set)

如何選擇一個 function 的模型呢？畢竟只有確定了模型才能調參。這裡沒有明確的思路，只能憑經驗去一種種地試

###### Linear Model 線性模型

$y=b+w \cdot X_{cp}$

y 代表進化後的 cp 值，$X_{cp}$代表進化前的 cp 值，w 和 b 代表未知參數，可以是任何數值

根據不同的 w 和 b，可以確定不同的無窮無盡的 function，而$y=b+w \cdot X_{cp}$這個抽象出來的式子就叫做 model，是以上這些具體化的 function 的集合，即 function set

實際上這是一種**Linear Model**，但只考慮了寶可夢進化前的 cp 值，因而我們可以將其擴展為：

==$y=b+ \sum w_ix_i$==

**x~i~**： an attribute of input X ( x~i~ is also called **feature**，即特徵值)

**w~i~**：weight of x~i~

**b**： bias

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/model.png" alt="model" style="width:60%;" /></center>
##### Step2：Goodness of Function

###### 參數說明

$x^i$：用上標來表示一個完整的 object 的編號，$x^{i}$表示第 i 只寶可夢(下標表示該 object 中的 component)

$\widehat{y}^i$：用$\widehat{y}$表示一個實際觀察到的 object 輸出，上標為 i 表示是第 i 個 object

注：由於 regression 的輸出值是 scalar，因此$\widehat{y}$裡面並沒有 component，只是一個簡單的數值；但是未來如果考慮 structured Learning 的時候，我們 output 的 object 可能是有 structured 的，所以我們還是會需要用上標下標來表示一個完整的 output 的 object 和它包含的 component

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/goodness-of-function.png" alt="goodness-of-function" style="width:60%;" /></center>
###### Loss function 損失函數

為了衡量 function set 中的某個 function 的好壞，我們需要一個評估函數，即==Loss function==，損失函數，簡稱`L`；`loss function`是一個 function 的 function

$$L(f)=L(w,b)$$

input：a function；

output：how bad/good it is

由於$f:y=b+w \cdot x_{cp}$，即`f`是由`b`和`w`決定的，因此`input f`就等價於`input`這個`f`里的`b`和`w`，因此==Loss function 實際上是在衡量一組參數的好壞==

之前提到的 model 是由我們自主選擇的，這裡的 loss function 也是，最常用的方法就是採用類似於方差和的形式來衡量參數的好壞，即預測值與真值差的平方和；這裡真正的數值減估測數值的平方，叫做估測誤差，Estimation error，將 10 個估測誤差合起來就是 loss function

$$ L(f)=L(w,b)=\sum_{n=1}^{10}(\widehat{y}^n-(b+w \cdot {x}^n_{cp}))^2$$

如果$L(f)$越大，說明該 function 表現得越不好；$L(f)$越小，說明該 function 表現得越好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/loss-function.png" alt="loss-function" style="width:60%;" /></center>
###### Loss function可視化

下圖中是 loss function 的可視化，該圖中的每一個點都代表一組`(w,b)`，也就是對應著一個`function`；而該點的顏色對應著的 loss function 的結果`L(w,b)`，它表示該點對應 function 的表現有多糟糕，顏色越偏紅色代表 Loss 的數值越大，這個 function 的表現越不好，越偏藍色代表 Loss 的數值越小，這個 function 的表現越好

比如圖中用紅色箭頭標注的點就代表了 b=-180 , w=-2 對應的 function，即$y=-180-2 \cdot x_{cp}$，該點所在的顏色偏向於紅色區域，因此這個 function 的 loss 比較大，表現並不好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/loss-figure.png" alt="loss-figure" style="width:60%;" /></center>
##### Step3：Pick the Best Function

我們已經確定了 loss function，他可以衡量我們的 model 裡面每一個 function 的好壞，接下來我們要做的事情就是，從這個 function set 裡面，挑選一個最好的 function

挑選最好的 function 這一件事情，寫成 formulation/equation 的樣子如下：

$$f^*={arg} \underset{f}{min} L(f)$$，或者是

$$w^*,b^*={arg}\ \underset{w,b}{min} L(w,b)={arg}\  \underset{w,b}{min} \sum\limits^{10}_{n=1}(\widehat{y}^n-(b+w \cdot x^n_{cp}))^2$$

也就是那個使$L(f)=L(w,b)=Loss$最小的$f$或$(w,b)$，就是我們要找的$f^*$或$(w^*,b^*)$(有點像極大似然估計的思想)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/best-function.png" alt="best-function" style="width:60%;" /></center>
利用線性代數的知識，可以解得這個closed-form solution，但這裡採用的是一種更為普遍的方法——==gradient descent(梯度下降法)==

#### Gradient Descent 梯度下降

上面的例子比較簡單，用線性代數的知識就可以解；但是對於更普遍的問題來說，==gradient descent 的厲害之處在於，只要$L(f)$是可微分的，gradient descent 都可以拿來處理這個$f$，找到表現比較好的 parameters==

##### 單個參數的問題

以只帶單個參數 w 的 Loss Function `L(w)`為例，首先保證$L(w)$是**可微**的
$$w^*={arg}\ \underset{w}{min} L(w) $$ 我們的目標就是找到這個使 Loss 最小的$w^*$，實際上就是尋找切線 L 斜率為 0 的 global minima 最小值點(注意，存在一些 local minima 極小值點，其斜率也是 0)

有一個暴力的方法是，窮舉所有的 w 值，去找到使 loss 最小的$w^*$，但是這樣做是沒有效率的；而 gradient descent 就是用來解決這個效率問題的

- 首先隨機選取一個初始的點$w^0$ (當然也不一定要隨機選取，如果有辦法可以得到比較接近$w^*$的表現得比較好的$w^0$當初始點，可以有效地提高查找$w^*$的效率)

- 計算$L$在$w=w^0$的位置的微分，即$\frac{dL}{dw}|_{w=w^0}$，幾何意義就是切線的斜率

- 如果切線斜率是 negative 負的，那麼就應該使 w 變大，即往右踏一步；如果切線斜率是 positive 正的，那麼就應該使 w 變小，即往左踏一步，每一步的步長 step size 就是 w 的改變量

  w 的改變量 step size 的大小取決於兩件事

  - 一是現在的微分值$\frac{dL}{dw}$有多大，微分值越大代表現在在一個越陡峭的地方，那它要移動的距離就越大，反之就越小；

  - 二是一個常數項$η$，被稱為==learning rate==，即學習率，它決定了每次踏出的 step size 不只取決於現在的斜率，還取決於一個事先就定好的數值，如果 learning rate 比較大，那每踏出一步的時候，參數 w 更新的幅度就比較大，反之參數更新的幅度就比較小

    如果 learning rate 設置的大一些，那機器學習的速度就會比較快；但是 learning rate 如果太大，可能就會跳過最合適的 global minima 的點

- 因此每次參數更新的大小是 $η \frac{dL}{dw}$，為了滿足斜率為負時 w 變大，斜率為正時 w 變小，應當使原來的 w 減去更新的數值，即

  $$
  w^1=w^0-η \frac{dL}{dw}|_{w=w^0} \\
  w^2=w^1-η \frac{dL}{dw}|_{w=w^1} \\
  w^3=w^2-η \frac{dL}{dw}|_{w=w^2} \\
  ... \\
  w^{i+1}=w^i-η \frac{dL}{dw}|_{w=w^i} \\
  if\ \ (\frac{dL}{dw}|_{w=w^i}==0) \ \ then \ \ stop;
  $$

  此時$w^i$對應的斜率為 0，我們找到了一個極小值 local minima，這就出現了一個問題，當微分為 0 的時候，參數就會一直卡在這個點上沒有辦法再更新了，因此通過 gradient descent 找出來的 solution 其實並不是最佳解 global minima

  但幸運的是，在 linear regression 上，是沒有 local minima 的，因此可以使用這個方法

    <center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-descent.png" alt="gradient-descent" style="width:60%;" /></center>

##### 兩個參數的問題

今天要解決的關於寶可夢的問題，是含有 two parameters 的問題，即$(w^*,b^*)=arg\ \underset{w,b} {min} L(w,b)$

當然，它本質上處理單個參數的問題是一樣的

- 首先，也是隨機選取兩個初始值，$w^0$和$b^0$

- 然後分別計算$(w^0,b^0)$這個點上，L 對 w 和 b 的偏微分，即$\frac{\partial L}{\partial w}|_{w=w^0,b=b^0}$ 和 $\frac{\partial L}{\partial b}|_{w=w^0,b=b^0}$

- 更新參數，當迭代跳出時，$(w^i,b^i)$對應著極小值點
  $$
  w^1=w^0-η\frac{\partial L}{\partial w}|_{w=w^0,b=b^0} \ \ \ \ \ \ \ \  \ b^1=b^0-η\frac{\partial L}{\partial b}|_{w=w^0,b=b^0} \\
  w^2=w^1-η\frac{\partial L}{\partial w}|_{w=w^1,b=b^1} \ \ \ \ \ \ \ \  \ b^2=b^1-η\frac{\partial L}{\partial b}|_{w=w^1,b=b^1} \\
  ... \\
  w^{i+1}=w^{i}-η\frac{\partial L}{\partial w}|_{w=w^{i},b=b^{i}} \ \ \ \ \ \ \ \  \ b^{i+1}=b^{i}-η\frac{\partial L}{\partial b}|_{w=w^{i},b=b^{i}} \\
  if(\frac{\partial L}{\partial w}==0 \&\& \frac{\partial L}{\partial b}==0) \ \ \ then \ \ stop
  $$

實際上，L 的 gradient 就是微積分中的那個梯度的概念，即

$$
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w} \\
\frac{\partial L}{\partial b}
\end{bmatrix}_{gradient}
$$

可視化效果如下：(三維坐標顯示在二維圖像中，loss 的值用顏色來表示)

橫坐標是 b，縱坐標是 w，顏色代表 loss 的值，越偏藍色表示 loss 越小，越偏紅色表示 loss 越大

**每次計算得到的梯度 gradient，即由$\frac{\partial L}{\partial b}和\frac{\partial L}{\partial w}$組成的 vector 向量，就是該等高線的法線方向(對應圖中紅色箭頭的反方向)；而$(-η\frac{\partial L}{\partial b},-η\frac{\partial L}{\partial w})$的作用就是讓原先的$(w^i,b^i)$朝著 gradient 的反方向即等高線法線方向前進，其中 η(learning rate)的作用是每次更新的跨度(對應圖中紅色箭頭的長度)；經過多次迭代，最終 gradient 達到極小值點**

注：這裡兩個方向的 η(learning rate)必須保持一致，這樣每次更新坐標的 step size 是等比例縮放的，保證坐標前進的方向始終和梯度下降的方向一致；否則坐標前進的方向將會發生偏移

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-two-parameters.png" alt="gradient-two-parameters" style="width: 60%;" /></center>
##### Gradient Descent的缺點

gradient descent 有一個令人擔心的地方，也就是我之前一直提到的，它每次迭代完畢，尋找到的梯度為 0 的點必然是極小值點，local minima；卻不一定是最小值點，global minima

這會造成一個問題是說，如果 loss function 長得比較坑坑窪窪(極小值點比較多)，而每次初始化$w^0$的取值又是隨機的，這會造成每次 gradient descent 停下來的位置都可能是不同的極小值點；而且當遇到梯度比較平緩(gradient≈0)的時候，gradient descent 也可能會效率低下甚至可能會 stuck 卡住；也就是說通過這個方法得到的結果，是看人品的(滑稽

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-stuck.png" alt="gradient-stuck" style="width:60%;" /></center>
但是！在==linear regression==里，loss function實際上是**convex**的，是一個**凸函數**，是沒有local optimal局部最優解的，他只有一個global minima，visualize出來的圖像就是從里到外一圈一圈包圍起來的橢圓形的等高線(就像前面的等高線圖)，因此隨便選一個起始點，根據gradient descent最終找出來的，都會是同一組參數

#### 回到 pokemon 的問題上來

##### 偏微分的計算

現在我們來求具體的 L 對 w 和 b 的偏微分

$$
L(w,b)=\sum\limits_{n=1}^{10}(\widehat{y}^n-(b+w\cdot x_{cp}^n))^2 \\
\frac{\partial L}{\partial w}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n))(-x_{cp}^n) \\
\frac{\partial L}{\partial b}=\sum\limits_{n=1}^{10}2(\widehat{y}^n-(b+w\cdot x_{cp}^n))(-1)
$$

##### How's the results?

根據 gradient descent，我們得到的$y=b+w\cdot x_{cp}$中最好的參數是 b=-188.4, w=2.7

我們需要有一套評估系統來評價我們得到的最後這個 function 和實際值的誤差 error 的大小；這裡我們將 training data 里每一隻寶可夢 $i$ 進化後的實際 cp 值與預測值之差的絕對值叫做$e^i$，而這些誤差之和 Average Error on Training Data 為$\sum\limits_{i=1}^{10}e^i=31.9$

> What we really care about is the error on new data (testing data)

當然我們真正關心的是 generalization 的 case，也就是用這個 model 去估測新抓到的 pokemon，誤差會有多少，這也就是所謂的 testing data 的誤差；於是又抓了 10 只新的 pokemon，算出來的 Average Error on Testing Data 為$\sum\limits_{i=1}^{10}e^i=35.0$；可見 training data 里得到的誤差一般是要比 testing data 要小，這也符合常識

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/results.png" alt="results" style="width:60%;" /></center>
##### How can we do better?

我們有沒有辦法做得更好呢？這時就需要我們重新去設計 model；如果仔細觀察一下上圖的 data，就會發現在原先的 cp 值比較大和比較小的地方，預測值是相當不准的

實際上，從結果來看，最終的 function 可能不是一條直線，可能是稍微更複雜一點的曲線

###### 考慮$(x_{cp})^2$的 model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-2.png" alt="Xcp-2" style="width:50%;" /></center>
###### 考慮$(x_{cp})^3$的model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-3.png" alt="Xcp-3" style="width:50%;" /></center>
###### 考慮$(x_{cp})^4$的model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-4.png" alt="Xcp-4" style="width:50%;" /></center>
###### 考慮$(x_{cp})^5$的model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-5.png" alt="Xcp-5" style="width:50%;" /></center>
###### 5個model的對比

這 5 個 model 的 training data 的表現：隨著$(x_{cp})^i$的高次項的增加，對應的 average error 會不斷地減小；實際上這件事情非常容易解釋，實際上低次的式子是高次的式子的特殊情況(令高次項$(X_{cp})^i$對應的$w_i$為 0，高次式就轉化成低次式)

也就是說，在 gradient descent 可以找到 best function 的前提下(多次式為 Non-linear model，存在 local optimal 局部最優解，gradient descent 不一定能找到 global minima)，function 所包含的項的次數越高，越複雜，error 在 training data 上的表現就會越來越小；但是，我們關心的不是 model 在 training data 上的 error 表現，而是 model 在 testing data 上的 error 表現

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-compare.png" alt="compare" style="width:60%;" /></center>
在training data上，model越複雜，error就會越低；但是在testing data上，model複雜到一定程度之後，error非但不會減小，反而會暴增，在該例中，從含有$(X_{cp})^4$項的model開始往後的model，testing data上的error出現了大幅增長的現象，通常被稱為**overfitting過擬合**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/Xcp-overfitting.png" alt="overfitting" style="width:60%;" /></center>
因此model不是越複雜越好，而是選擇一個最適合的model，在本例中，包含$(X_{cp})^3$的式子是最適合的model

##### 進一步討論其他參數

###### 物種$x_s$的影響

之前我們的 model 只考慮了寶可夢進化前的 cp 值，這顯然是不對的，除了 cp 值外，還受到物種$x_s$的影響

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/hidden-factors.png" alt="hidden-factors" style="width:60%;" /></center>
因此我們重新設計model：
$$
if \ \ x_s=Pidgey: \ \ \ \ \ \ \ y=b_1+w_1\cdot x_{cp} \\
if \ \ x_s=Weedle: \ \ \ \ \ \ y=b_2+w_2\cdot x_{cp} \\
if \ \ x_s=Caterpie: \ \ \ \ y=b_3+w_3\cdot x_{cp} \\
if \ \ x_s=Eevee: \ \ \ \ \ \ \ \ \ y=b_4+w_4\cdot x_{cp} 
$$
也就是根據不同的物種，設計不同的linear model(這裡$x_s=species \ of \ x$)，那如何將上面的四個if語句合併成一個linear model呢？

這裡引入$δ(條件表達式)$的概念，當條件表達式為 true，則 δ 為 1；當條件表達式為 false，則 δ 為 0，因此可以通過下圖的方式，將 4 個 if 語句轉化成同一個 linear model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/new-model.png" alt="new-model" style="width:60%;" /></center>
有了上面這個model以後，我們分別得到了在training data和testing data上測試的結果：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/new-results.png" alt="new-results" style="width:60%;" /></center>
###### Hp值$x_{hp}$、height值$x_h$、weight值$x_w$的影響

考慮所有可能有影響的參數，設計出這個最複雜的 model：

$$
if \ \ x_s=Pidgey: \ \  \ \ y'=b_1+w_1\cdot x_{cp}+w_5\cdot(x_{cp})^2 \\
if \ \ x_s=Weedle: \ \ \ y'=b_2+w_2\cdot x_{cp}+w_6\cdot(x_{cp})^2 \\
if \ \ x_s=Pidgey: \ \ \ y'=b_3+w_3\cdot x_{cp}+w_7\cdot(x_{cp})^2 \\
if \ \ x_s=Eevee: \ \ \ \ y'=b_4+w_4\cdot x_{cp}+w_8\cdot(x_{cp})^2 \\
y=y'+w_9\cdot x_{hp}+w_{10}\cdot(x_{hp})^2+w_{11}\cdot x_h+w_{12}\cdot (x_h)^2+w_{13}\cdot x_w+w_{14}\cdot (x_w)^2
$$

算出的 training error=1.9，但是，testing error=102.3！**這麼複雜的 model 很大概率會發生 overfitting**(按照我的理解，overfitting 實際上是我們多使用了一些 input 的變量或是變量的高次項使曲線跟 training data 擬合的更好，但不幸的是這些項並不是實際情況下被使用的，於是這個 model 在 testing data 上會表現得很糟糕)，overfitting 就相當於是那個範圍更大的韋恩圖，它包含了更多的函數更大的範圍，代價就是在準確度上表現得更糟糕

###### regularization 解決 overfitting(L2 正則化解決過擬合問題)

> regularization 可以使曲線變得更加 smooth，training data 上的 error 變大，但是 testing data 上的 error 變小。有關 regularization 的具體原理說明詳見下一部分

原來的 loss function 只考慮了 prediction 的 error，即$\sum\limits_i^n(\widehat{y}^i-(b+\sum\limits_{j}w_jx_j))^2$；而 regularization 則是在原來的 loss function 的基礎上加上了一項$\lambda\sum(w_i)^2$，就是把這個 model 裡面所有的$w_i$的平方和用 λ 加權(其中 i 代表遍歷 n 個 training data，j 代表遍歷 model 的每一項)

也就是說，**我們期待參數$w_i$越小甚至接近於 0 的 function，為什麼呢？**

因為參數值接近 0 的 function，是比較平滑的；所謂的平滑的意思是，當今天的輸入有變化的時候，output 對輸入的變化是比較不敏感的

舉例來說，對$y=b+\sum w_ix_i$這個 model，當 input 變化$\Delta x_i$，output 的變化就是$w_i\Delta x_i$，也就是說，如果$w_i$越小越接近 0 的話，輸出對輸入就越不 sensitive 敏感，我們的 function 就是一個越平滑的 function；說到這裡你會發現，我們之前沒有把 bias——b 這個參數考慮進去的原因是**bias 的大小跟 function 的平滑程度是沒有關係的**，bias 值的大小只是把 function 上下移動而已

**那為什麼我們喜歡比較平滑的 function 呢？**

如果我們有一個比較平滑的 function，由於輸出對輸入是不敏感的，測試的時候，一些 noises 噪聲對這個平滑的 function 的影響就會比較小，而給我們一個比較好的結果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization.png" alt="regularization" style="width:60%;" /></center>
**注：這裡的λ需要我們手動去調整以取得最好的值**

λ 值越大代表考慮 smooth 的那個 regularization 那一項的影響力越大，我們找到的 function 就越平滑

觀察下圖可知，當我們的 λ 越大的時候，在 training data 上得到的 error 其實是越大的，但是這件事情是非常合理的，因為當 λ 越大的時候，我們就越傾向於考慮 w 的值而越少考慮 error 的大小；但是有趣的是，雖然在 training data 上得到的 error 越大，但是在 testing data 上得到的 error 可能會是比較小的

下圖中，當 λ 從 0 到 100 變大的時候，training error 不斷變大，testing error 反而不斷變小；但是當 λ 太大的時候(>100)，在 testing data 上的 error 就會越來越大

==我們喜歡比較平滑的 function，因為它對 noise 不那麼 sensitive；但是我們又不喜歡太平滑的 function，因為它就失去了對 data 擬合的能力；而 function 的平滑程度，就需要通過調整 λ 來決定==，就像下圖中，當 λ=100 時，在 testing data 上的 error 最小，因此我們選擇 λ=100

注：這裡的 error 指的是$\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^i-y^i|$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization-performance.png" alt="regularization-performance" style="width:60%;" /></center>
#### conclusion總結

##### 關於 pokemon 的 cp 值預測的流程總結：

- 根據已有的 data 特點(labeled data，包含寶可夢及進化後的 cp 值)，確定使用 supervised learning 監督學習

- 根據 output 的特點(輸出的是 scalar 數值)，確定使用 regression 回歸(linear or non-linear)

- 考慮包括進化前 cp 值、species、hp 等各方面變量屬性以及高次項的影響，我們的 model 可以採用這些 input 的一次項和二次型之和的形式，如：

  $$
  if \ \ x_s=Pidgey: \ \  \ \ y'=b_1+w_1\cdot x_{cp}+w_5\cdot(x_{cp})^2 \\
  if \ \ x_s=Weedle: \ \ \ y'=b_2+w_2\cdot x_{cp}+w_6\cdot(x_{cp})^2 \\
  if \ \ x_s=Pidgey: \ \ \ y'=b_3+w_3\cdot x_{cp}+w_7\cdot(x_{cp})^2 \\
  if \ \ x_s=Eevee: \ \ \ \ y'=b_4+w_4\cdot x_{cp}+w_8\cdot(x_{cp})^2 \\
  y=y'+w_9\cdot x_{hp}+w_{10}\cdot(x_{hp})^2+w_{11}\cdot x_h+w_{12}\cdot (x_h)^2+w_{13}\cdot x_w+w_{14}\cdot (x_w)^2
  $$

  而為了保證 function 的平滑性，loss function 應使用 regularization，即$L=\sum\limits_{i=1}^n(\widehat{y}^i-y^i)^2+\lambda\sum\limits_{j}(w_j)^2$，注意 bias——參數 b 對 function 平滑性無影響，因此不額外再次計入 loss function(y 的表達式里已包含 w、b)

- 利用 gradient descent 對 regularization 版本的 loss function 進行梯度下降迭代處理，每次迭代都減去 L 對該參數的微分與 learning rate 之積，假設所有參數合成一個 vector：$[w_0,w_1,w_2,...,w_j,...,b]^T$，那麼每次梯度下降的表達式如下：

  $$
  梯度:
  \nabla L=
  \begin{bmatrix}
  \frac{\partial L}{\partial w_0} \\
  \frac{\partial L}{\partial w_1} \\
  \frac{\partial L}{\partial w_2} \\
  ... \\
  \frac{\partial L}{\partial w_j} \\
  ... \\
  \frac{\partial L}{\partial b}
  \end{bmatrix}_{gradient}
  \ \ \
  gradient \ descent:
  \begin{bmatrix}
  w'_0\\
  w'_1\\
  w'_2\\
  ...\\
  w'_j\\
  ...\\
  b'
  \end{bmatrix}_{L=L'}
  = \ \ \ \ \ \
  \begin{bmatrix}
  w_0\\
  w_1\\
  w_2\\
  ...\\
  w_j\\
  ...\\
  b
  \end{bmatrix}_{L=L_0}
  -\ \ \ \ \eta
  \begin{bmatrix}
  \frac{\partial L}{\partial w_0} \\
  \frac{\partial L}{\partial w_1} \\
  \frac{\partial L}{\partial w_2} \\
  ... \\
  \frac{\partial L}{\partial w_j} \\
  ... \\
  \frac{\partial L}{\partial b}
  \end{bmatrix}_{L=L_0}
  $$

  當梯度穩定不變時，即$\nabla L$為 0 時，gradient descent 便停止，此時如果採用的 model 是 linear 的，那麼 vector 必然落於 global minima 處(凸函數)；如果採用的 model 是 Non-linear 的，vector 可能會落於 local minima 處(此時需要採取其他辦法獲取最佳的 function)

  假定我們已經通過各種方法到達了 global minima 的地方，此時的 vector：$[w_0,w_1,w_2,...,w_j,...,b]^T$所確定的那個唯一的 function 就是在該 λ 下的最佳$f^*$，即 loss 最小

- 這裡 λ 的最佳數值是需要通過我們不斷調整來獲取的，因此令 λ 等於 0，10，100，1000，...不斷使用 gradient descent 或其他算法得到最佳的 parameters：$[w_0,w_1,w_2,...,w_j,...,b]^T$，並計算出這組參數確定的 function——$f^*$對 training data 和 testing data 上的 error 值，直到找到那個使 testing data 的 error 最小的 λ，(這裡一開始 λ=0，就是沒有使用 regularization 時的 loss function)

  注：引入評價$f^*$的 error 機制，令 error=$\frac{1}{n}\sum\limits_{i=1}^n|\widehat{y}^i-y^i|$，分別計算該$f^*$對 training data 和 testing data(more important)的$error(f^*)$大小

  > 先設定 λ->確定 loss function->找到使 loss 最小的$[w_0,w_1,w_2,...,w_j,...,b]^T$->確定 function->計算 error->重新設定新的 λ 重復上述步驟->使 testing data 上的 error 最小的 λ 所對應的$[w_0,w_1,w_2,...,w_j,...,b]^T$所對應的 function 就是我們能夠找到的最佳的 function

##### 本章節總結：

- Pokémon: Original CP and species almost decide the CP after evolution
- There are probably other hidden factors
- Gradient descent

  - More theory and tips in the following lectures

- Overfitting and Regularization

- We finally get average error = 11.1 on the testing data
- How about new data? Larger error? Lower error?(larger->need validation)
- Next lecture: Where does the error come from?
  - More theory about overfitting and regularization
  - The concept of validation(用來解決 new data 的 error 高於 11.1 的問題)

#### 附：Regularization(L1 L2 正則化解決 overfitting)

> Regularization -> redefine the loss function

關於 overfitting 的問題，很大程度上是由於曲線為了更好地擬合 training data 的數據，而引入了更多的高次項，使得曲線更加「蜿蜒曲折」，反而導致了對 testing data 的誤差更大

回過頭來思考，我們之前衡量 model 中某個 function 的好壞所使用的 loss function，僅引入了真實值和預測值差值的平方和這一個衡量標準；我們想要避免 overfitting 過擬合的問題，就要使得高次項對曲線形狀的影響盡可能小，因此我們要在 loss function 里引入高次項(非線性部分)的衡量標準，也就是將高次項的系數也加權放進 loss function 中，這樣可以使得訓練出來的 model 既滿足預測值和真實值的誤差小，又滿足高次項的系數盡可能小而使曲線的形狀比較穩定集中

以下圖為例，如果 loss function 僅考慮了$(\widehat{y}-y)^2$這一誤差衡量標準，那麼擬合出來的曲線就是紅色虛線部分(過擬合)，而過擬合就是所謂的 model 對 training data 過度自信, 非常完美的擬合上了這些數據, 如果具備過擬合的能力, 那麼這個方程就可能是一個比較複雜的非線性方程 , 正是因為這裡的$x^3$和$x^2$使得這條虛線能夠被彎來彎去, 所以整個模型就會特別努力地去學習作用在$x^3$和$x^2$上的 c、d 參數. **但是在這個例子里，我們期望模型要學到的卻是這條藍色的曲線. 因為它能更有效地概括數據**.而且只需要一個$y=a+bx$就能表達出數據的規律.

或者是說, 藍色的線最開始時, 和紅色線同樣也有 c、d 兩個參數, 可是最終學出來時, c 和 d 都學成了 0, 雖然藍色方程的誤差要比紅色大, 但是概括起數據來還是藍色好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/L1L2regularization.png" alt="regularization"  /></center>
這也是我們通常採用的方法，我們不可能一開始就否定高次項而直接只採用低次線性表達式的model，因為有時候真實數據的確是符合高次項非線性曲線的分布的；而如果一開始直接採用高次非線性表達式的model，就很有可能造成overfitting，在曲線偏折的地方與真實數據的誤差非常大。我們的目標應該是這樣的：

**==在無法確定真實數據分布的情況下，我們盡可能去改變 loss function 的評價標準==**

- **我們的 model 的表達式要盡可能的複雜，包含盡可能多的參數和盡可能多的高次非線性項；**
- **但是我們的 loss function 又有能力去控制這條曲線的參數和形狀，使之不會出現 overfitting 過擬合的現象；**
- **在真實數據滿足高次非線性曲線分布的時候，loss function 控制訓練出來的高次項的系數比較大，使得到的曲線比較彎折起伏；**
- **在真實數據滿足低次線性分布的時候，loss function 控制訓練出來的高次項的系數比較小甚至等於 0，使得到的曲線接近 linear 分布**

那我們如何保證能學出來這樣的參數呢? 這就是 L1 L2 正規化出現的原因.

之前的 loss function 僅考慮了$(\widehat{y}-y)^2$這一誤差衡量標準，而**L1 L2 正規化**就是在這個 loss function 的後面多加了一個東西，即 model 中跟高次項系數有關的表達式；

- L1 正規化即加上$λ\sum |w_j|$這一項，loss function 變成$L=\sum\limits_{i=1}^n(\widehat{y}^i-y^i)^2+\lambda\sum\limits_{j}|w_j|$，即 n 個 training data 里的數據的真實值與預測值差值的平方和加上 λ 權重下的 model 表達式中所有項系數的絕對值之和

- L2 正規化即加上$\lambda\sum(w_j)^2$這一項，loss function 變成$L=\sum\limits_{i=1}^n(\widehat{y}^i-y^i)^2+\lambda\sum\limits_{j}(w_j)^2$，即 n 個 training data 里的數據的真實值與預測值差值的平方和加上 λ 權重下的 model 表達式中所有項系數的平方和

相對來說，L2 要更穩定一些，L1 的結果則不那麼穩定，如果用 p 表示正規化程度，上面兩式可總結如下：$L=\sum\limits_{i=1}^n(\widehat{y}^i-y^i)^2+\lambda\sum\limits_{j}(w_j)^p$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/L1-L2.png" alt="L1-L2"  /></center>
