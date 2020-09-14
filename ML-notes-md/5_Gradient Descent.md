# Gradient Descent

#### Review

前面預測寶可夢 cp 值的例子里，已經初步介紹了 Gradient Descent 的用法：

In step 3,​ we have to solve the following optimization problem:

$\theta^{*}=\arg \underset{\theta}{\min} L(\theta) \quad$

L : loss function
$\theta:$ parameters(上標表示第幾組參數，下標表示這組參數中的第幾個參數)

假設$\theta$是參數的集合：Suppose that $\theta$ has two variables $\left\{\theta_{1}, \theta_{2}\right\}$

隨機選取一組起始的參數：Randomly start at $\theta^{0}=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right] \quad$

計算$\theta$處的梯度 gradient：$\nabla L(\theta)=\left[\begin{array}{l}{\partial L\left(\theta_{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}\right) / \partial \theta_{2}}\end{array}\right]$

$\left[\begin{array}{l}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]=\left[\begin{array}{l}{\theta_{1}^{0}} \\ {\theta_{2}^{0}}\end{array}\right]-\eta\left[\begin{array}{l}{\partial L\left(\theta_{1}^{0}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{0}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{1}=\theta^{0}-\eta \nabla L\left(\theta^{0}\right)$

$\left[\begin{array}{c}{\theta_{1}^{2}} \\ {\theta_{2}^{2}}\end{array}\right]=\left[\begin{array}{c}{\theta_{1}^{1}} \\ {\theta_{2}^{1}}\end{array}\right]-\eta\left[\begin{array}{c}{\partial L\left(\theta_{1}^{1}\right) / \partial \theta_{1}} \\ {\partial L\left(\theta_{2}^{1}\right) / \partial \theta_{2}}\end{array}\right] \Rightarrow \theta^{2}=\theta^{1}-\eta \nabla L\left(\theta^{1}\right)$

下圖是將 gradient descent 在投影到二維坐標系中可視化的樣子，圖上的每一個點都是$(\theta_1,\theta_2,loss)$在該平面的投影

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-descent-visualize.png" width="60%;"></center>

紅色箭頭是指在$(\theta_1,\theta_2)$這點的梯度，梯度方向即箭頭方向(從低處指向高處)，梯度大小即箭頭長度(表示在$\theta^i$點處最陡的那條切線的導數大小，該方向也是梯度上升最快的方向)

藍色曲線代表實際情況下參數$\theta_1$和$\theta_2$的更新過程圖，每次更新沿著藍色箭頭方向 loss 會減小，藍色箭頭方向與紅色箭頭方向剛好相反，代表著梯度下降的方向

因此，==在整個 gradient descent 的過程中，梯度不一定是遞減的(紅色箭頭的長度可以長短不一)，但是沿著梯度下降的方向，函數值 loss 一定是遞減的，且當 gradient=0 時，loss 下降到了局部最小值，總結：梯度下降法指的是函數值 loss 隨梯度下降的方向減小==

初始隨機在三維坐標系中選取一個點，這個三維坐標系的三個變量分別為$(\theta_1,\theta_2,loss)$，我們的目標是找到最小的那個 loss 也就是三維坐標系中高度最低的那個點，而 gradient 梯度可以理解為高度上升最快的那個方向，它的反方向就是梯度下降最快的那個方向，於是每次 update 沿著梯度反方向，update 的步長由梯度大小和 learning rate 共同決定，當某次 update 完成後，該點的 gradient=0，說明到達了局部最小值

下面是關於 gradient descent 的一點思考：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-byhand.png" /></center>

#### Learning rate 存在的問題

gradient descent 過程中，影響結果的一個很關鍵的因素就是 learning rate 的大小

- 如果 learning rate 剛剛好，就可以像下圖中紅色線段一樣順利地到達到 loss 的最小值
- 如果 learning rate 太小的話，像下圖中的藍色線段，雖然最後能夠走到 local minimal 的地方，但是它可能會走得非常慢，以至於你無法接受
- 如果 learning rate 太大，像下圖中的綠色線段，它的步伐太大了，它永遠沒有辦法走到特別低的地方，可能永遠在這個「山谷」的口上振蕩而無法走下去
- 如果 learning rate 非常大，就會像下圖中的黃色線段，一瞬間就飛出去了，結果會造成 update 參數以後，loss 反而會越來越大(這一點在上次的 demo 中有體會到，當 lr 過大的時候，每次更新 loss 反而會變大)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/learning-rate.png" width="60%;"></center>

當參數有很多個的時候(>3)，其實我們很難做到將 loss 隨每個參數的變化可視化出來(因為最多只能可視化出三維的圖像，也就只能可視化三維參數)，但是我們可以把 update 的次數作為唯一的一個參數，將 loss 隨著 update 的增加而變化的趨勢給可視化出來(上圖右半部分)

所以做 gradient descent 一個很重要的事情是，==要把不同的 learning rate 下，loss 隨 update 次數的變化曲線給可視化出來==，它可以提醒你該如何調整當前的 learning rate 的大小，直到出現穩定下降的曲線

#### Adaptive Learning rates

顯然這樣手動地去調整 learning rates 很麻煩，因此我們需要有一些自動調整 learning rates 的方法

##### 最基本、最簡單的大原則是：learning rate 通常是隨著參數的 update 越來越小的

因為在起始點的時候，通常是離最低點是比較遠的，這時候步伐就要跨大一點；而經過幾次 update 以後，會比較靠近目標，這時候就應該減小 learning rate，讓它能夠收斂在最低點的地方

舉例：假設到了第 t 次 update，此時$\eta^t=\eta/ \sqrt{t+1}$

這種方法使所有參數以同樣的方式同樣的 learning rate 進行 update，而最好的狀況是每個參數都給他不同的 learning rate 去 update

##### Adagrad

> Divide the learning rate of each parameter by the root mean square(方均根) of its previous derivatives

Adagrad 就是將不同參數的 learning rate 分開考慮的一種算法(adagrad 算法 update 到後面速度會越來越慢，當然這只是 adaptive 算法中最簡單的一種)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adagrad-definition.png" width="60%;"></center>

這裡的 w 是 function 中的某個參數，t 表示第 t 次 update，$g^t$表示 Loss 對 w 的偏微分，而$\sigma^t$是之前所有 Loss 對 w 偏微分的方均根(根號下的平方均值)，這個值對每一個參數來說都是不一樣的

$$
\begin{equation}
\begin{split}
&Adagrad\\
&w^1=w^0-\frac{\eta^0}{\sigma^0}\cdot g^0 \ \ \ \sigma^0=\sqrt{(g^0)^2} \\
&w^2=w^1-\frac{\eta^1}{\sigma^1}\cdot g^1 \ \ \ \sigma^1=\sqrt{\frac{1}{2}[(g^0)^2+(g^1)^2]} \\
&w^3=w^2-\frac{\eta2}{\sigma^2}\cdot g^2 \ \ \ \sigma^2=\sqrt{\frac{1}{3}[(g^0)^2+(g^1)^2+(g^2)^2]} \\
&... \\
&w^{t+1}=w^t-\frac{\eta^t}{\sigma^t}\cdot g^t \ \ \ \sigma^t=\sqrt{\frac{1}{1+t}\sum\limits_{i=0}^{t}(g^i)^2}
\end{split}
\end{equation}
$$

由於$\eta^t$和$\sigma^t$中都有一個$\sqrt{\frac{1}{1+t}}$的因子，兩者相消，即可得到 adagrad 的最終表達式：

$w^{t+1}=w^t-\frac{\eta}{\sum\limits_{i=0}^t(g^i)^2}\cdot g^t$

##### Adagrad 的 contradiction 解釋

Adagrad 的表達式$w^{t+1}=w^t-\frac{\eta}{\sum\limits_{i=0}^t(g^i)^2}\cdot g^t$裡面有一件很矛盾的事情：

我們在做 gradient descent 的時候，希望的是當梯度值即微分值$g^t$越大的時候(此時斜率越大，還沒有接近最低點)更新的步伐要更大一些，但是 Adagrad 的表達式中，分母表示梯度越大步伐越小，分子卻表示梯度越大步伐越大，兩者似乎相互矛盾

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adagrad-contradiction.png" width="60%;"></center>

在一些 paper 里是這樣解釋的：Adagrad 要考慮的是，這個 gradient 有多 surprise，即反差有多大，假設 t=4 的時候$g^4$與前面的 gradient 反差特別大，那麼$g^t$與$\sqrt{\frac{1}{t+1}\sum\limits_{i=0}^t(g^i)^2}$之間的大小反差就會比較大，它們的商就會把這一反差效果體現出來

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adagrad-reason.png" width="60%" /></center>

**gradient 越大，離最低點越遠這件事情在有多個參數的情況下是不一定成立的**

如下圖所示，w1 和 w2 分別是 loss function 的兩個參數，loss 的值投影到該平面中以顏色深度表示大小，分別在 w2 和 w1 處垂直切一刀(這樣就只有另一個參數的 gradient 會變化)，對應的情況為右邊的兩條曲線，可以看出，比起 a 點，c 點距離最低點更近，但是它的 gradient 卻越大

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adagrad-cross-parameters.png" width="60%;"></center>
實際上，對於一個二次函數$y=ax^2+bx+c$來說，最小值點的$x=-\frac{b}{2a}$，而對於任意一點$x_0$，它邁出最好的步伐長度是$|x_0+\frac{b}{2a}|=|\frac{2ax_0+b}{2a}|$(這樣就一步邁到最小值點了)，聯繫該函數的一階和二階導數$y'=2ax+b$、$y''=2a$，可以發現the best step is $|\frac{y'}{y''}|$，也就是說他不僅跟一階導數(gradient)有關，還跟二階導師有關，因此我們可以通過這種方法重新比較上面的a和c點，就可以得到比較正確的答案

再來回顧 Adagrad 的表達式：$w^{t+1}=w^t-\frac{\eta}{\sum\limits_{i=0}^t(g^i)^2}\cdot g^t$

$g^t$就是一次微分，而分母中的$\sum\limits_{i=0}^t(g^i)^2$反映了二次微分的大小，所以 Adagrad 想要做的事情就是，在不增加任何額外運算的前提下，想辦法去估測二次微分的值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/adagrad-second-derivative.png" width="60%;"></center>

#### Stochastic Gradicent Descent

隨機梯度下降的方法可以讓訓練更快速，傳統的 gradient descent 的思路是看完所有的樣本點之後再構建 loss function，然後去 update 參數；而 stochastic gradient descent 的做法是，看到一個樣本點就 update 一次，因此它的 loss function 不是所有樣本點的 error 平方和，而是這個隨機樣本點的 error 平方

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/stochastic-gradient-descent.png" width="60%;" /></center>
stochastic gradient descent與傳統gradient descent的效果對比如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/stochastic-visualize.png" width="60%;" /></center>

#### Feature Scaling

##### 概念介紹

特徵縮放，當多個特徵的分布範圍很不一樣時，最好將這些不同 feature 的範圍縮放成一樣

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/feature-scaling.png" width="60%;" /></center>

##### 原理解釋

$y=b+w_1x_1+w_2x_2$，假設 x1 的值都是很小的，比如 1,2...；x2 的值都是很大的，比如 100,200...

此時去畫出 loss 的 error surface，如果對 w1 和 w2 都做一個同樣的變動$\Delta w$，那麼 w1 的變化對 y 的影響是比較小的，而 w2 的變化對 y 的影響是比較大的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/feature-scaling-example.png" width="60%;"></center>

左邊的 error surface 表示，w1 對 y 的影響比較小，所以 w1 對 loss 是有比較小的偏微分的，因此在 w1 的方向上圖像是比較平滑的；w2 對 y 的影響比較大，所以 w2 對 loss 的影響比較大，因此在 w2 的方向上圖像是比較 sharp 的

如果 x1 和 x2 的值，它們的 scale 是接近的，那麼 w1 和 w2 對 loss 就會有差不多的影響力，loss 的圖像接近於圓形，那這樣做對 gradient descent 有什麼好處呢？

##### 對 gradient decent 的幫助

之前我們做的 demo 已經表明瞭，對於這種長橢圓形的 error surface，如果不使用 Adagrad 之類的方法，是很難搞定它的，因為在像 w1 和 w2 這樣不同的參數方向上，會需要不同的 learning rate，用相同的 lr 很難達到最低點

如果有 scale 的話，loss 在參數 w1、w2 平面上的投影就是一個正圓形，update 參數會比較容易

而且 gradient descent 的每次 update 並不都是向著最低點走的，每次 update 的方向是順著等高線的方向(梯度 gradient 下降的方向)，而不是徑直走向最低點；但是當經過對 input 的 scale 使 loss 的投影是一個正圓的話，不管在這個區域的哪一個點，它都會向著圓心走。因此 feature scaling 對參數 update 的效率是有幫助的

##### 如何做 feature scaling

假設有 R 個 example(上標 i 表示第 i 個樣本點)，$x^1,x^2,x^3,...,x^r,...x^R$，每一筆 example，它裡面都有一組 feature(下標 j 表示該樣本點的第 j 個特徵)

對每一個 demension i，都去算出它的平均值 mean=$m_i$，以及標準差 standard deviation=$\sigma_i$

對第 r 個 example 的第 i 個 component，減掉均值，除以標準差，即$x_i^r=\frac{x_i^r-m_i}{\sigma_i}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/feature-scaling-method.png" width="60%;"/></center>

說了那麼多，實際上就是==將每一個參數都歸一化成標準正態分布，即$f(x_i)=\frac{1}{\sqrt{2\pi}}e^{-\frac{x_i^2}{2}}$==，其中$x_i$表示第 i 個參數

#### Gradient Descent 的理論基礎

##### Taylor Series

泰勒表達式：$h(x)=\sum\limits_{k=0}^\infty \frac{h^{(k)}(x_0)}{k!}(x-x_0)^k=h(x_0)+h'(x_0)(x-x_0)+\frac{h''(x_0)}{2!}(x-x_0)^2+...$

When x is close to $x_0$ : $h(x)≈h(x_0)+h'(x_0)(x-x_0)$

同理，對於二元函數，when x and y is close to $x_0$ and $y_0$：

$h(x,y)≈h(x_0,y_0)+\frac{\partial h(x_0,y_0)}{\partial x}(x-x_0)+\frac{\partial h(x_0,y_0)}{\partial y}(y-y_0)$

##### 從泰勒展開式推導出 gradient descent

對於 loss 圖像上的某一個點(a,b)，如果我們想要找這個點附近 loss 最小的點，就可以用泰勒展開的思想

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/taylor-visualize.png" width="60%;" /></center>

假設用一個 red circle 限定點的範圍，這個圓足夠小以滿足泰勒展開的精度，那麼此時我們的 loss function 就可以化簡為：

$L(\theta)≈L(a,b)+\frac{\partial L(a,b)}{\partial \theta_1}(\theta_1-a)+\frac{\partial L(a,b)}{\partial \theta_2}(\theta_2-b)$

令$s=L(a,b)$，$u=\frac{\partial L(a,b)}{\partial \theta_1}$，$v=\frac{\partial L(a,b)}{\partial \theta_2}$

則$L(\theta)≈s+u\cdot (\theta_1-a)+v\cdot (\theta_2-b)$

假定 red circle 的半徑為 d，則有限制條件：$(\theta_1-a)^2+(\theta_2-b)^2≤d^2$

此時去求$L(\theta)_{min}$，這裡有個小技巧，把$L(\theta)$轉化為兩個向量的乘積：$u\cdot (\theta_1-a)+v\cdot (\theta_2-b)=(u,v)\cdot (\theta_1-a,\theta_2-b)=(u,v)\cdot (\Delta \theta_1,\Delta \theta_2)$

觀察圖形可知，當向量$(\theta_1-a,\theta_2-b)$與向量$(u,v)$反向，且剛好到達 red circle 的邊緣時(用$\eta$去控制向量的長度)，$L(\theta)$最小

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/taylor.png" width="60%;"/></center>

$(\theta_1-a,\theta_2-b)$實際上就是$(\Delta \theta_1,\Delta \theta_2)$，於是$L(\theta)$局部最小值對應的參數為中心點減去 gradient 的加權

$$
\begin{bmatrix}
\Delta \theta_1 \\
\Delta \theta_2
\end{bmatrix}=
-\eta
\begin{bmatrix}
u \\
v
\end{bmatrix}=>
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}=
\begin{bmatrix}
a\\
b
\end{bmatrix}-\eta
\begin{bmatrix}
u\\
v
\end{bmatrix}=
\begin{bmatrix}
a\\
b
\end{bmatrix}-\eta
\begin{bmatrix}
\frac{\partial L(a,b)}{\partial \theta_1}\\
\frac{\partial L(a,b)}{\partial \theta_2}
\end{bmatrix}
$$

這就是 gradient descent 在數學上的推導，注意它的重要前提是，給定的那個紅色圈圈的範圍要足夠小，這樣泰勒展開給我們的近似才會更精確，而$\eta$的值是與圓的半徑成正比的，因此理論上 learning rate 要無窮小才能夠保證每次 gradient descent 在 update 參數之後的 loss 會越來越小，於是當 learning rate 沒有設置好，泰勒近似不成立，就有可能使 gradient descent 過程中的 loss 沒有越來越小

當然泰勒展開可以使用二階、三階乃至更高階的展開，但這樣會使得運算量大大增加，反而降低了運行效率

#### Gradient Descent 的限制

之前已經討論過，gradient descent 有一個問題是它會停在 local minima 的地方就停止 update 了

事實上還有一個問題是，微分值是 0 的地方並不是只有 local minima，settle point 的微分值也是 0

以上都是理論上的探討，到了實踐的時候，其實當 gradient 的值接近於 0 的時候，我們就已經把它停下來了，但是微分值很小，不見得就是很接近 local minima，也有可能像下圖一樣在一個高原的地方

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-limits.png" width="60%;"/></center>

綜上，==**gradient descent 的限制是，它在 gradient 即微分值接近於 0 的地方就會停下來，而這個地方不一定是 global minima，它可能是 local minima，可能是 saddle point 鞍點，甚至可能是一個 loss 很高的 plateau 平緩高原**==
