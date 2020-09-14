# Backpropagation

> Backpropagation(反向傳播)，就是告訴我們用gradient descent來train一個neural network的時候該怎麼做，它只是求微分的一種方法，而不是一種新的算法

#### Gradient Descent

gradient descent的使用方法，跟前面講到的linear Regression或者是Logistic Regression是一模一樣的，唯一的區別就在於當它用在neural network的時候，network parameters $\theta=w_1,w_2,...,b_1,b_2,...$裡面可能會有將近million個參數

所以現在最大的困難是，如何有效地把這個近百萬維的vector給計算出來，這就是Backpropagation要做的事情，所以**Backpropagation並不是一個和gradient descent不同的training的方法，它就是gradient descent，它只是一個比較有效率的算法**，讓你在計算這個gradient的vector的時候更有效率

#### Chain Rule

Backpropagation裡面並沒有什麼高深的數學，你唯一需要記得的就只有Chain Rule(鏈式法則)

對整個neural network，我們定義了一個loss function：$L(\theta)=\sum\limits_{n=1}^N l^n(\theta)$，它等於所有training data的loss之和

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-loss.png" width="50%;" /></center>
我們把training data里任意一個樣本點$x^n$代到neural network裡面，它會output一個$y^n$，我們把這個output跟樣本點本身的label標注的target $\hat{y}^n$作cross entropy，這個**交叉熵定義了output $y^n$和target $\hat{y}^n$之間的距離$l^n(\theta)$**，如果cross entropy比較大的話，說明output和target之間距離很遠，這個network的parameter的loss是比較大的，反之則說明這組parameter是比較好的

然後summation over所有training data的cross entropy $l^n(\theta)$，得到total loss $L(\theta)$，這就是我們的loss function，用這個$L(\theta)$對某一個參數w做偏微分，表達式如下：
$$
\frac{\partial L(\theta)}{\partial w}=\sum\limits_{n=1}^N\frac{\partial l^n(\theta)}{\partial w}
$$
這個表達式告訴我們，只需要考慮如何計算對某一筆data的$\frac{\partial l^n(\theta)}{\partial w}$，再將所有training data的cross entropy對參數w的偏微分累計求和，就可以把total loss對某一個參數w的偏微分給計算出來

我們先考慮某一個neuron，先拿出上圖中被紅色三角形圈住的neuron，假設只有兩個input $x_1,x_2$，通過這個neuron，我們先得到$z=b+w_1 x_1+w_2 x_2$，然後經過activation function從這個neuron中output出來，作為後續neuron的input，再經過了非常非常多的事情以後，會得到最終的output $y_1,y_2$

現在的問題是這樣：$\frac{\partial l}{\partial w}$該怎麼算？按照chain rule，可以把它拆分成兩項，$\frac{\partial l}{\partial w}=\frac{\partial z}{\partial w} \frac{\partial l}{\partial z}$，這兩項分別去把它計算出來。前面這一項是比較簡單的，後面這一項是比較複雜的

計算前面這一項$\frac{\partial z}{\partial w}$的這個process，我們稱之為Forward pass；而計算後面這項$\frac{\partial l}{\partial z}$的process，我們稱之為Backward pass

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-forward-backward.png" width="50%;" /></center>
#### Forward pass

先考慮$\frac{\partial z}{\partial w}$這一項，完全可以秒算出來，$\frac{\partial z}{\partial w_1}=x_1 ,\ \frac{\partial z}{\partial w_2}=x_2$

它的規律是這樣的：==**求$\frac{\partial z}{\partial w}$，就是看w前面連接的input是什麼，那微分後的$\frac{\partial z}{\partial w}$值就是什麼**==，因此只要計算出neural network裡面每一個neuron的output就可以知道任意的z對w的偏微分

- 比如input layer作為neuron的輸入時，$w_1$前面連接的是$x_1$，所以微分值就是$x_1$；$w_2$前面連接的是$x_2$，所以微分值就是$x_2$
- 比如hidden layer作為neuron的輸入時，那該neuron的input就是前一層neuron的output，於是$\frac{\partial z}{\partial w}$的值就是前一層的z經過activation function之後輸出的值(下圖中的數據是假定activation function為sigmoid function得到的)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/forward-pass.png" width="50%;" /></center>
#### Backward pass

再考慮$\frac{\partial l}{\partial z}$這一項，它是比較複雜的，這裡我們依舊假設activation function是sigmoid function

##### 公式推導

我們的z通過activation function得到a，這個neuron的output是$a=\sigma(z)$，接下來這個a會乘上某一個weight $w_3$，再加上其它一大堆的value得到$z'$，它是下一個neuron activation function的input，然後a又會乘上另一個weight $w_4$，再加上其它一堆value得到$z''$，後面還會發生很多很多其他事情，不過這裡我們就只先考慮下一步會發生什麼事情：
$$
\frac{\partial l}{\partial z}=\frac{\partial a}{\partial z} \frac{\partial l}{\partial a}
$$
這裡的$\frac{\partial a}{\partial z}$實際上就是activation function的微分(在這裡就是sigmoid function的微分)，接下來的問題是$\frac{\partial l}{\partial a}$應該長什麼樣子呢？a會影響$z'$和$z''$，而$z'$和$z''$會影響$l$，所以通過chain rule可以得到
$$
\frac{\partial l}{\partial a}=\frac{\partial z'}{\partial a} \frac{\partial l}{\partial z'}+\frac{\partial z''}{\partial a} \frac{\partial l}{\partial z''}
$$
這裡的$\frac{\partial z'}{\partial a}=w_3$，$\frac{\partial z''}{\partial a}=w_4$，那$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$又該怎麼算呢？這裡先假設我們已經通過某種方法把$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$這兩項給算出來了，然後回過頭去就可以把$\frac{\partial l}{\partial z}$給輕易地算出來
$$
\frac{\partial l}{\partial z}=\frac{\partial a}{\partial z} \frac{\partial l}{\partial a}=\sigma'(z)[w_3 \frac{\partial l}{\partial z'}+w_4 \frac{\partial l}{\partial z''}]
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/backward-pass.png" width="50%;" /></center>
##### 另一個觀點

這個式子還是蠻簡單的，然後，我們可以從另外一個觀點來看待這個式子

你可以想象說，現在有另外一個neuron，它不在我們原來的network裡面，在下圖中它被畫成三角形，這個neuron的input就是$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$，那input $\frac{\partial l}{\partial z'}$就乘上$w_3$，input $\frac{\partial l}{\partial z''}$就乘上$w_4$，它們兩個相加再乘上activation function的微分 $\sigma'(z)$，就可以得到output $\frac{\partial l}{\partial z}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/backward-neuron.png" width="50%;" /></center>
這張圖描述了一個新的「neuron」，它的含義跟圖下方的表達式是一模一樣的，作這張圖的目的是為了方便理解

值得注意的是，這裡的$\sigma'(z)$是一個constant常數，它並不是一個function，因為z其實在計算forward pass的時候就已經被決定好了，z是一個固定的值

所以這個neuron其實跟我們之前看到的sigmoid function是不一樣的，它並不是把input通過一個non-linear進行轉換，而是直接把input乘上一個constant $\sigma'(z)$，就得到了output，因此這個neuron被畫成三角形，代表它跟我們之前看到的圓形的neuron的運作方式是不一樣的，它是直接乘上一個constant(這裡的三角形有點像電路里的運算放大器op-amp，它也是乘上一個constant)

##### 兩種情況

ok，現在我們最後需要解決的問題是，怎麼計算$\frac{\partial l}{\partial z'}$和$\frac{\partial l}{\partial z''}$這兩項，假設有兩個不同的case：

###### case 1：Output Layer

假設藍色的這個neuron已經是hidden layer的最後一層了，也就是說連接在$z'$和$z''$後的這兩個紅色的neuron已經是output layer，它的output就已經是整個network的output了，這個時候計算就比較簡單
$$
\frac{\partial l}{\partial z'}=\frac{\partial y_1}{\partial z'} \frac{\partial l}{\partial y_1}
$$
其中$\frac{\partial y_1}{\partial z'}$就是output layer的activation function (softmax) 對$z'$的偏微分

而$\frac{\partial l}{\partial y_1}$就是loss對$y_1$的偏微分，它取決於你的loss function是怎麼定義的，也就是你的output和target之間是怎麼evaluate的，你可以用cross entropy，也可以用mean square error，用不同的定義，$\frac{\partial l}{\partial y_1}$的值就不一樣

這個時候，你就已經可以把$l$對$w_1$和$w_2$的偏微分$\frac{\partial l}{\partial w_1}$、$\frac{\partial l}{\partial w_2}$算出來了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-output-layer.png" width="50%;" /></center>
###### Case 2：Not Output Layer

假設現在紅色的neuron並不是整個network的output，那$z'$經過紅色neuron的activation function得到$a'$，然後output $a'$和$w_5$、$w_6$相乘並加上一堆其他東西分別得到$z_a$和$z_b$，如下圖所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/not-output-layer.png" width="50%;" /></center>
根據之前的推導證明類比，如果知道$\frac{\partial l}{\partial z_a}$和$\frac{\partial l}{\partial z_b}$，我們就可以計算$\frac{\partial l}{\partial z'}$，如下圖所示，借助運算放大器的輔助理解，將$\frac{\partial l}{\partial z_a}$乘上$w_5$和$\frac{\partial l}{\partial z_b}$乘上$w_6$的值加起來再通過op-amp，乘上放大系數$\sigma'(z')$，就可以得到output $\frac{\partial l}{\partial z'}$
$$
\frac{\partial l}{\partial z'}=\sigma'(z')[w_5 \frac{\partial l}{\partial z_a} + w_6 \frac{\partial l}{\partial z_b}]
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-not-output-layer.png" width="50%;" /></center>
知道$z'$和$z''$就可以知道$z$，知道$z_a$和$z_b$就可以知道$z'$，...... ，現在這個過程就可以反復進行下去，直到找到output layer，我們可以算出確切的值，然後再一層一層反推回去

你可能會想說，這個方法聽起來挺讓人崩潰的，每次要算一個微分的值，都要一路往後走，一直走到network的output，如果寫成表達式的話，一層一層往後展開，感覺會是一個很可怕的式子，但是！實際上並不是這個樣子做的

你只要換一個方向，從output layer的$\frac{\partial l}{\partial z}$開始算，你就會發現它的運算量跟原來的network的Feedforward path其實是一樣的

假設現在有6個neuron，每一個neuron的activation function的input分別是$z_1$、$z_2$、$z_3$、$z_4$、$z_5$、$z_6$，我們要計算$l$對這些$z$的偏微分，按照原來的思路，我們想要知道$z_1$的偏微分，就要去算$z_3$和$z_4$的偏微分，想要知道$z_3$和$z_4$的偏微分，就又要去計算兩遍$z_5$和$z_6$的偏微分，因此如果我們是從$z_1$、$z_2$的偏微分開始算，那就沒有效率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/input-z.png" width="50%;" /></center>
但是，如果你反過來先去計算$z_5$和$z_6$的偏微分的話，這個process，就突然之間變得有效率起來了，我們先去計算$\frac{\partial l}{\partial z_5}$和$\frac{\partial l}{\partial z_6}$，然後就可以算出$\frac{\partial l}{\partial z_3}$和$\frac{\partial l}{\partial z_4}$，最後就可以算出$\frac{\partial l}{\partial z_1}$和$\frac{\partial l}{\partial z_2}$，而這一整個過程，就可以轉化為op-amp運算放大器的那張圖

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-op-amp.png" width="50%;" /></center>
這裡每一個op-amp的放大系數就是$\sigma'(z_1)$、$\sigma'(z_2)$、$\sigma'(z_3)$、$\sigma'(z_4)$，所以整一個流程就是，先快速地計算出$\frac{\partial l}{\partial z_5}$和$\frac{\partial l}{\partial z_6}$，然後再把這兩個偏微分的值乘上路徑上的weight匯集到neuron上面，再通過op-amp的放大，就可以得到$\frac{\partial l}{\partial z_3}$和$\frac{\partial l}{\partial z_4}$這兩個偏微分的值，再讓它們乘上一些weight，並且通過一個op-amp，就得到$\frac{\partial l}{\partial z_1}$和$\frac{\partial l}{\partial z_2}$這兩個偏微分的值，這樣就計算完了，這個步驟，就叫做Backward pass

在做Backward pass的時候，實際上的做法就是建另外一個neural network，本來正向neural network裡面的activation function都是sigmoid function，而現在計算Backward pass的時候，就是建一個反向的neural network，它的activation function就是一個運算放大器op-amp，每一個反向neuron的input是loss $l$對後面一層layer的$z$的偏微分$\frac{\partial l}{\partial z}$，output則是loss $l$對這個neuron的$z$的偏微分$\frac{\partial l}{\partial z}$，做Backward pass就是通過這樣一個反向neural network的運算，把loss $l$對每一個neuron的$z$的偏微分$\frac{\partial l}{\partial z}$都給算出來

注：如果是正向做Backward pass的話，實際上每次計算一個$\frac{\partial l}{\partial z}$，就需要把該neuron後面所有的$\frac{\partial l}{\partial z}$都給計算一遍，會造成很多不必要的重復運算，如果寫成code的形式，就相當於調用了很多次重復的函數；而如果是反向做Backward pass，實際上就是把這些調用函數的過程都變成調用「值」的過程，因此可以直接計算出結果，而不需要佔用過多的堆棧空間

#### Summary

最後，我們來總結一下Backpropagation是怎麼做的

**Forward pass**，每個neuron的activation function的output，就是它所連接的weight的$\frac{\partial z}{\partial w}$

**Backward pass**，建一個與原來方向相反的neural network，它的三角形neuron的output就是$\frac{\partial l}{\partial z}$

把通過forward pass得到的$\frac{\partial z}{\partial w}$和通過backward pass得到的$\frac{\partial l}{\partial z}$乘起來就可以得到$l$對$w$的偏微分$\frac{\partial l}{\partial w}$
$$
\frac{\partial l}{\partial w} = \frac{\partial z}{\partial w}|_{forward\ pass} \cdot \frac{\partial l}{\partial z}|_{backward \ pass}
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bp-summary.png" width="50%;" /></center>
