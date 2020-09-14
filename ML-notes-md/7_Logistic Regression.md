# Logistic Regression

#### Review

在 classification 這一章節，我們討論了如何通過樣本點的均值$u$和協方差$\Sigma$來計算$P(C_1),P(C_2),P(x|C_1),P(x|C_2)$，進而利用$P(C_1|x)=\frac{P(C_1)P(x|C_1)}{P(C_1)P(x|C_1)+P(C_2)P(x|C_2)}$計算得到新的樣本點 x 屬於 class 1 的概率，由於是二元分類，屬於 class 2 的概率$P(C_2|x)=1-P(C_1|x)$

之後我們還推導了$P(C_1|x)=\sigma(z)=\frac{1}{1+e^{-z}}$，並且在 Gaussian 的 distribution 下考慮 class 1 和 class 2 共用$\Sigma$，可以得到一個線性的 z(其實很多其他的 Probability model 經過化簡以後也都可以得到同樣的結果)

$$
P_{w,b}(C_1|x)=\sigma(z)=\frac{1}{1+e^{-z}} \\
z=w\cdot x+b=\sum\limits_i w_ix_i+b \\
$$

這裡的 w 和 x 都是 vector，兩者的乘積是 inner product，從上式中我們可以看出，現在這個 model(function set)是受 w 和 b 控制的，因此我們不必要再去像前面一樣計算一大堆東西，而是用這個全新的由 w 和 b 決定的 model——**Logistic Regression(邏輯回歸)**

#### Three Steps of machine learning

##### Step 1：function set

這裡的 function set 就是 Logistic Regression——邏輯回歸

$w_i$：weight，$b$：bias，$\sigma(z)$：sigmoid function，$x_i$：input

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/activation-function.png" width="60%;" /></center>
##### Step 2：Goodness of a function

現在我們有 N 筆 Training data，每一筆 data 都要標注它是屬於哪一個 class

假設這些 Training data 是從我們定義的 posterior Probability 中產生的(後置概率，某種意義上就是概率密度函數)，而 w 和 b 就決定了這個 posterior Probability，那我們就可以去計算某一組 w 和 b 去產生這 N 筆 Training data 的概率，利用極大似然估計的思想，最好的那組參數就是有最大可能性產生當前 N 筆 Training data 分布的$w^*$和$b^*$

似然函數只需要將每一個點產生的概率相乘即可，注意，這裡假定是二元分類，class 2 的概率為 1 減去 class 1 的概率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/likelihood.png" width="60%;" /></center>
由於$L(w,b)$是乘積項的形式，為了方便計算，我們將上式做個變換：
$$
\begin{split}
&w^*,b^*=\arg \max\limits_{w,b} L(w,b)=\arg\min\limits_{w,b}(-\ln L(w,b)) \\
&\begin{equation}
\begin{split}
-\ln L(w,b)=&-\ln f_{w,b}(x^1)\\
&-\ln f_{w,b}(x^2)\\
&-\ln(1-f_{w,b}(x^3))\\
&\ -...
\end{split}
\end{equation}
\end{split}
$$
由於class 1和class 2的概率表達式不統一，上面的式子無法寫成統一的形式，為了統一格式，這裡將Logistic Regression里的所有Training data都打上0和1的標籤，即output  $\hat{y}=1$代表class 1，output  $\hat{y}=0$代表class 2，於是上式進一步改寫成：
$$
\begin{split}
-\ln L(w,b)=&-[\hat{y}^1 \ln f_{w,b}(x^1)+(1-\hat{y}^1)ln(1-f_{w,b}(x^1))]\\
&-[\hat{y}^2 \ln f_{w,b}(x^2)+(1-\hat{y}^2)ln(1-f_{w,b}(x^2))]\\
&-[\hat{y}^3 \ln f_{w,b}(x^3)+(1-\hat{y}^3)ln(1-f_{w,b}(x^3))]\\
&\ -...
\end{split}
$$

現在已經有了統一的格式，我們就可以把要 minimize 的對象寫成一個 summation 的形式：

$$
-\ln L(w,b)=\sum\limits_n -[\hat{y}^n \ln f_{w,b}(x^n)+(1-\hat{y}^n) \ln(1-f_{w,b}(x^n))]
$$

這裡$x^n$表示第 n 個樣本點，$\hat{y}^n$表示第 n 個樣本點的 class 標籤(1 表示 class 1,0 表示 class 2)，最終這個 summation 的形式，裡面其實是<u>兩個 Bernouli distribution(兩點分布)的 cross entropy(交叉熵)</u>

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cross-entropy.png" width="60%;" /></center>
假設有如上圖所示的兩個distribution p和q，它們的交叉熵就是$H(p,q)=-\sum\limits_{x} p(x) \ln (q(x))$，這也就是之前的推導中在$-\ln L(w,b)$前加一個負號的原因

cross entropy 交叉熵的含義是表達這兩個 distribution 有多接近，如果 p 和 q 這兩個 distribution 一模一樣的話，那它們算出來的 cross entropy 就是 0(詳細解釋在「信息論」中)，而這裡$f(x^n)$表示 function 的 output，$\hat{y}^n$表示預期 的 target，因此**交叉熵實際上表達的是希望這個 function 的 output 和它的 target 越接近越好**

總之，我們要找的參數實際上就是：

$$
w^*,b^*=\arg \max\limits_{w,b} L(w,b)=\arg\min\limits_{w,b}(-\ln L(w,b)=\sum\limits_n -[\hat{y}^n \ln f_{w,b}(x^n)+(1-\hat{y}^n) \ln(1-f_{w,b}(x^n))]
$$

##### step 3：Find the best function

實際上就是去找到使 loss function 即交叉熵之和最小的那組參數$w^*,b^*$就行了，這裡用 gradient descent 的方法進行運算就 ok

這裡 sigmoid function 的微分可以直接作為公式記下來：$\frac{\partial \sigma(z)}{\partial z}=\sigma(z)(1-\sigma(z))$，sigmoid 和它的微分的圖像如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/sigmoid.png" width="40%;" /></center>
先計算$-\ln L(w,b)=\sum\limits_n -[\hat{y}^n \ln f_{w,b}(x^n)+(1-\hat{y}^n) \ln(1-f_{w,b}(x^n))]$對$w_i$的偏微分，這裡$\hat{y}^n$和$1-\hat{y}^n$是常數先不用管它，只需要分別求出$\ln f_{w,b}(x^n)$和$\ln (1-f_{w,b}(x^n))$對$w_i$的偏微分即可，整體推導過程如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-contribute.png" width="60%;" /></center>
將得到的式子進行進一步化簡，可得：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-simple.png" width="60%;" /></center>
我們發現最終的結果竟然異常的簡潔，gradient descent每次update只需要做：
$$
w_i=w_i-\eta \sum\limits_{n}-(\hat{y}^n-f_{w,b}(x^n))x_i^n
$$
那這個式子到底代表著什麼意思呢？現在你的update取決於三件事：

- learning rate，是你自己設定的
- $x_i$，來自於 data
- $\hat{y}^n-f_{w,b}(x^n)$，代表 function 的 output 跟理想 target 的差距有多大，如果離目標越遠，update 的步伐就要越大

#### Logistic Regression V.s. Linear Regression

我們可以把邏輯回歸和之前將的線性回歸做一個比較

##### compare in step1

Logistic Regression 是把每一個 feature $x_i$加權求和，加上 bias，再通過 sigmoid function，當做 function 的 output

因為 Logistic Regression 的 output 是通過 sigmoid function 產生的，因此一定是介於 0~1 之間；而 linear Regression 的 output 並沒有通過 sigmoid function，所以它可以是任何值

##### compare in step2

在 Logistic Regression 中，我們定義的 loss function，即要去 minimize 的對象，是所有 example(樣本點)的 output( $f(x^n)$ )和實際 target( $\hat{y}^n$ )在 Bernoulli distribution(兩點分布)下的 cross entropy(交叉熵)總和

**交叉熵**的描述：這裡把$f(x^n)$和$\hat{y}^n$各自<u>看做</u>是一個**Bernoulli distribution(兩點分布)**，那它們的 cross entropy $l(f(x^n),\hat{y}^n)=-[\hat{y}^n \ln f(x^n)+(1-\hat{y}^n) \ln (1-f(x^n))]$之和，就是我們要去 minimize 的對象，直觀來講，就是**希望 function 的 output $f(x^n)$和它的 target $\hat{y}^n$越接近越好**

注：這裡的「看做」只是為了方便理解和計算，並不是真的做出它們是兩點分布的假設

而在 linear Regression 中，loss function 的定義相對比較簡單，就是單純的 function 的 output( $f(x^n)$ )和實際 target( $\hat{y}^n$ )在數值上的平方和的均值

這裡可能會有一個疑惑，為什麼 Logistic Regression 的 loss function 不能像 linear Regression 一樣用 square error 來表示呢？後面會有進一步的解釋

##### compare in step3

神奇的是，Logistic Regression 和 linear Regression 的$w_i$update 的方式是一模一樣的，唯一不一樣的是，Logistic Regression 的 target $\hat{y}^n$和 output $f(x^n)$都必須是在 0 和 1 之間的，而 linear Regression 的 target 和 output 的範圍可以是任意值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-linear-regression.png" width="60%;" /></center>
#### Logistic Regression + Square error？

之前提到了，為什麼 Logistic Regression 的 loss function 不能用 square error 來描述呢？我們現在來試一下這件事情，重新做一下 machine learning 的三個 step

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-square.png" width="60%;" /></center>
現在會遇到一個問題：如果第n個點的目標target是class 1，則$\hat{y}^n=1$，此時如果function的output $f_{w,b}(x^n)=1$的話，說明現在離target很接近了，$f_{w,b}(x)-\hat{y}$這一項是0，於是得到的微分$\frac{\partial L}{\partial w_i}$會變成0，這件事情是很合理的；但是當function的output $f_{w,b}(x^n)=0$的時候，說明離target還很遙遠，但是由於在step3中求出來的update表達式中有一個$f_{w,b}(x^n)$，因此這個時候也會導致得到的微分$\frac{\partial L}{\partial w_i}$變成0

如果舉 class 2 的例子，得到的結果與 class 1 是一樣的

如果我們把參數的變化對 total loss 作圖的話，loss function 選擇 cross entropy 或 square error，參數的變化跟 loss 的變化情況可視化出來如下所示：(黑色的是 cross entropy，紅色的是 square error)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cross-entropy-vs-square-error.png" width="60%;" /></center>
假設中心點就是距離目標很近的地方，如果是cross entropy的話，距離目標越遠，微分值就越大，參數update的時候變化量就越大，邁出去的步伐也就越大

但當你選擇 square error 的時候，過程就會很卡，因為距離目標遠的時候，微分也是非常小的，移動的速度是非常慢的，我們之前提到過，實際操作的時候，當 gradient 接近於 0 的時候，其實就很有可能會停下來，因此使用 square error 很有可能在一開始的時候就卡住不動了，而且這裡也不能隨意地增大 learning rate，因為在做 gradient descent 的時候，你的 gradient 接近於 0，有可能離 target 很近也有可能很遠，因此不知道 learning rate 應該設大還是設小

綜上，儘管 square error 可以使用，但是會出現 update 十分緩慢的現象，而使用 cross entropy 可以讓你的 Training 更順利

#### Discriminative v.s. Generative

##### same model but different currency

Logistic Regression 的方法，我們把它稱之為 discriminative 的方法；而我們用 Gaussian 來描述 posterior Probability 這件事，我們稱之為 Generative 的方法

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/discriminative-generative.png" width="60%;" /></center>
實際上它們用的model(function set)是一模一樣的，都是$P(C_1|x)=\sigma(w\cdot x+b)$，如果是用Logistic Regression的話，可以用gradient descent的方法直接去把b和w找出來；如果是用Generative model的話，我們要先去算$u_1,u_2,\Sigma^{-1}$，然後算出b和w

你會發現用這兩種方法得到的 b 和 w 是不同的，儘管我們的 function set 是同一個，但是由於做了不同的假設，最終從同樣的 Training data 里找出來的參數會是不一樣的

在 Logistic Regression 裡面，我們**沒有做任何實質性的假設**，沒有對 Probability distribution 有任何的描述，我們就是單純地去找 b 和 w(推導過程中的假設只是便於理解和計算，對實際結果沒有影響)

而在 Generative model 裡面，我們對 Probability distribution 是**有實質性的假設**的，之前我們假設的是 Gaussian(高斯分布)，甚至假設在相互獨立的前提下是否可以是 naive bayes(樸素貝葉斯)，根據這些假設我們才找到最終的 b 和 w

哪一個假設的結果是比較好的呢？Generative model 和 discriminative model 的預測結果比較如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/generative-discriminative-visualize.png" width="60%;" /></center>
實際上Discriminative的方法常常會比Generative的方法表現得更好，這裡舉一個簡單的例子來解釋一下

##### toy example

假設總共有兩個 class，有這樣的 Training data：每一筆 data 有兩個 feature，總共有 1+4+4+4=13 筆 data

如果我們的 testing data 的兩個 feature 都是 1，憑直覺來說會認為它肯定是 class 1，但是如果用 naive bayes 的方法(樸素貝葉斯假設所有的 feature 相互獨立，方便計算)，得到的結果又是怎樣的呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/toy-example.png" width="60%;" /></center>
通過Naive bayes得到的結果竟然是這個測試點屬於class 2的可能性更大，這跟我們的直覺比起來是相反的，實際上我們直覺認為兩個feature都是1的測試點屬於class 1的可能性更大是因為我們潛意識里認為這兩個feature之間是存在某種聯繫的，但是對Naive bayes來說，它是不考慮不同dimension之間的correlation，Naive bayes認為在dimension相互獨立的前提下，class 2沒有sample出都是1的data，是因為sample的數量不夠多，如果sample夠多，它認為class 2觀察到都是1的data的可能性會比class 1要大

Naive bayes 認為從 class 2 中找到樣本點 x 的概率是 x 中第一個 feature 出現的概率與第二個 feature 出現的概率之積：$P(x|C_2)=P(x_1=1|C_2)\cdot P(x_2=1|C_2)$；但是我們的直覺告訴自己，兩個 feature 之間肯定是有某種聯繫的，$P(x|C_2)$不能夠那麼輕易地被拆分成兩個獨立的概率乘積，也就是說 Naive bayes 自作聰明地多假設了一些條件

所以，==**Generative model 和 discriminative model 的差別就在於，Generative 的 model 它有做了某些假設，假設你的 data 來自於某個概率模型；而 Discriminative 的 model 是完全不作任何假設的**==

Generative model 做的事情就是腦補，它會自己去想象一些事情，於是會做出一個和我們人類直覺想法不太一樣的判斷結果，就像 toy example 里，我們做了 naive bayes 這樣一個假設(事實上我們並不知道這兩個 feature 是否相互獨立)，於是 Naive bayes 會在 class 2 里並沒有出現過兩個 feature 都是 1 的樣本點的前提下，自己去腦補有這樣的點

通常腦補不是一件好的事情，因為你給你的 data 強加了一些它並沒有告訴你的屬性，但是在 data 很少的情況下，腦補也是有用的，discriminative model 並不是在所有的情況下都可以贏過 Generative model，discriminative model 是十分依賴於 data 的，當 data 數量不足或是 data 本身的 label 就有一些問題，那 Generative model 做一些腦補和假設，反而可以把 data 的不足或是有問題部分的影響給降到最低

在 Generative model 中，priors probabilities 和 class-dependent probabilities 是可以拆開來考慮的，以語音辨識為例，現在用的都是 neural network，是一個 discriminative 的方法，但事實上整個語音辨識的系統是一個 Generative 的 system，它的 prior probability 是某一句話被說出來的幾率，而想要 estimate 某一句話被說出來的幾率並不需要有聲音的 data，可以去互聯網上爬取大量文字，就可以計算出某一段文字出現的幾率，並不需要聲音的 data，這個就是 language model，而 class-dependent 的部分才需要聲音和文字的配合，這樣的處理可以把 prior 預測地更精確

#### Conclusion

對於分類的問題(主要是二元分類)，我們一般有兩種方法去處理問題，一種是 Generative 的方法，另一種是 Discriminative 的方法，注意到分類問題的 model 都是從貝葉斯方程出發的，即

$$
\begin{split}
P(C_i|x)&=\frac{P(C_i)P(x|C_i)}{\sum\limits_{j=1}^nP(C_j)P(x|C_j)} \ \ (1) \\
&=\sigma(z)=\frac{1}{1+e^{-z}}=\frac{1}{1+e^{-(b+\sum\limits_k w_k x_k)}} \ \ (2)
\end{split}
$$

其中分子表示屬於第 i 類的可能性，分母表示遍歷從 1 到 n 所有的類的可能性，兩種方法的區別在於：

Generative model 會假設一個帶參數的 Probability contribute，利用這個假設的概率分布函數帶入(1)中去計算$P(x|C_i)$和$P(x|C_j)$，結合極大似然估計法最終得到最優的參數以確定這個 model 的具體形式

DIscriminative model 不作任何假設，因此它無法通過假定的 Probability distribution 得到$P(x|C_i)$的表達式，因此它使用的是(2)，直接去利用交叉熵和 gradient descent 結合極大似然估計法得到最優的 b 和 w，以確定 model 的具體形式

最後，利用得到的$P(C_i|x)$與 0.5 相比較來判斷它屬於那個 class 的可能性更大

Generative model 的好處是，它對 data 的依賴並沒有像 discriminative model 那麼嚴重，在 data 數量少或者 data 本身就存在 noise 的情況下受到的影響會更小，而它還可以做到 Prior 部分與 class-dependent 部分分開處理，如果可以借助其他方式提高 Prior model 的準確率，對整一個 model 是有所幫助的(比如前面提到的語音辨識)

而 Discriminative model 的好處是，在 data 充足的情況下，它訓練出來的 model 的準確率一般是比 Generative model 要來的高的

#### Multi-class Classification

##### softmax

之前講的都是二元分類的情況，這裡討論一下多元分類問題，其原理的推導過程與二元分類基本一致

假設有三個 class：$C_1,C_2,C_3$，每一個 class 都有自己的 weight 和 bias，這裡$w_1,w_2,w_3$分布代表三個 vector，$b_1,b_2,b_3$分別代表三個 const，input x 也是一個 vector

> softmax 的意思是對最大值做強化，因為在做第一步的時候，對$z$取 exponential 會使大的值和小的值之間的差距被拉得更開，也就是強化大的值

我們把$z_1,z_2,z_3$丟進一個**softmax**的 function，softmax 做的事情是這樣三步：

- 取 exponential，得到$e^{z_1},e^{z_2},e^{z_3}$
- 把三個 exponential 累計求和，得到 total sum=$\sum\limits_{j=1}^3 e^{z_j}$
- 將 total sum 分別除去這三項(歸一化)，得到$y_1=\frac{e^{z_1}}{\sum\limits_{j=1}^3 e^{z_j}}$、$y_2=\frac{e^{z_2}}{\sum\limits_{j=1}^3 e^{z_j}}$、$y_3=\frac{e^{z_3}}{\sum\limits_{j=1}^3 e^{z_j}}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/multi-class.png" width="60%;" /></center>
原來的output z可以是任何值，但是做完softmax之後，你的output $y_i$的值一定是介於0~1之間，並且它們的和一定是1，$\sum\limits_i y_i=1$，以上圖為例，$y_i$表示input x屬於第i個class的概率，比如屬於C1的概率是$y_1=0.88$，屬於C2的概率是$y_2=0.12$，屬於C3的概率是$y_3=0$

而 softmax 的 output，就是拿來當 z 的 posterior probability

假設我們用的是 Gaussian distribution(共用 covariance)，經過一般推導以後可以得到 softmax 的 function，而從 information theory 也可以推導出 softmax function，[Maximum entropy](https://en.wikipedia.org/wiki/Maximum_entropy)本質內容和 Logistic Regression 是一樣的，它是從另一個觀點來切入為什麼我們的 classifier 長這樣子

##### multi-class classification 的過程：

如下圖所示，input x 經過三個式子分別生成$z_1,z_2,z_3$，經過 softmax 轉化成 output $y_1,y_2,y_3$，它們分別是這三個 class 的 posterior probability，由於 summation=1，因此做完 softmax 之後就可以把 y 的分布當做是一個 probability contribution，我們在訓練的時候還需要有一個 target，因為是三個 class，output 是三維的，對應的 target 也是三維的，為了滿足交叉熵的條件，target $\hat{y}$也必須是 probability distribution，這裡我們不能使用 1,2,3 作為 class 的區分，為了保證所有 class 之間的關係是一樣的，這裡使用類似於 one-hot 編碼的方式，即

$$
\hat{y}=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix}_{x \ ∈ \ class 1}
\hat{y}=
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}_{x \ ∈ \ class 2}
\hat{y}=
\begin{bmatrix}
0\\
0\\
1
\end{bmatrix}_{x \ ∈ \ class 3}
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/softmax.png" width="60%;" /></center>
這個時候就可以計算一下output $y$和 target $\hat{y}$之間的交叉熵，即$-\sum\limits_{i=1}^3 \hat{y}_i \ln y_i$，同二元分類一樣，多元分類問題也是通過極大似然估計法得到最終的交叉熵表達式的，這裡不再贅述

##### Limitation of Logistic Regression

Logistic Regression 其實有很強的限制，給出下圖的例子中的 Training data，想要用 Logistic Regression 對它進行分類，其實是做不到的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-limitation.png" width="60%;" /></center>
因為Logistic Regression在兩個class之間的boundary就是一條直線，但是在這個平面上無論怎麼畫直線都不可能把圖中的兩個class分隔開來

##### Feature Transformation

如果堅持要用 Logistic Regression 的話，有一招叫做**Feature Transformation**，原來的 feature 分布不好劃分，那我們可以將之轉化以後，找一個比較好的 feature space，讓 Logistic Regression 能夠處理

假設這裡定義$x_1'$是原來的點到$\begin{bmatrix}0\\0 \end{bmatrix}$之間的距離，$x_2'$是原來的點到$\begin{bmatrix}1\\ 1 \end{bmatrix}$之間的距離，重新映射之後如下圖右側(紅色兩個點重合)，此時 Logistic Regression 就可以把它們劃分開來

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/feature-transformation.png" width="60%;" /></center>
但麻煩的是，我們並不知道怎麼做feature Transformation，如果在這上面花費太多的時間就得不償失了，於是我們會希望這個Transformation是機器自己產生的，怎麼讓機器自己產生呢？==**我們可以讓很多Logistic Regression cascade(連接)起來**==

我們讓一個 input x 的兩個 feature $x_1,x_2$經過兩個 Logistic Regression 的 transform，得到新的 feature $x_1',x_2'$，在這個新的 feature space 上，class 1 和 class 2 是可以用一條直線分開的，那麼最後只要再接另外一個 Logistic Regression 的 model(對它來說，$x_1',x_2'$才是每一個樣本點的"feature"，而不是原先的$x_1,x_2$)，它根據新的 feature，就可以把 class 1 和 class 2 分開

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cascade-logistic-regression.png" width="60%;" /></center>
因此著整個流程是，先用n個Logistic Regression做feature Transformation(n為每個樣本點的feature數量)，生成n個新的feature，然後再用一個Logistic Regression作classifier

Logistic Regression 的 boundary 一定是一條直線，它可以有任何的畫法，但肯定是按照某個方向從高到低的等高線分布，具體的分布是由 Logistic Regression 的參數決定的，每一條直線都是由$z=b+\sum\limits_i^nw_ix_i$組成的(二維 feature 的直線畫在二維平面上，多維 feature 的直線則是畫在多維空間上)

下圖是二維 feature 的例子，分別表示四個點經過 transform 之後的$x_1'$和$x_2'$，在新的 feature space 中可以通過最後的 Logistic Regression 劃分開來

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/logistic-example.png" width="60%;" /></center>
注意，這裡的Logistic Regression只是一條直線，它指的是「屬於這個類」或「不屬於這個類」這兩種情況，因此最後的這個Logistic Regression是跟要檢測的目標類相關的，當只是二元分類的時候，最後只需要一個Logistic Regression即可，當面對多元分類問題，需要用到多個Logistic Regression來畫出多條直線劃分所有的類，每一個Logistic Regression對應它要檢測的那個類

##### Powerful Cascading Logistic Regression

通過上面的例子，我們發現，多個 Logistic Regression 連接起來會產生 powerful 的效果，==**我們把每一個 Logistic Regression 叫做一個 neuron(神經元)，把這些 Logistic Regression 串起來所形成的 network，就叫做 Neural Network，就是類神經網路，這個東西就是 Deep Learning！**==

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/powerful-network.png" width="60%;" /></center>
