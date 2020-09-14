# Matrix Factorization

> 本文將通過一個詳細的例子分析矩陣分解思想及其在推薦系統上的應用

#### Introduction

接下來介紹**矩陣分解**的思想：有時候存在兩種 object，它們之間會受到某種共同**潛在因素**(latent factor)的操控，如果我們找出這些潛在因素，就可以對用戶的行為進行預測，這也是**推薦系統**常用的方法之一

假設我們現在去調查每個人購買的公仔數目，ABCDE 代表 5 個人，每個人或者每個公仔實際上都是有著傲嬌的屬性或天然呆的屬性

我們可以用 vector 去描述人和公仔的屬性，如果某個人的屬性和某個公仔的屬性是 match 的，即他們背後的 vector 很像(內積值很大)，這個人就會偏向於擁有更多這種類型的公仔

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf.png" width="60%"/></center>

#### matrix expression

但是，我們沒有辦法直接觀察某個人背後這些潛在的屬性，也不會有人在意一個肥宅心裡想的是什麼，我們同樣也沒有辦法直接得到動漫人物背後的屬性；我們目前有的，只是動漫人物和人之間的關係，即每個人已購買的公仔數目，我們要通過這個關係去推測出動漫人物與人背後的潛在因素(latent factor)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf2.png" width="60%"/></center>

我們可以把每個人的屬性用 vector $r^A$、$r^B$、$r^C$、$r^D$、$r^E$來表示，而動漫人物的屬性則用 vector $r^1$、$r^2$、$r^3$、$r^4$來表示，購買的公仔數目可以被看成是 matrix $X$，對$X$來說，行數為人數，列數為動漫角色的數目

做一個假設：matrix $X$里的每個 element，都是屬於人的 vector 和屬於動漫角色的 vector 的內積

比如，$r^A\cdot r^1≈5$，表示$r^A$和$r^1$的屬性比較貼近

接下來就用下圖所示的矩陣相乘的方式來表示這樣的關係，其中$K$為 latent factor 的數量，這是未知的，需要你自己去調整選擇

我們要找一組$r^A$\~$r^E$和$r^1$\~$r^4$，使得右側兩個矩陣相乘的結果與左側的 matrix $X$越接近越好，可以使用 SVD 的方法求解

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf3.png" width="60%"/></center>

#### prediction

但有時候，部分的 information 可能是會 missing 的，這時候就難以用 SVD 精確描述，但我們可以使用梯度下降的方法求解，loss function 如下：

$$
L=\sum\limits_{(i,j)}(r^i\cdot r^j-n_{ij})^2
$$

其中$r^i$值的是人背後的 latent factor，$r^j$指的是動漫角色背後的 latent factor，我們要讓這兩個 vector 的內積與實際購買該公仔的數量$n_{ij}$越接近越好，這個方法的關鍵之處在於，計算上式時，可以跳過 missing 的數據，最終通過 gradient descent 求得$r^i$和$r^j$的值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf4.png" width="60%"/></center>

假設 latent factor 的數目等於 2，則人的屬性$r^i$和動漫角色的屬性$r^j$都是 2 維的 vector，這裡實際進行計算後，把屬性中較大值標注出來，可以發現：

- 人：A、B 屬於同一組屬性，C、D、E 屬於同一組屬性
- 動漫角色：1、2 屬於同一組屬性，3、4 屬於同一組屬性

- 結合動漫角色，可以分析出動漫角色的第一個維度是天然呆屬性，第二個維度是傲嬌屬性

- 接下來就可以預測未知的值，只需要將人和動漫角色的 vector 做內積即可

這也是**推薦系統的常用方法**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf5.png" width="60%"/></center>

#### more about matrix factorization

實際上除了人和動漫角色的屬性之外，可能還存在其他因素操控購買數量這一數值，因此我們可以將式子更精確地改寫為：

$$
r^A\cdot r^1+b_A+b_1≈5
$$

其中$b_A$表示 A 這個人本身有多喜歡買公仔，$b_1$則表示這個動漫角色本身有多讓人想要購買，這些內容是跟屬性 vector 無關的，此時 loss function 被改寫為：

$$
L=\sum\limits_{(i,j)}(r^i\cdot r^j+b_i+b_j-n_{ij})^2
$$

當然你也可以加上一些 regularization 去對結果做約束

有關 Matrix Factorization 和推薦系統更多內容的介紹，可以參考 paper(公眾號回復「推薦系統」獲取 pdf )：_Matrix Factorization Techniques For Recommender Systems_

#### for Topic Analysis

如果把 matrix factorization 的方法用在 topic analysis 上，就叫做 LSA(Latent semantic analysis)，潛在語義分析

我們只需要把動漫人物換成文章，人換成詞彙，表中的值從購買數量換成詞頻即可，我們可以用詞彙的重要性給詞頻加權，在各種文章中出現次數越多的詞彙越不重要，出現次數越少則越重要

這個場景下找出的 latent factor 可能會是主題(topic)，比如某個詞彙或某個文檔有多少比例是偏向於財經主題、政治主題...

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/mf6.png" width="60%"/></center>
