# Unsupervised Learning: Introduction

#### Unsupervised Learning

無監督學習(Unsupervised Learning)可以分為兩種：

- 化繁為簡
  - 聚類(Clustering)
  - 降維(Dimension Reduction)
- 無中生有(Generation)

對於無監督學習(Unsupervised Learning)來說，我們通常只會擁有$(x,\hat y)$中的$x$或$\hat y$，其中：

- **化繁為簡**就是把複雜的 input 變成比較簡單的 output，比如把一大堆沒有打上 label 的樹圖片轉變為一棵抽象的樹，此時 training data 只有 input $x$，而沒有 output $\hat y$
- **無中生有**就是隨機給 function 一個數字，它就會生成不同的圖像，此時 training data 沒有 input $x$，而只有 output $\hat y$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/unsupervised.png" width="60%"/></center>

#### Clustering

##### Introduction

聚類，顧名思義，就是把相近的樣本劃分為同一類，比如對下面這些沒有標籤的 image 進行分類，手動打上 cluster 1、cluster 2、cluster 3 的標籤，這個分類過程就是化繁為簡的過程

一個很 critical 的問題：我們到底要分幾個 cluster？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/clustering.png" width="60%"/></center>

##### K-means

最常用的方法是**K-means**：

- 我們有一大堆的 unlabeled data $\{x^1,...,x^n,...,x^N\}$，我們要把它劃分為 K 個 cluster
- 對每個 cluster 都要找一個 center $c^i,i\in \{1,2,...,K\}$，initial 的時候可以從 training data 里隨機挑 K 個 object $x^n$出來作為 K 個 center $c^i$的初始值
- 遍歷所有的 object $x^n$，並判斷它屬於哪一個 cluster，如果$x^n$與第 i 個 cluster 的 center $c^i$最接近，那它就屬於該 cluster，我們用$b_i^n=1$來表示第 n 個 object 屬於第 i 個 cluster，$b_i^n=0$表示不屬於
- 更新 center：把每個 cluster 里的所有 object 取平均值作為新的 center 值，即$c^i=\sum\limits_{x^n}b_i^n x^n/\sum\limits_{x^n} b_i^n$
- 反復進行以上的操作

注：如果不是從原先的 data set 里取 center 的初始值，可能會導致部分 cluster 沒有樣本點

##### HAC

HAC，全稱 Hierarchical Agglomerative Clustering，層次聚類

假設現在我們有 5 個樣本點，想要做 clustering：

- build a tree:

  整個過程類似建立 Huffman Tree，只不過 Huffman 是依據詞頻，而 HAC 是依據相似度建樹

  - 對 5 個樣本點兩兩計算相似度，挑出最相似的一對，比如樣本點 1 和 2
  - 將樣本點 1 和 2 進行 merge (可以對兩個 vector 取平均)，生成代表這兩個樣本點的新結點
  - 此時只剩下 4 個結點，再重復上述步驟進行樣本點的合併，直到只剩下一個 root 結點

- pick a threshold：

  選取閾值，形象來說就是在構造好的 tree 上橫著切一刀，相連的葉結點屬於同一個 cluster

  下圖中，不同顏色的橫線和葉結點上不同顏色的方框對應著切法與 cluster 的分法

    <center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/HAC.png" width="60%"/></center>

HAC 和 K-means 最大的區別在於如何決定 cluster 的數量，在 K-means 里，K 的值是要你直接決定的；而在 HAC 里，你並不需要直接決定分多少 cluster，而是去決定這一刀切在樹的哪裡

#### Dimension Reduction

##### Introduction

clustering 的缺點是**以偏概全**，它強迫每個 object 都要屬於某個 cluster

但實際上某個 object 可能擁有多種屬性，或者多個 cluster 的特徵，如果把它強制歸為某個 cluster，就會失去很多信息；我們應該用一個 vector 來描述該 object，這個 vector 的每一維都代表 object 的某種屬性，這種做法就叫做 Distributed Representation，或者說，Dimension Reduction

如果原先的 object 是 high dimension 的，比如 image，那現在用它的屬性來描述自身，就可以使之從高維空間轉變為低維空間，這就是所謂的**降維(Dimension Reduction)**

下圖為動漫「全職獵人」中小傑的念能力分布，從表中可以看出我們不能僅僅把他歸為強化系

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR.png" width="60%"/></center>

##### Why Dimension Reduction Help?

接下來我們從另一個角度來看為什麼 Dimension Reduction 可能是有用的

假設 data 為下圖左側中的 3D 螺旋式分布，你會發現用 3D 的空間來描述這些 data 其實是很浪費的，因為我們完全可以把這個卷攤平，此時只需要用 2D 的空間就可以描述這個 3D 的信息

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR2.png" width="60%"/></center>

如果以 MNIST(手寫數字集)為例，每一張 image 都是 28\*28 的 dimension，但我們反過來想，大多數 28\*28 dimension 的 vector 轉成 image，看起來都不會像是一個數字，所以描述數字所需要的 dimension 可能遠比 28\*28 要來得少

舉一個極端的例子，下面這幾張表示「3」的 image，我們完全可以用中間這張 image 旋轉$\theta$角度來描述，也就是說，我們只需要用$\theta$這一個 dimension 就可以描述原先 28\*28 dimension 的圖像

你只要抓住角度的變化就可以知道 28 維空間中的變化，這裡的 28 維 pixel 就是之前提到的樊一翁的鬍子，而 1 維的角度則是他的頭，也就是去蕪存菁，化繁為簡的思想

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR3.png" width="60%"/></center>

##### How to do Dimension Reduction？

在 Dimension Reduction 里，我們要找一個 function，這個 function 的 input 是原始的 x，output 是經過降維之後的 z

最簡單的方法是**Feature Selection**，即直接從原有的 dimension 里拿掉一些直觀上就對結果沒有影響的 dimension，就做到了降維，比如下圖中從$x_1,x_2$兩個維度中直接拿掉$x_1$；但這個方法不總是有用，因為很多情況下任何一個 dimension 其實都不能被拿掉，就像下圖中的螺旋卷

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/DR4.png" width="60%"/></center>

另一個常見的方法叫做**PCA**(Principe Component Analysis)

PCA 認為降維就是一個很簡單的 linear function，它的 input x 和 output z 之間是 linear transform，即$z=Wx$，PCA 要做的，就是根據一大堆的 x**把 W 給找出來**(現在還不知道 z 長什麼樣子)

關於 PCA 算法的介紹詳見下一篇文章
