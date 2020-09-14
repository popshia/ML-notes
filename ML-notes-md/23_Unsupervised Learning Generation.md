# Unsupervised Learning: Generation

> 本文將簡單介紹無監督學習中的生成模型，包括 PixelRNN、VAE 和 GAN，以後將會有一個專門的系列介紹對抗生成網絡 GAN

#### Introduction

正如*Richard Feynman*所說，_「What I cannot create, I do not understand」_，我無法創造的東西，我也無法真正理解，機器可以做貓狗分類，但卻不一定知道「貓」和「狗」的概念，但如果機器能自己畫出「貓」來，它或許才真正理解了「貓」這個概念

這裡將簡要介紹：PixelRNN、VAE 和 GAN 這三種方法

#### PixelRNN

##### Introduction

RNN 可以處理長度可變的 input，它的基本思想是根據過去發生的所有狀態去推測下一個狀態

PixelRNN 的基本思想是每次只畫一個 pixel，這個 pixel 是由過去所有已產生的 pixel 共同決定的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pixel-rnn.png" width="60%"/></center>

這個方法也適用於語音生成，可以用前面一段的語音去預測接下來生成的語音信號

總之，這種方法的精髓在於根據過去預測未來，畫出來的圖一般都是比較清晰的

##### pokemon creation

用這個方法去生成寶可夢，有幾個 tips：

- 為了減少運算量，將 40×40 的圖像截取成 20×20

- 如果將每個 pixel 都以[R, G, B]的 vector 表示的話，生成的圖像都是灰蒙蒙的，原因如下：

  - 亮度比較高的圖像，一般都是 RGB 值差距特別大而形成的，如果各個維度的值大小比較接近，則生成的圖像偏向於灰色

  - 如果用 sigmoid function，最終生成的 RGB 往往都是在 0.5 左右，導致色彩度不鮮艷

  - 解決方案：將所有色彩集合成一個 1-of-N 編碼，由於色彩種類比較多，因此這裡先對類似的顏色做 clustering 聚類，最終獲得了 167 種色彩組成的向量

    我們用這樣的向量去表示每個 pixel，可以讓生成的色彩比較鮮艷

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pixel-rnn-pokemon.png" width="60%"/></center>

相關數據連接如下：

- 原始圖像(40\*40)數據的[鏈接](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Pokemon_creation/image.rar)
- 裁剪後的圖像(20\*20)數據[鏈接](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Pokemon_creation/pixel_color.txt)
- 數值與色彩(RGB)的映射關係[鏈接](http://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Pokemon_creation/colormap.txt)

使用 PixelRNN 訓練好模型之後，給它看沒有被放在訓練集中的 3 張圖像的一部分，分別遮住原圖的 50%和 75%，得到的原圖和預測結果的對比如下：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pixel-rnn-pokemon2.png" width="60%"/></center>

#### VAE

VAE 全稱 Variational Autoencoder，可變自動編碼器

##### Introduction

前面的文章中已經介紹過 Autoencoder 的基本思想，我們拿出其中的 Decoder，給它隨機的輸入數據，就可以生成對應的圖像

但普通的 Decoder 生成效果並不好，VAE 可以得到更好的效果

在 VAE 中，code 不再直接等於 Encoder 的輸出，這裡假設目標降維空間為 3 維，那我們使 Encoder 分別輸出$m_1,m_2,m_3$和$\sigma_1,\sigma_2,\sigma_3$，此外我們從正態分布中隨機取出三個點$e_1,e_2,e_3$，將下式作為最終的編碼結果：

$$
c_i = e^{\sigma_i}\cdot e_i+m_i
$$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vae.png" width="60%"/></center>

此時，我們的訓練目標不僅要最小化 input 和 output 之間的差距，還要同時最小化下式：

$$
\sum\limits_{i=1}^3 (1+\sigma_i-(m_i)^2-e^{\sigma_i})
$$

與 PixelRNN 不同的是，VAE 畫出的圖一般都是不太清晰的，但在某種程度上我們可以控制生成的圖像

##### write poetry

VAE 還可以用來寫詩，我們只需要得到某兩句話對應的 code，然後在降維後的空間中得到這兩個 code 所在點的連線，從中取樣，並輸入給 Decoder，就可以得到類似下圖中的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vae-poetry.png" width="60%"/></center>

##### Why VAE?

VAE 和傳統的 Autoencoder 相比，有什麼優勢呢？

事實上，VAE 就是加了噪聲 noise 的 Autoencoder，它的抗干擾能力更強，過渡生成能力也更強

對原先的 Autoencoder 來說，假設我們得到了滿月和弦月的 code，從兩者連線中隨機獲取一個點並映射回原來的空間，得到的圖像很可能是完全不一樣的東西

而對 VAE 來說，它要保證在降維後的空間中，加了 noise 的一段範圍內的所有點都能夠映射到目標圖像，如下圖所示，當某個點既被要求映射到滿月、又被要求映射到弦月，則它最終映射出來的結果就很有可能是兩者之間的過渡圖像

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vae-why.png" width="60%"/></center>

再回過來頭看 VAE 的結構，其中：

- $m_i$其實就代表原來的 code

- $c_i$則代表加了 noise 以後的 code

- $\sigma_i$代表了 noise 的 variance，描述了 noise 的大小，這是由 NN 學習到的參數

  注：使用$e^{\sigma_i}$的目的是保證 variance 是正的

- $e_i$是正態分布中隨機採樣的點

注意到，損失函數僅僅讓 input 和 output 差距最小是不夠的，因為 variance 是由機器自己決定的，如果不加以約束，它自然會去讓 variance=0，這就跟普通的 Autoencoder 沒有區別了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vae-why2.png" width="60%"/></center>

額外加的限制函數解釋如下：

下圖中，藍線表示$e^{\sigma_i}$，紅線表示$1+\sigma_i$，兩者相減得到綠線

綠線的最低點$\sigma_i=0$，則 variance $e^{\sigma_i}=1$，此時 loss 最低

而$(m_i)^2$項則是對 code 的 L2 regularization，讓它比較 sparse，不容易過擬合

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/vae-why3.png" width="60%"/></center>

關於 VAE 原理的具體推導比較複雜，這裡不再列出

##### problems of VAE

VAE 有一個缺點，它只是在努力做到讓生成的圖像與數據集里的圖像盡可能相似，卻從來沒有想過怎麼樣真的產生一張新的圖像，因此由 VAE 生成的圖像大多是數據集中圖像的線性變化，而很難自主生成全新的圖像

VAE 做到的只是模仿，而不是創造，GAN 的誕生，就是為了創造

#### GAN

GAN，對抗生成網絡，是近兩年非常流行的神經網絡，基本思想就像是天敵之間相互競爭，相互進步

GAN 由生成器(Generator)和判別器(Discriminator)組成：

- 對判別器的訓練：把生成器產生的圖像標記為 0，真實圖像標記為 1，丟給判別器訓練分類
- 對生成器的訓練：調整生成器的參數，使產生的圖像能夠騙過判別器
- 每次訓練調整判別器或生成器參數的時候，都要固定住另一個的參數

GAN 的問題：沒有明確的訓練目標，很難調整生成器和判別器的參數使之始終處於勢均力敵的狀態，當兩者之間的 loss 很小的時候，並不意味著訓練結果是好的，有可能它們兩個一起走向了一個壞的極端，所以在訓練的同時還必須要有人在旁邊關注著訓練的情況

以後將會有 GAN 系列的文章介紹，本文不再做詳細說明
