# Unsupervised Learning: Deep Auto-encoder

> 文本介紹了自編碼器的基本思想，與 PCA 的聯繫，從單層編碼到多層的變化，在文字搜索和圖像搜索上的應用，預訓練 DNN 的基本過程，利用 CNN 實現自編碼器的過程，加噪聲的自編碼器，利用解碼器生成圖像等內容

#### Introduction

**Auto-encoder 本質上就是一個自我壓縮和解壓的過程**，我們想要獲取壓縮後的 code，它代表了對原始數據的某種緊湊精簡的有效表達，即降維結果，這個過程中我們需要：

- Encoder(編碼器)，它可以把原先的圖像壓縮成更低維度的向量
- Decoder(解碼器)，它可以把壓縮後的向量還原成圖像

注意到，Encoder 和 Decoder 都是 Unsupervised Learning，由於 code 是未知的，對 Encoder 來說，我們手中的數據只能提供圖像作為 NN 的 input，卻不能提供 code 作為 output；對 Decoder 來說，我們只能提供圖像作為 NN 的 output，卻不能提供 code 作為 input

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto.png" width="60%"/></center>

因此 Encoder 和 Decoder 單獨拿出一個都無法進行訓練，我們需要把它們連接起來，這樣整個神經網絡的輸入和輸出都是我們已有的圖像數據，就可以同時對 Encoder 和 Decoder 進行訓練，而降維後的編碼結果就可以從最中間的那層 hidden layer 中獲取

#### Compare with PCA

實際上 PCA 用到的思想與之非常類似，**PCA 的過程本質上就是按組件拆分，再按組件重構的過程**

在 PCA 中，我們先把均一化後的$x$根據組件$W$分解到更低維度的$c$，然後再將組件權重$c$乘上組件的反置$W^T$得到重組後的$\hat x$，同樣我們期望重構後的$\hat x$與原始的$x$越接近越好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-pca.png" width="60%"/></center>

如果把這個過程看作是神經網絡，那麼原始的$x$就是 input layer，重構$\hat x$就是 output layer，中間組件分解權重$c$就是 hidden layer，在 PCA 中它是 linear 的，我們通常又叫它瓶頸層(Bottleneck layer)

由於經過組件分解降維後的$c$，維數要遠比輸入輸出層來得低，因此 hidden layer 實際上非常窄，因而有瓶頸層的稱呼

對比於 Auto-encoder，從 input layer 到 hidden layer 的按組件分解實際上就是編碼(encode)過程，從 hidden layer 到 output layer 按組件重構實際上就是解碼(decode)的過程

這時候你可能會想，可不可以用更多層 hidden layer 呢？答案是肯定的

#### Deep Auto-encoder

##### Multi Layer

對 deep 的自編碼器來說，實際上就是通過多級編碼降維，再經過多級解碼還原的過程

此時：

- 從 input layer 到 bottleneck layer 的部分都屬於$Encoder$
- 從 bottleneck layer 到 output layer 的部分都屬於$Decoder$
- bottleneck layer 的 output 就是自編碼結果$code$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-deep.png" width="60%"/></center>

注意到，如果按照 PCA 的思路，則 Encoder 的參數$W_i$需要和 Decoder 的參數$W_i^T$保持一致的對應關係，這可以通過給兩者相同的初始值並設置同樣的更新過程得到，這樣做的好處是，可以節省一半的參數，降低 overfitting 的概率

但這件事情並不是必要的，實際操作的時候，你完全可以對神經網絡進行直接訓練而不用保持編碼器和解碼器的參數一致

##### Visualize

下圖給出了 Hinton 分別採用 PCA 和 Deep Auto-encoder 對手寫數字進行編碼解碼後的結果，從 784 維降到 30 維，可以看出，Deep 的自編碼器還原效果比 PCA 要更好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-deep2.png" width="60%"/></center>

如果將其降到二維平面做可視化，不同顏色代表不同的數字，可以看到

- 通過 PCA 降維得到的編碼結果中，不同顏色代表的數字被混雜在一起
- 通過 Deep Auto-encoder 降維得到的編碼結果中，不同顏色代表的數字被分散成一群一群的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-visual.png" width="60%"/></center>

#### Text Retrieval

Auto-encoder 也可以被用在文字處理上

比如我們要做文字檢索，很簡單的一個做法是 Vector Space Model，把每一篇文章都表示成空間中的一個 vector

假設查詢者輸入了某個詞彙，那我們就把該查詢詞彙也變成空間中的一個點，並計算 query 和每一篇 document 之間的內積(inner product)或余弦相似度(cos-similarity)

注：余弦相似度有均一化的效果，可能會得到更好的結果

下圖中跟 query 向量最接近的幾個向量的 cosine-similarity 是最大的，於是可以從這幾篇文章中去檢索

實際上這個模型的好壞，就取決於從 document 轉化而來的 vector 的好壞，它是否能夠充分表達文章信息

##### Bag-of-word

最簡單的 vector 表示方法是 Bag-of-word，維數等於所有詞彙的總數，某一維等於 1 則表示該詞彙在這篇文章中出現，此外還可以根據詞彙的重要性將其加權；但這個模型是非常脆弱的，對它來說每個詞彙都是相互獨立的，無法體現出詞彙之間的語義(semantic)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-text.png" width="60%"/></center>

##### Auto-encoder

雖然 Bag-of-word 不能直接用於表示文章，但我們可以把它作為 Auto-encoder 的 input，通過降維來抽取有效信息，以獲取所需的 vector

同樣為了可視化，這裡將 Bag-of-word 降維到二維平面上，下圖中每個點都代表一篇文章，不同顏色則代表不同的文章類型

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-visual2.png" width="60%"/></center>

如果用戶做查詢，就把查詢的語句用相同的方式映射到該二維平面上，並找出屬於同一類別的所有文章即可

在矩陣分解(Matrix Factorization)中，我們介紹了 LSA 算法，它可以用來尋找每個詞彙和每篇文章背後的隱藏關係(vector)，如果在這裡我們採用 LSA，並使用二維 latent vector 來表示每篇文章，得到的可視化結果如上圖右下角所示，可見效果並沒有 Auto-encoder 好

#### Similar Image Search

Auto-encoder 同樣可以被用在圖像檢索上

以圖找圖最簡單的做法就是直接對輸入的圖片與數據庫中的圖片計算 pixel 的相似度，並挑出最像的圖片，但這種方法的效果是不好的，因為單純的 pixel 所能夠表達的信息太少了

我們需要使用 Auto-encoder 對圖像進行降維和特徵提取，並在編碼得到的 code 所在空間做檢索

下圖展示了 Encoder 的過程，並給出了原圖與 Decoder 後的圖像對比

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-img.png" width="60%"/></center>

這麼做的好處如下：

- Auto-encoder 可以通過降維提取出一張圖像中最有用的特徵信息，包括 pixel 與 pixel 之間的關係
- 降維之後數據的 size 變小了，這意味著模型所需的參數也變少了，同樣的數據量對參數更少的模型來說，可以訓練出更精確的結果，一定程度上避免了過擬合的發生
- Auto-encoder 是一個無監督學習的方法，數據不需要人工打上標籤，這意味著我們只需簡單處理就可以獲得大量的可用數據

下圖給出了分別以原圖的 pixel 計算相似度和以 auto-encoder 後的 code 計算相似度的兩種方法在圖像檢索上的結果，可以看到，通過 pixel 檢索到的圖像會出現很多奇怪的物品，而通過 code 檢索到的圖像則都是人臉

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-img2.png" width="60%"/></center>

可能有些人臉在原圖的 pixel 上看起來並不像，但把它們投影到 256 維的空間中卻是相像的，可能在投影空間中某一維就代表了人臉的特徵，因此能夠被檢索出來

#### Pre-training DNN

在訓練神經網絡的時候，我們一般都會對如何初始化參數比較困擾，預訓練(pre-training)是一種尋找比較好的參數初始化值的方法，而我們可以用 Auto-encoder 來做 pre-training

以 MNIST 數據集為例，我們對每層 hidden layer 都做一次 auto-encoder，**使每一層都能夠提取到上一層最佳的特徵向量**

為了方便表述，這裡用$x-z-x$來表示一個自編碼器，其中$x$表述輸入輸出層的維數，$z$表示隱藏層的維數

- 首先使 input 通過一個$784-1000-784$的自編碼器，當該自編碼器訓練穩定後，就把參數$W^1$固定住，然後將數據集中所有 784 維的圖像都轉化為 1000 維的 vector

  注意：這裡做的不是降維而是升維，當編碼後的維數比輸入維數要高時，需要注意可能會出現編碼前後原封不動的情況，為此需要額外加一個正則項，比如 L1 regularization，強迫使 code 的分布是分散的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-pre.png" width="60%"/></center>

- 接下來再讓這些 1000 維的 vector 通過一個$1000-1000-1000$的編碼器，當其訓練穩定後，再把參數$W^2$固定住，對數據集再做一次轉換

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-pre2.png" width="60%"/></center>

- 接下來再用轉換後的數據集去訓練第三個$1000-500-1000$的自編碼器，訓練穩定後固定$W^3$，數據集再次更新轉化為 500 維

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-pre3.png" width="60%"/></center>

- 此時三個隱藏層的參數$W^1$、$W^2$、$W^3$就是訓練整個神經網絡時的參數初始值

- 然後隨機初始化最後一個隱藏層到輸出層之間的參數$W^4$

- 再用反向傳播去調整一遍參數，因為$W^1$、$W^2$、$W^3$都已經是很好的參數值了，這裡只是做微調，這個步驟也因此得名為**Find-tune**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-pre4.png" width="60%"/></center>

由於現在訓練機器的條件比以往更好，因此 pre-training 並不是必要的，但它也有自己的優勢

如果你只有大量的 unlabeled data 和少量的 labeled data，那你可以先用這些 unlabeled data 把$W^1$、$W^2$、$W^3$先初始化好，最後再用 labeled data 去微調$W^1$~$W^4$即可

因此 pre-training 在有大量 unlabeled data 的場景下(如半監督學習)是比較有用的

#### CNN

##### CNN as Encoder

處理圖像通常都會用卷積神經網絡 CNN，它的基本思想是交替使用卷積層和池化層，讓圖像越來越小，最終展平，這個過程跟 Encoder 編碼的過程其實是類似的

理論上要實現自編碼器，Decoder 只需要做跟 Encoder 相反的事即可，那對 CNN 來說，解碼的過程也就變成了交替使用去卷積層和去池化層即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-cnn.png" width="60%"/></center>

那什麼是去卷積層(Deconvolution)和去池化層(Unpooling)呢？

##### Unpooling

做 pooling 的時候，假如得到一個 4×4 的 matrix，就把每 4 個 pixel 分為一組，從每組中挑一個最大的留下，此時圖像就變成了原來的四分之一大小

如果還要做 Unpooling，就需要提前記錄 pooling 所挑選的 pixel 在原圖中的位置，下圖中用灰色方框標注

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-unpooling.png" width="60%"/></center>

然後做 Unpooling，就要把當前的 matrix 放大到原來的四倍，也就是把 2×2 matrix 里的 pixel 按照原先記錄的位置插入放大後的 4×4 matrix 中，其餘項補 0 即可

當然這不是唯一的做法，在 Keras 中，pooling 並沒有記錄原先的位置，做 Unpooling 的時候就是直接把 pixel 的值複製四份填充到擴大後的 matrix 里即可

##### Deconvolution

實際上，Deconvolution 就是 convolution

這裡以一維的卷積為例，假設輸入是 5 維，過濾器(filter)的大小是 3

卷積的過程就是每三個相鄰的點通過過濾器生成一個新的點，如下圖左側所示

在你的想象中，去卷積的過程應該是每個點都生成三個點，不同的點對生成同一個點的貢獻值相加；但實際上，這個過程就相當於在周圍補 0 之後再次做卷積，如下圖右側所示，兩個過程是等價的

卷積和去卷積的過程中，不同點在於，去卷積需要補零且過濾器的 weight 與卷積是相反的：

- 在卷積過程中，依次是橙線、藍線、綠線
- 在去卷積過程中，依次是綠線、藍線、橙線

因此在實踐中，做去卷積的時候直接對模型加卷積層即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-deconvolution.png" width="60%"/></center>

#### Other Auto-encoder

##### De-noising Auto-encoder

去噪自編碼器的基本思想是，把輸入的$x$加上一些噪聲(noise)變成$x'$，再對$x'$依次做編碼(encode)和解碼(decode)，得到還原後的$y$

值得注意的是，一般的自編碼器都是讓輸入輸出盡可能接近，但在去噪自編碼器中，我們的目標是讓解碼後的$y$與加噪聲之前的$x$越接近越好

這種方法可以增加系統的魯棒性，因為此時的編碼器 Encoder 不僅僅是在學習如何做編碼，它還學習到了如何過濾掉噪聲這件事情

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-noise.png" width="60%"/></center>

參考文獻：_Vincent, Pascal, et al. "Extracting and composing robust features with denoising autoencoders." ICML, 2008._

##### Contractive Auto-encoder

收縮自動編碼器的基本思想是，在做 encode 編碼的時候，要加上一個約束，它可以使得：input 的變化對編碼後得到的 code 的影響最小化

這個描述跟去噪自編碼器很像，只不過去噪自編碼器的重點在於加了噪聲之後依舊可以還原回原先的輸入，而收縮自動編碼器的重點在於加了噪聲之後能夠保持編碼結果不變

參考文獻：_Rifai, Salah, et al. "Contractive auto-encoders: Explicit invariance during feature extraction.「 Proceedings of the 28th International Conference on Machine Learning (ICML-11). 2011._

##### Seq2Seq Auto-encoder

在之前介紹的自編碼器中，輸入都是一個固定長度的 vector，但類似文章、語音等信息實際上不應該單純被表示為 vector，那會丟失很多前後聯繫的信息

Seq2Seq 就是為瞭解決這個問題提出的，具體內容將在 RNN 部分介紹

#### Generate

在用自編碼器的時候，通常是獲取 Encoder 之後的 code 作為降維結果，但實際上 Decoder 也是有作用的，我們可以拿它來生成新的東西

以 MNIST 為例，訓練好編碼器之後，取出其中的 Decoder，輸入一個隨機的 code，就可以生成一張圖像

假設將 28×28 維的圖像通過一層 2 維的 hidden layer 投影到二維平面上，得到的結果如下所示，不同顏色的點代表不同的數字，然後在紅色方框中，等間隔的挑選二維向量丟進 Decoder 中，就會生成許多數字的圖像

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-gene.png" width="60%"/></center>

此外，我們還可以對 code 加 L2 regularization，以限制 code 分布的範圍集中在 0 附近，此時就可以直接以 0 為中心去隨機採取樣本點，再通過 Decoder 生成圖像

觀察生成的數字圖像，可以發現橫軸的維度表示是否含有圓圈，縱軸的維度表示是否傾斜

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/auto-gene2.png" width="60%"/></center>
