### Unsupervised Learning: Word Embedding

> 本文介紹 NLP 中詞嵌入(Word Embedding)相關的基本知識，基於降維思想提供了 count-based 和 prediction-based 兩種方法，並介紹了該思想在機器問答、機器翻譯、圖像分類、文檔嵌入等方面的應用

#### Introduction

詞嵌入(word embedding)是降維算法(Dimension Reduction)的典型應用

那如何用 vector 來表示一個 word 呢？

##### 1-of-N Encoding

最傳統的做法是 1-of-N Encoding，假設這個 vector 的維數就等於世界上所有單詞的數目，那麼對每一個單詞來說，只需要某一維為 1，其餘都是 0 即可；但這會導致任意兩個 vector 都是不一樣的，你無法建立起同類 word 之間的聯繫

##### Word Class

還可以把有同樣性質的 word 進行聚類(clustering)，劃分成多個 class，然後用 word 所屬的 class 來表示這個 word，但光做 clustering 是不夠的，不同 class 之間關聯依舊無法被有效地表達出來

##### Word Embedding

詞嵌入(Word Embedding)把每一個 word 都投影到高維空間上，當然這個空間的維度要遠比 1-of-N Encoding 的維度低，假如後者有 10w 維，那前者只需要 50\~100 維就夠了，這實際上也是 Dimension Reduction 的過程

類似語義(semantic)的詞彙，在這個 word embedding 的投影空間上是比較接近的，而且該空間里的每一維都可能有特殊的含義

假設詞嵌入的投影空間如下圖所示，則橫軸代表了生物與其它東西之間的區別，而縱軸則代表了會動的東西與靜止的東西之間的差別

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we.png" width="60%"/></center>

word embedding 是一個無監督的方法(unsupervised approach)，只要讓機器閱讀大量的文章，它就可以知道每一個詞彙 embedding 之後的特徵向量應該長什麼樣子

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we2.png" width="60%"/></center>

我們的任務就是訓練一個 neural network，input 是詞彙，output 則是它所對應的 word embedding vector，實際訓練的時候我們只有 data 的 input，該如何解這類問題呢？

之前提到過一種基於神經網絡的降維方法，Auto-encoder，就是訓練一個 model，讓它的輸入等於輸出，取出中間的某個隱藏層就是降維的結果，自編碼的本質就是通過自我壓縮和解壓的過程來尋找各個維度之間的相關信息；但 word embedding 這個問題是不能用 Auto-encoder 來解的，因為輸入的向量通常是 1-of-N 編碼，各維無關，很難通過自編碼的過程提取出什麼有用信息

#### Word Embedding

##### basic idea

基本精神就是，每一個詞彙的含義都可以根據它的上下文來得到

比如機器在兩個不同的地方閱讀到了「馬英九 520 宣誓就職」、「蔡英文 520 宣誓就職」，它就會發現「馬英九」和「蔡英文」前後都有類似的文字內容，於是機器就可以推測「馬英九」和「蔡英文」這兩個詞彙代表了可能有同樣地位的東西，即使它並不知道這兩個詞彙是人名

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we3.png" width="60%"/></center>

怎麼用這個思想來找出 word embedding 的 vector 呢？有兩種做法：

- Count based
- Prediction based

#### Count based

假如$w_i$和$w_j$這兩個詞彙常常在同一篇文章中出現(co-occur)，它們的 word vector 分別用$V(w_i)$和$V(w_j)$來表示，則$V(w_i)$和$V(w_j)$會比較接近

假設$N_{i,j}$是$w_i$和$w_j$這兩個詞彙在相同文章里同時出現的次數，我們希望它與$V(w_i)\cdot V(w_j)$的內積越接近越好，這個思想和之前的文章中提到的矩陣分解(matrix factorization)的思想其實是一樣的

這種方法有一個很代表性的例子是[Glove Vector](http://nlp.stanford.edu/projects/glove/)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/count-based.png" width="60%"/></center>

#### Prediction based

##### how to do perdition

給定一個 sentence，我們要訓練一個神經網絡，它要做的就是根據當前的 word $w_{i-1}$，來預測下一個可能出現的 word $w_i$是什麼

假設我們使用 1-of-N encoding 把$w_{i-1}$表示成 feature vector，它作為 neural network 的 input，output 的維數和 input 相等，只不過每一維都是小數，代表在 1-of-N 編碼中該維為 1 其餘維為 0 所對應的 word 會是下一個 word $w_i$的概率

把第一個 hidden layer 的 input $z_1,z_2,...$拿出來，它們所組成的$Z$就是 word 的另一種表示方式，當我們 input 不同的詞彙，向量$Z$就會發生變化

也就是說，第一層 hidden layer 的維數可以由我們決定，而它的 input 又唯一確定了一個 word，因此提取出第一層 hidden layer 的 input，實際上就得到了一組可以自定義維數的 Word Embedding 的向量

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb.png" width="60%"/></center>

##### Why prediction works

prediction-based 方法是如何體現根據詞彙的上下文來瞭解該詞彙的含義這件事呢？

假設在兩篇文章中，「蔡英文」和「馬英九」代表$w_{i-1}$，「宣誓就職」代表$w_i$，我們希望對神經網絡輸入「蔡英文」或「馬英九」這兩個詞彙，輸出的 vector 中對應「宣誓就職」詞彙的那個維度的概率值是高的

為了使這兩個不同的 input 通過 NN 能得到相同的 output，就必須在進入 hidden layer 之前，就通過 weight 的轉換將這兩個 input vector 投影到位置相近的低維空間上

也就是說，儘管兩個 input vector 作為 1-of-N 編碼看起來完全不同，但經過參數的轉換，將兩者都降維到某一個空間中，在這個空間里，經過轉換後的 new vector 1 和 vector 2 是非常接近的，因此它們同時進入一系列的 hidden layer，最終輸出時得到的 output 是相同的

因此，詞彙上下文的聯繫就自動被考慮在這個 prediction model 裡面

總結一下，對 1-of-N 編碼進行 Word Embedding 降維的結果就是神經網絡模型第一層 hidden layer 的輸入向量$\left [ \begin{matrix} z_1\ z_2\ ... \end{matrix} \right ]^T$，該向量同時也考慮了上下文詞彙的關聯，我們可以通過控制第一層 hidden layer 的大小從而控制目標降維空間的維數

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb2.png" width="60%"/></center>

##### Sharing Parameters

你可能會覺得通過當前詞彙預測下一個詞彙這個約束太弱了，由於不同詞彙的搭配千千萬萬，即便是人也無法準確地給出下一個詞彙具體是什麼

你可以擴展這個問題，使用 10 個及以上的詞彙去預測下一個詞彙，可以幫助得到較好的結果

這裡用 2 個詞彙舉例，如果是一般是神經網絡，我們直接把$w_{i-2}$和$w_{i-1}$這兩個 vector 拼接成一個更長的 vector 作為 input 即可

但實際上，我們希望和$w_{i-2}$相連的 weight 與和$w_{i-1}$相連的 weight 是 tight 在一起的，簡單來說就是$w_{i-2}$與$w_{i-1}$的相同 dimension 對應到第一層 hidden layer 相同 neuron 之間的連線擁有相同的 weight，在下圖中，用同樣的顏色標注相同的 weight：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb3.png" width="60%"/></center>

如果我們不這麼做，那把同一個 word 放在$w_{i-2}$的位置和放在$w_{i-1}$的位置，得到的 Embedding 結果是會不一樣的，把兩組 weight 設置成相同，可以使$w_{i-2}$與$w_{i-1}$的相對位置不會對結果產生影響

除此之外，這麼做還可以通過共享參數的方式有效地減少參數量，不會由於 input 的 word 數量增加而導致參數量劇增

##### Formulation

假設$w_{i-2}$的 1-of-N 編碼為$x_{i-2}$，$w_{i-1}$的 1-of-N 編碼為$x_{i-1}$，維數均為$|V|$，表示數據中的 words 總數

hidden layer 的 input 為向量$z$，長度為$|Z|$，表示降維後的維數

$$
z=W_1 x_{i-2}+W_2 x_{i-1}
$$

其中$W_1$和$W_2$都是$|Z|×|V|$維的 weight matrix，它由$|Z|$組$|V|$維的向量構成，第一組$|V|$維向量與$|V|$維的$x_{i-2}$相乘得到$z_1$，第二組$|V|$維向量與$|V|$維的$x_{i-2}$相乘得到$z_2$，...，依次類推

我們強迫讓$W_1=W_2=W$，此時$z=W(x_{i-2}+x_{i-1})$

因此，只要我們得到了這組參數$W$，就可以與 1-of-N 編碼$x$相乘得到 word embedding 的結果$z$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb4.png" width="60%"/></center>

##### In Practice

那在實際操作上，我們如何保證$W_1$和$W_2$一樣呢？

以下圖中的$w_i$和$w_j$為例，我們希望它們的 weight 是一樣的：

- 首先在訓練的時候就要給它們一樣的初始值

- 然後分別計算 loss function $C$對$w_i$和$w_j$的偏微分，並對其進行更新

  $$
  w_i=w_i-\eta \frac{\partial C}{\partial w_i}\\
  w_j=w_j-\eta \frac{\partial C}{\partial w_j}
  $$

  這個時候你就會發現，$C$對$w_i$和$w_j$的偏微分是不一樣的，這意味著即使給了$w_i$和$w_j$相同的初始值，更新過一次之後它們的值也會變得不一樣，因此我們必須保證兩者的更新過程是一致的，即：

  $$
  w_i=w_i-\eta \frac{\partial C}{\partial w_i}-\eta \frac{\partial C}{\partial w_j}\\
  w_j=w_j-\eta \frac{\partial C}{\partial w_j}-\eta \frac{\partial C}{\partial w_i}
  $$

- 這個時候，我們就保證了$w_i$和$w_j$始終相等：
  - $w_i$和$w_j$的初始值相同
  - $w_i$和$w_j$的更新過程相同

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb5.png" width="60%"/></center>

如何去訓練這個神經網絡呢？注意到這個 NN 完全是 unsupervised，你只需要上網爬一下文章數據直接餵給它即可

比如餵給 NN 的 input 是「潮水」和「退了」，希望它的 output 是「就」，之前提到這個 NN 的輸出是一個由概率組成的 vector，而目標「就」是只有某一維為 1 的 1-of-N 編碼，我們希望 minimize 它們之間的 cross entropy，也就是使得輸出的那個 vector 在「就」所對應的那一維上概率最高

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb6.png" width="60%"/></center>

##### Various Architectures

除了上面的基本形態，Prediction-based 方法還可以有多種變形

- CBOW(Continuous bag of word model)

  拿前後的詞彙去預測中間的詞彙

- Skip-gram

  拿中間的詞彙去預測前後的詞彙

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pb7.png" width="60%"/></center>

##### others

儘管 word vector 是 deep learning 的一個應用，但這個 neural network 其實並不是 deep 的，它就只有一個 linear 的 hidden layer

我們把 1-of-N 編碼輸入給神經網絡，經過 weight 的轉換得到 Word Embedding，再通過第一層 hidden layer 就可以直接得到輸出

其實過去有很多人使用過 deep model，但這個 task 不用 deep 就可以實現，這樣做既可以減少運算量，跑大量的 data，又可以節省下訓練的時間(deep model 很可能需要長達好幾天的訓練時間)

#### Application

##### Subtraction

_機器問答_

從得到的 word vector 里，我們可以發現一些原本並不知道的 word 與 word 之間的關係

把 word vector 兩兩相減，再投影到下圖中的二維平面上，如果某兩個 word 之間有類似包含於的相同關係，它們就會被投影到同一塊區域

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we4.png" width="60%"/></center>

利用這個概念，我們可以做一些簡單的推論：

- 在 word vector 的特徵上，$V(Rome)-V(Italy)≈V(Berlin)-V(Germany)$

- 此時如果有人問「羅馬之於意大利等於柏林之於？」，那機器就可以回答這個問題

  因為德國的 vector 會很接近於「柏林的 vector-羅馬的 vector+意大利的 vector」，因此機器只需要計算$V(Berlin)-V(Rome)+V(Italy)$，然後選取與這個結果最接近的 vector 即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we5.png" width="60%"/></center>

##### Multi-lingual Embedding

_機器翻譯_

此外，word vector 還可以建立起不同語言之間的聯繫

如果你要用上述方法分別訓練一個英文的語料庫(corpus)和中文的語料庫，你會發現兩者的 word vector 之間是沒有任何關係的，因為 Word Embedding 只體現了上下文的關係，如果你的文章沒有把中英文混合在一起使用，機器就沒有辦法判斷中英文詞彙之間的關係

但是，如果你知道某些中文詞彙和英文詞彙的對應關係，你可以先分別獲取它們的 word vector，然後再去訓練一個模型，把具有相同含義的中英文詞彙投影到新空間上的同一個點

接下來遇到未知的新詞彙，無論是中文還是英文，你都可以採用同樣的方式將其投影到新空間，就可以自動做到類似翻譯的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we6.png" width="60%"/></center>

參考文獻：_Bilingual Word Embeddings for Phrase-Based Machine Translation, Will Zou, Richard Socher, Daniel Cer and Christopher Manning, EMNLP, 2013_

##### Multi-domain Embedding

_圖像分類_

這個做法不只局限於文字的應用，你也可以對文字+圖像做 Embedding

假設你已經得到 horse、cat 和 dog 這些**詞彙**的 vector 在空間上的分布情況，你就可以去訓練一個模型，把一些已知的 horse、cat 和 dog**圖片**去投影到和對應詞彙相同的空間區域上

比如對模型輸入一張圖像，使之輸出一個跟 word vector 具有相同維數的 vector，使 dog 圖像的映射向量就散布在 dog 詞彙向量的周圍，horse 圖像的映射向量就散布在 horse 詞彙向量的周圍...

訓練好這個模型之後，輸入新的未知圖像，根據投影之後的位置所對應的 word vector，就可以判斷它所屬的類別

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/we7.png" width="60%"/></center>

我們知道在做圖像分類的時候，很多情況下都是事先定好要分為哪幾個具體的類別，再用這幾個類別的圖像去訓練模型，由於我們無法在訓練的時候窮盡所有類別的圖像，因此在實際應用的時候一旦遇到屬於未知類別的圖像，這個模型就無能為力了

而使用 image+word Embedding 的方法，就算輸入的圖像類別在訓練時沒有被遇到過，比如上圖中的 cat，但如果這張圖像能夠投影到 cat 的 word vector 的附近，根據詞彙向量與圖像向量的對應關係，你自然就可以知道這張圖像叫做 cat

##### Document Embedding

_文檔嵌入_

除了 Word Embedding，我們還可以對 Document 做 Embedding

最簡單的方法是把 document 變成 bag-of-word，然後用 Auto-encoder 就可以得到該文檔的語義嵌入(Semantic Embedding)，但光這麼做是不夠的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/se.png" width="60%"/></center>

詞彙的順序代表了很重要的含義，兩句詞彙相同但語序不同的話可能會有完全不同的含義，比如

- 白血球消滅了傳染病——正面語義
- 傳染病消滅了白血球——負面語義

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/se2.png" width="60%"/></center>

想要解決這個問題，具體可以參考下面的幾種處理方法：

- **Paragraph Vector**: _Le, Quoc, and Tomas Mikolov. "Distributed Representations of Sentences and Documents.「 ICML, 2014_
- **Seq2seq Auto-encoder**: _Li, Jiwei, Minh-Thang Luong, and Dan Jurafsky. "A hierarchical neural autoencoder for paragraphs and documents." arXiv preprint, 2015_
- **Skip Thought**: _Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler, 「Skip-Thought Vectors」 arXiv preprint, 2015._

關於**word2vec**，可以參考博客：http://blog.csdn.net/itplus/article/details/37969519
