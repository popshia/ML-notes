# Recurrent Neural Network(Ⅱ)

> 上一篇文章介紹了 RNN 的基本架構，像這麼複雜的結構，我們該如何訓練呢？

#### Learning Target

##### Loss Function

依舊是 Slot Filling 的例子，我們需要把 model 的輸出$y^i$與映射到 slot 的 reference vector 求交叉熵，比如「Taipei」對應到的是「dest」這個 slot，則 reference vector 在「dest」位置上值為 1，其餘維度值為 0

RNN 的 output 和 reference vector 的 cross entropy 之和就是損失函數，也是要 minimize 的對象

需要注意的是，word 要依次輸入 model，比如「arrive」必須要在「Taipei」前輸入，不能打亂語序

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-learn.png" width="60%"/></center>

##### Training

有了損失函數後，訓練其實也是用梯度下降法，為了計算方便，這裡採取了反向傳播(Backpropagation)的進階版，Backpropagation through time，簡稱 BPTT 算法

BPTT 算法與 BP 算法非常類似，只是多了一些時間維度上的信息，這裡不做詳細介紹

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-learn2.png" width="60%"/></center>

不幸的是，RNN 的訓練並沒有那麼容易

我們希望隨著 epoch 的增加，參數的更新，loss 應該要像下圖的藍色曲線一樣慢慢下降，但在訓練 RNN 的時候，你可能會遇到類似綠色曲線一樣的學習曲線，loss 劇烈抖動，並且會在某個時刻跳到無窮大，導致程序運行失敗

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-learn3.png" width="60%"/></center>

##### Error Surface

分析可知，RNN 的 error surface，即 loss 由於參數產生的變化，是非常陡峭崎嶇的

下圖中，$z$軸代表 loss，$x$軸和$y$軸代表兩個參數$w_1$和$w_2$，可以看到 loss 在某些地方非常平坦，在某些地方又非常的陡峭

如果此時你的訓練過程類似下圖中從下往上的橙色的點，它先經過一塊平坦的區域，又由於參數的細微變化跳上了懸崖，這就會導致 loss 上下抖動得非常劇烈

如果你的運氣特別不好，一腳踩在懸崖上，由於之前一直處於平坦區域，gradient 很小，你會把參數更新的步長(learning rate)調的比較大，而踩到懸崖上導致 gradient 突然變得很大，這會導致參數一下子被更新了一個大步伐，導致整個就飛出去了，這就是學習曲線突然跳到無窮大的原因

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-learn4.png" width="60%"/></center>

想要解決這個問題，就要採用 Clipping 方法，當 gradient 即將大於某個 threshold 的時候，就讓它停止增長，比如當 gradient 大於 15 的時候就直接讓它等於 15

為什麼 RNN 會有這種奇特的特性呢？下圖給出了一個直觀的解釋：

假設 RNN 只含 1 個 neuron，它是 linear 的，input 和 output 的 weight 都是 1，沒有 bias，從當前時刻的 memory 值接到下一時刻的 input 的 weight 是$w$，按照時間點順序輸入[1, 0, 0, 0, ..., 0]

當第 1 個時間點輸入 1 的時候，在第 1000 個時間點，RNN 輸出的$y^{1000}=w^{999}$，想要知道參數$w$的梯度，只需要改變$w$的值，觀察對 RNN 的輸出有多大的影響即可：

- 當$w$從 1->1.01，得到的$y^{1000}$就從 1 變到了 20000，這表示$w$的梯度很大，需要調低學習率
- 當$w$從 0.99->0.01，則$y^{1000}$幾乎沒有變化，這表示$w$的梯度很小，需要調高學習率
- 從中可以看出 gradient 時大時小，error surface 很崎嶇，尤其是在$w=1$的周圍，gradient 幾乎是突變的，這讓我們很難去調整 learning rate

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-why.png" width="60%"/></center>

因此我們可以解釋，RNN 訓練困難，是由於它把同樣的操作在不斷的時間轉換中重復使用

從 memory 接到 neuron 輸入的參數$w$，在不同的時間點被反復使用，$w$的變化有時候可能對 RNN 的輸出沒有影響，而一旦產生影響，經過長時間的不斷累積，該影響就會被放得無限大，因此 RNN 經常會遇到這兩個問題：

- 梯度消失(gradient vanishing)，一直在梯度平緩的地方停滯不前
- 梯度爆炸(gradient explode)，梯度的更新步伐邁得太大導致直接飛出有效區間

#### Help Techniques

有什麼技巧可以幫我們解決這個問題呢？LSTM 就是最廣泛使用的技巧，它會把 error surface 上那些比較平坦的地方拿掉，從而解決梯度消失(gradient vanishing)的問題，但它無法處理梯度崎嶇的部分，因而也就無法解決梯度爆炸的問題(gradient explode)

但由於做 LSTM 的時候，大部分地方的梯度變化都很劇烈，因此訓練時可以放心地把 learning rate 設的小一些

Q：為什麼要把 RNN 換成 LSTM？A：LSTM 可以解決梯度消失的問題

Q：為什麼 LSTM 能夠解決梯度消失的問題？

A：RNN 和 LSTM 對 memory 的處理其實是不一樣的：

- 在 RNN 中，每個新的時間點，memory 里的舊值都會被新值所覆蓋
- 在 LSTM 中，每個新的時間點，memory 里的值會乘上$f(g_f)$與新值相加

對 RNN 來說，$w$對 memory 的影響每次都會被清除，而對 LSTM 來說，除非 forget gate 被打開，否則$w$對 memory 的影響就不會被清除，而是一直累加保留，因此它不會有梯度消失的問題

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-tech.png" width="60%"/></center>

另一個版本 GRU (Gated Recurrent Unit)，只有兩個 gate，需要的參數量比 LSTM 少，魯棒性比 LSTM 好，不容易過擬合，它的基本精神是舊的不去，新的不來，GRU 會把 input gate 和 forget gate 連起來，當 forget gate 把 memory 里的值清空時，input gate 才會打開，再放入新的值

此外，還有很多技術可以用來處理梯度消失的問題，比如 Clockwise RNN、SCRN 等

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-tech2.png" width="60%"/></center>

#### More Applications

在 Slot Filling 中，我們輸入一個 word vector 輸出它的 label，除此之外 RNN 還可以做更複雜的事情

- 多對一
- 多對多

##### Sentiment Analysis

語義情緒分析，我們可以把某影片相關的文章爬下來，並分析其正面情緒 or 負面情緒

RNN 的輸入是字符序列，在不同時間點輸入不同的字符，並在最後一個時間點輸出該文章的語義情緒

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app.png" width="60%"/></center>

##### Key term Extraction

關鍵詞分析，RNN 可以分析一篇文章並提取出其中的關鍵詞，這裡需要把含有關鍵詞標籤的文章作為 RNN 的訓練數據

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app2.png" width="60%"/></center>

##### Output is shorter

如果輸入輸出都是 sequence，且輸出的 sequence 比輸入的 sequence 要短，RNN 可以處理這個問題

以語音識別為例，輸入是一段聲音信號，每隔一小段時間就用 1 個 vector 來表示，因此輸入為 vector sequence，而輸出則是 character vector

如果依舊使用 Slot Filling 的方法，只能做到每個 vector 對應 1 個輸出的 character，識別結果就像是下圖中的「好好好棒棒棒棒棒」，但這不是我們想要的，可以使用 Trimming 的技術把重復內容消去，剩下「好棒」

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app3.png" width="60%"/></center>

但「好棒」和「好棒棒」實際上是不一樣的，如何區分呢？

需要用到 CTC 算法，它的基本思想是，輸出不只是字符，還要填充 NULL，輸出的時候去掉 NULL 就可以得到連詞的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app4.png" width="60%"/></center>

下圖是 CTC 的示例，RNN 的輸出就是英文字母+NULL，google 的語音識別系統就是用 CTC 實現的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app5.png" width="60%"/></center>

##### Sequence to Sequence Learning

在 Seq2Seq 中，RNN 的輸入輸出都是 sequence，但是長度不同

在 CTC 中，input 比較長，output 比較短；而在 Seq2Seq 中，並不確定誰長誰短

比如現在要做機器翻譯，將英文的 word sequence 翻譯成中文的 character sequence

假設在兩個時間點分別輸入「machine」和「learning」，則在最後 1 個時間點 memory 就存了整個句子的信息，接下來讓 RNN 輸出，就會得到「機」，把「機」當做 input，並讀取 memory 里的值，就會輸出「器」，依次類推，這個 RNN 甚至會一直輸出，不知道什麼時候會停止

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app6.png" width="60%"/></center>

怎樣才能讓機器停止輸出呢？

可以多加一個叫做「斷」的 symbol 「===」，當輸出到這個 symbol 時，機器就停止輸出

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app7.png" width="60%"/></center>

具體的處理技巧這裡不再詳述

##### Seq2Seq for Syntatic Parsing

Seq2Seq 還可以用在句法解析上，讓機器看一個句子，它可以自動生成樹狀的語法結構圖

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app8.png" width="60%"/></center>

##### Seq2Seq for Auto-encoder Text

如果用 bag-of-word 來表示一篇文章，就很容易丟失詞語之間的聯繫，丟失語序上的信息

比如「白血球消滅了感染病」和「感染病消滅了白血球」，兩者 bag-of-word 是相同的，但語義卻是完全相反的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app9.png" width="60%"/></center>

這裡就可以使用 Seq2Seq Autoencoder，在考慮了語序的情況下，把文章編碼成 vector，只需要把 RNN 當做編碼器和解碼器即可

我們輸入 word sequence，通過 RNN 變成 embedded vector，再通過另一個 RNN 解壓回去，如果能夠得到一模一樣的句子，則壓縮後的 vector 就代表了這篇文章中最重要的信息

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app10.png" width="60%"/></center>

這個結構甚至可以被層次化，我們可以對句子的幾個部分分別做 vector 的轉換，最後合併起來得到整個句子的 vector

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app11.png" width="60%"/></center>

##### Seq2Seq for Auto-encoder Speech

Seq2Seq autoencoder 還可以用在語音處理上，它可以把一段語音信號編碼成 vector

這種方法可以把聲音信號都轉化為低維的 vecotr，並通過計算相似度來做語音搜索

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app12.png" width="60%"/></center>

先把聲音信號轉化成聲學特徵向量(acoustic features)，再通過 RNN 編碼，最後一個時間點存在 memory 里的值就代表了整個聲音信號的信息

為了能夠對該神經網絡訓練，還需要一個 RNN 作為解碼器，得到還原後的$y_i$，使之與$x_i$的差距最小

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app13.png" width="60%"/></center>

##### Attention-based Model

除了 RNN 之外，Attention-based Model 也用到了 memory 的思想

機器會有自己的記憶池，神經網絡通過操控讀寫頭去讀或者寫指定位置的信息，這個過程跟圖靈機很像，因此也被稱為 neural turing machine

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-app14.png" width="60%"/></center>

這種方法通常用在閱讀理解上，讓機器讀一篇文章，再把每句話的語義都存到不同的 vector 中，接下來讓用戶向機器提問，神經網絡就會去調用讀寫頭的中央處理器，取出 memory 中與查詢語句相關的信息，綜合處理之後，可以給出正確的回答
