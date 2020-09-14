# Recurrent Neural Network(Ⅰ)

> RNN，或者說最常用的 LSTM，一般用於記住之前的狀態，以供後續神經網絡的判斷，它由 input gate、forget gate、output gate 和 cell memory 組成，每個 LSTM 本質上就是一個 neuron，特殊之處在於有 4 個輸入：$z$和三門控制信號$z_i$、$z_f$和$z_o$，每個時間點的輸入都是由當前輸入值+上一個時間點的輸出值+上一個時間點 cell 值來組成

#### Introduction

##### Slot Filling

在智能客服、智能訂票系統中，往往會需要 slot filling 技術，它會分析用戶說出的語句，將時間、地址等有效的關鍵詞填到對應的槽上，並過濾掉無效的詞語

詞彙要轉化成 vector，可以使用 1-of-N 編碼，word hashing 或者是 word vector 等方式，此外我們可以嘗試使用 Feedforward Neural Network 來分析詞彙，判斷出它是屬於時間或是目的地的概率

但這樣做會有一個問題，該神經網絡會先處理「arrive」和「leave」這兩個詞彙，然後再處理「Taipei」，這時對 NN 來說，輸入是相同的，它沒有辦法區分出「Taipei」是出發地還是目的地

這個時候我們就希望神經網絡是有記憶的，如果 NN 在看到「Taipei」的時候，還能記住之前已經看過的「arrive」或是「leave」，就可以根據上下文得到正確的答案

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-example.png" width="60%"/></center>

這種有記憶力的神經網絡，就叫做 Recurrent Neural Network(RNN)

在 RNN 中，hidden layer 每次產生的 output $a_1$、$a_2$，都會被存到 memory 里，下一次有 input 的時候，這些 neuron 就不僅會考慮新輸入的$x_1$、$x_2$，還會考慮存放在 memory 中的$a_1$、$a_2$

注：在 input 之前，要先給內存里的$a_i$賦初始值，比如 0

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn.png" width="60%"/></center>

注意到，每次 NN 的輸出都要考慮 memory 中存儲的臨時值，而不同的輸入產生的臨時值也盡不相同，因此改變輸入序列的順序會導致最終輸出結果的改變(Changing the sequence order will change the output)

##### Slot Filling with RNN

用 RNN 處理 Slot Filling 的流程舉例如下：

- 「arrive」的 vector 作為$x^1$輸入 RNN，通過 hidden layer 生成$a^1$，再根據$a^1$生成$y^1$，表示「arrive」屬於每個 slot 的概率，其中$a^1$會被存儲到 memory 中
- 「Taipei」的 vector 作為$x^2$輸入 RNN，此時 hidden layer 同時考慮$x^2$和存放在 memory 中的$a^1$，生成$a^2$，再根據$a^2$生成$y^2$，表示「Taipei」屬於某個 slot 的概率，此時再把$a^2$存到 memory 中
- 依次類推

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-example2.png" width="60%"/></center>

注意：上圖為同一個 RNN 在三個不同時間點被分別使用了三次，並非是三個不同的 NN

這個時候，即使輸入同樣是「Taipei」，我們依舊可以根據前文的「leave」或「arrive」來得到不一樣的輸出

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-example3.png" width="60%"/></center>

##### Elman Network & Jordan Network

RNN 有不同的變形：

- Elman Network：將 hidden layer 的輸出保存在 memory 里
- Jordan Network：將整個 neural network 的輸出保存在 memory 里

由於 hidden layer 沒有明確的訓練目標，而整個 NN 具有明確的目標，因此 Jordan Network 的表現會更好一些

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-type.png" width="60%"/></center>

##### Bidirectional RNN

RNN 還可以是雙向的，你可以同時訓練一對正向和反向的 RNN，把它們對應的 hidden layer $x^t$拿出來，都接給一個 output layer，得到最後的$y^t$

使用 Bi-RNN 的好處是，NN 在產生輸出的時候，它能夠看到的範圍是比較廣的，RNN 在產生$y^{t+1}$的時候，它不只看了從句首$x^1$開始到$x^{t+1}$的輸入，還看了從句尾$x^n$一直到$x^{t+1}$的輸入，這就相當於 RNN 在看了整個句子之後，才決定每個詞彙具體要被分配到哪一個槽中，這會比只看句子的前一半要更好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rnn-bi.png" width="60%"/></center>

#### LSTM

前文提到的 RNN 只是最簡單的版本，並沒有對 memory 的管理多加約束，可以隨時進行讀取，而現在常用的 memory 管理方式叫做長短期記憶(Long Short-term Memory)，簡稱 LSTM

冷知識：可以被理解為比較長的短期記憶，因此是 short-term，而非是 long-short term

##### Three-gate

LSTM 有三個 gate：

- 當某個 neuron 的輸出想要被寫進 memory cell，它就必須要先經過一道叫做**input gate**的閘門，如果 input gate 關閉，則任何內容都無法被寫入，而關閉與否、什麼時候關閉，都是由神經網絡自己學習到的

- output gate 決定了外界是否可以從 memory cell 中讀取值，當**output gate**關閉的時候，memory 裡面的內容同樣無法被讀取
- forget gate 則決定了什麼時候需要把 memory cell 里存放的內容忘記清空，什麼時候依舊保存

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm.png" width="60%"/></center>

整個 LSTM 可以看做是 4 個 input，1 個 output：

- 4 個 input=想要被存到 memory cell 里的值+操控 input gate 的信號+操控 output gate 的信號+操控 forget gate 的信號
- 1 個 output=想要從 memory cell 中被讀取的值

##### Memory Cell

如果從表達式的角度看 LSTM，它比較像下圖中的樣子

- $z$是想要被存到 cell 里的輸入值
- $z_i$是操控 input gate 的信號
- $z_o$是操控 output gate 的信號
- $z_f$是操控 forget gate 的信號
- $a$是綜合上述 4 個 input 得到的 output 值

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm2.png" width="60%"/></center>

把$z$、$z_i$、$z_o$、$z_f$通過 activation function，分別得到$g(z)$、$f(z_i)$、$f(z_o)$、$f(z_f)$

其中對$z_i$、$z_o$和$z_f$來說，它們通過的激活函數$f()$一般會選 sigmoid function，因為它的輸出在 0\~1 之間，代表 gate 被打開的程度

令$g(z)$與$f(z_i)$相乘得到$g(z)\cdot f(z_i)$，然後把原先存放在 cell 中的$c$與$f(z_f)$相乘得到$cf(z_f)$，兩者相加得到存在 memory 中的新值$c'=g(z)\cdot f(z_i)+cf(z_f)$

- 若$f(z_i)=0$，則相當於沒有輸入，若$f(z_i)=1$，則相當於直接輸入$g(z)$
- 若$f(z_f)=1$，則保存原來的值$c$並加到新的值上，若$f(z_f)=0$，則舊的值將被遺忘清除

從中也可以看出，forget gate 的邏輯與我們的直覺是相反的，控制信號打開表示記得，關閉表示遺忘

此後，$c'$通過激活函數得到$h(c')$，與 output gate 的$f(z_o)$相乘，得到輸出$a=h(c')f(z_o)$

##### LSTM Example

下圖演示了一個 LSTM 的基本過程，$x_1$、$x_2$、$x_3$是輸入序列，$y$是輸出序列，基本原則是：

- 當$x_2=1$時，將$x_1$的值寫入 memory
- 當$x_2=-1$時，將 memory 里的值清零
- 當$x_3=1$時，將 memory 里的值輸出
- 當 neuron 的輸入為正時，對應 gate 打開，反之則關閉

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm3.png" width="60%"/></center>

##### LSTM Structure

你可能會覺得上面的結構與平常所見的神經網絡不太一樣，實際上我們只需要把 LSTM 整體看做是下面的一個 neuron 即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm4.png" width="50%"/></center>

假設目前我們的 hidden layer 只有兩個 neuron，則結構如下圖所示：

- 輸入$x_1$、$x_2$會分別乘上四組不同的 weight，作為 neuron 的輸入以及三個狀態門的控制信號
- 在原來的 neuron 里，1 個 input 對應 1 個 output，而在 LSTM 里，4 個 input 才產生 1 個 output，並且所有的 input 都是不相同的
- 從中也可以看出 LSTM 所需要的參數量是一般 NN 的 4 倍

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm5.png" width="60%"/></center>

##### LSTM for RNN

從上圖中你可能看不出 LSTM 與 RNN 有什麼關係，接下來我們用另外的圖來表示它

假設我們現在有一整排的 LSTM 作為 neuron，每個 LSTM 的 cell 里都存了一個 scalar 值，把所有的 scalar 連接起來就組成了一個 vector $c^{t-1}$

在時間點$t$，輸入了一個 vector $x^t$，它會乘上一個 matrix，通過轉換得到$z$，而$z$的每個 dimension 就代表了操控每個 LSTM 的輸入值，同理經過不同的轉換得到$z^i$、$z^f$和$z^o$，得到操控每個 LSTM 的門信號

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm6.png" width="60%"/></center>

下圖是單個 LSTM 的運算情景，其中 LSTM 的 4 個 input 分別是$z$、$z^i$、$z^f$和$z^o$的其中 1 維，每個 LSTM 的 cell 所得到的 input 都是各不相同的，但它們卻是可以一起共同運算的，整個運算流程如下圖左側所示：

$f(z^f)$與上一個時間點的 cell 值$c^{t-1}$相乘，並加到經過 input gate 的輸入$g(z)\cdot f(z^i)$上，得到這個時刻 cell 中的值$c^t$，最終再乘上 output gate 的信號$f(z^o)$，得到輸出$y^t$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm7.png" width="60%"/></center>

上述的過程反復進行下去，就得到下圖中各個時間點上，LSTM 值的變化情況，其中與上面的描述略有不同的是，這裡還需要把 hidden layer 的最終輸出$y^t$以及當前 cell 的值$c^t$都連接到下一個時間點的輸入上

因此在下一個時間點操控這些 gate 值，不只是看輸入的$x^{t+1}$，還要看前一個時間點的輸出$h^t$和 cell 值$c^t$，你需要把$x^{t+1}$、$h^t$和$c^t$這 3 個 vector 並在一起，乘上 4 個不同的轉換矩陣，去得到 LSTM 的 4 個輸入值$z$、$z^i$、$z^f$、$z^o$，再去對 LSTM 進行操控

注意：下圖是**同一個**LSTM 在兩個相鄰時間點上的情況

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm8.png" width="60%"/></center>

上圖是單個 LSTM 作為 neuron 的情況，事實上 LSTM 基本上都會疊多層，如下圖所示，左邊兩個 LSTM 代表了兩層疊加，右邊兩個則是它們在下一個時間點的狀態

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/lstm9.png" width="60%"/></center>
