# Why Deep?

> 本文主要圍繞 Deep 這個關鍵詞展開，重點比較了 shallow learning 和 deep learning 的區別：
> shallow：不考慮不同 input 之間的關聯，針對每一種 class 都設計了一個獨立的 model 檢測
> deep：考慮了 input 之間的某些共同特徵，所有 class 用同個 model 分類，share 參數，modularization 思想，hierarchy 架構，更有效率地使用 data 和參數

#### Shallow V.s. Deep

##### Deep is Better？

我們都知道 deep learning 在很多問題上的表現都是比較好的，越 deep 的 network 一般都會有更好的 performance

那為什麼會這樣呢？有一種解釋是：

- 一個 network 的層數越多，參數就越多，這個 model 就越複雜，它的 bias 就越小，而使用大量的 data 可以降低這個 model 的 variance，performance 當然就會更好

如下圖所示，隨著 layer 層數從 1 到 7，得到的 error rate 不斷地降低，所以有人就認為，deep learning 的表現這麼好，完全就是用大量的 data 去硬 train 一個非常複雜的 model 而得到的結果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deeper1.png" width="60%;" /></center>
既然大量的data加上參數足夠多的model就可以實現這個效果，那為什麼一定要用DNN呢？我們完全可以用一層的shallow neural network來做同樣的事情，理論上只要這一層里neuron的數目足夠多，有足夠的參數，就可以表示出任何函數；那DNN中deep的意義何在呢？

##### Fat + Short v.s. Thin + Tall

其實深和寬這兩種結構的 performance 是會不一樣的，這裡我們就拿下面這兩種結構的 network 做一下比較：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deeper2.png" width="60%;" /></center>
值得注意的是：如果要給Deep和Shallow的model一個公平的評比，你就要故意調整它們的形狀，讓它們的參數是一樣多的，在這個情況下Shallow的model就會是一個矮胖的model，Deep的model就會是一個瘦高的model

在這個公平的評比之下，得到的結果如下圖所示：

左側表示的是 deep network 的情況，右側表示的是 shallow network 的情況，為了保證兩種情況下參數的數量是比較接近的，因此設置了右側 1\*3772 和 1\*4634 這兩種 size 大小，它們分別對應比較左側 5\*2k 和 7\*2k 這兩種情況下的 network(注意參數數目和 neuron 的數目並不是等價的)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deeper3.png" width="60%;" /></center>
這個時候你會發現，在參數數量接近的情況下，只有1層的network，它的error rate是遠大於好幾層的network的；這裡甚至測試了1\*16k大小的shallow network，把它跟左側也是只有一層，但是沒有那麼寬的network進行比較，由於參數比較多所以才略有優勢；但是把1\*16k大小的shallow network和參數遠比它少的2\*2k大小的deep network進行比較，結果竟然是後者的表現更好

也就是說，只有 1 層的 shallow network 的 performance 甚至都比不過很多參數比它少但層數比它多的 deep network，這是為什麼呢？

有人覺得 deep learning 就是一個暴力輾壓的方法，我可以弄一個很大很大的 model，然後 collect 一大堆的 data，就可以得到比較好的 performance；但根據上面的對比可知，deep learning 顯然是在結構上存在著某種優勢，不然無法解釋它會比參數數量相同的 shallow learning 表現得更好這個現象

#### Modularization

##### introduction

DNN 結構一個很大的優勢是，Modularization(模塊化)，它用的是結構化的架構

就像寫程序一樣，shallow network 實際上就是把所有的程序都寫在了同一個 main 函數中，所以它去檢測不同的 class 使用的方法是相互獨立的；而 deep network 則是把整個任務分為了一個個小任務，每個小任務又可以不斷細分下去，以形成 modularization，就像下圖一樣

在 DNN 的架構中，實際上每一層 layer 里的 neuron 都像是在解決同一個級別的任務，它們的 output 作為下一層 layer 處理更高級別任務的數據來源，低層 layer 里的 neuron 做的是對不同小特徵的檢測，高層 layer 里的 neuron 則根據需要挑選低層 neuron 所抽取出來的不同小特徵，去檢測一個範圍更大的特徵；neuron 就像是一個個 classifier ，後面的 classifier 共享前面 classifier 的參數

這樣做的好處是，低層的 neuron 輸出的信息可以被高層不同的 neuron 重復使用，而並不需要像 shallow network 一樣，每次在用到的時候都要重新去檢測一遍，因此大大降低了程序的複雜度

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/modularization1.png" width="60%;" /></center>
##### example

這裡舉一個分類的例子，我們要把 input 的人物分為四類：長頭髮女生、長頭髮男生、短頭髮女生、短頭髮男生

如果按照 shallow network 的想法，我們分別獨立地 train 四個 classifier(其實就相當於訓練四個獨立的 model)，然後就可以解決這個分類的問題；但是這裡有一個問題，長頭髮男生的 data 是比較少的，沒有太多的 training data，所以，你 train 出來的 classifier 就比較 weak，去 detect 長頭髮男生的 performance 就比較差

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/modularization2.png" width="60%;" /></center>
但其實我們的input並不是沒有關聯的，長頭髮的男生和長頭髮的女生都有一個共同的特徵，就是長頭髮，因此如果我們分別**獨立地訓練四個model作為分類器**，實際上就是忽視了這個共同特徵，也就是沒有高效地用到data提供的全部信息，這恰恰是shallow network的弊端

而利用 modularization 的思想，使用 deep network 的架構，我們可以**訓練一個 model 作為分類器就可以完成所有的任務**，我們可以把整個任務分為兩個子任務：

- Classifier1：檢測是男生或女生
- Classifier2：檢測是長頭髮或短頭髮

雖然長頭髮的男生 data 很少，但長頭髮的人的 data 就很多，經過前面幾層 layer 的特徵抽取，就可以頭髮的 data 全部都丟給 Classifier2，把男生或女生的 data 全部都丟給 Classifier1，這樣就真正做到了充分、高效地利用數據，最終的 Classifier 再根據 Classifier1 和 Classifier2 提供的信息給出四類人的分類結果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/modularization3.png" width="60%;" /></center>
你會發現，經過層層layer的任務分解，其實每一個Classifier要做的事情都是比較簡單的，又因為這種分層的、模組化的方式充分利用了data，並提高了信息利用的效率，所以只要用比較少的training data就可以把結果train好

##### deep -> modularization

做 modularization 的好處是**把原來比較複雜的問題變得簡單**，比如原來的任務是檢測一個長頭髮的女生，但現在你的任務是檢測長頭髮和檢測性別，而當檢測對象變簡單的時候，就算 training data 沒有那麼多，我們也可以把這個 task 做好，並且**所有的 classifier 都用同一組參數檢測子特徵**，提高了參數使用效率，這就是 modularization、這就是模塊化的精神

==**由於 deep learning 的 deep 就是在做 modularization 這件事，所以它需要的 training data 反而是比較少的**==，這可能會跟你的認知相反，AI=big data+deep learning，但 deep learning 其實是為瞭解決 less data 的問題才提出的

每一個 neuron 其實就是一個 basic 的 classifier：

- 第一層 neuron，它是一個最 basic 的 classifier，檢測的是顏色、線條這樣的小特徵
- 第二層 neuron 是比較複雜的 classifier，它用第一層 basic 的 classifier 的 output 當作 input，也就是把第一層的 classifier 當作 module，利用第一層得到的小特徵分類出不同樣式的花紋
- 而第三層的 neuron 又把第二層的 neuron 當作它 module，利用第二層得到的特徵分類出蜂窩、輪胎、人
- 以此類推

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/modularization4.png" width="60%;" /></center>
這邊要強調的是，在做deep learning的時候，怎麼做模塊化這件事情是machine自動學到的，也就是說，第一層要檢測什麼特徵、第二層要檢測什麼特徵...這些都不是人為指定的，人只有定好有幾層layer、每層layer有幾個neuron，剩下的事情都是machine自己學到的

傳統的機器學習算法，是人為地根據 domain knowledge 指定特徵來進行提取，這種指定的提取方式，甚至是提取到的特徵，也許並不是實際最優的，所以它的識別成功率並沒有那麼高；但是如果提取什麼特徵、怎麼提取這件事讓機器自己去學，它所提取的就會是那個最優解，因此識別成功率普遍會比人為指定要來的高

#### Speech

前面講了 deep learning 的好處來自於 modularization(模塊化)，可以用比較 efficient 的方式來使用 data 和參數，這裡以語音識別為例，介紹 DNN 的 modularization 在語音領域的應用

##### language basics

當你說 what do you think 的時候，這句話其實是由一串 phoneme 所組成的，所謂 phoneme，中文翻成音素，它是由語言學家制訂的人類發音的基本單位，what 由 4 個 phoneme 組成，do 由兩個 phoneme 組成，you 由兩個 phoneme 組成，等等

同樣的 phoneme 也可能會有不太一樣的發音，當你發 d uw 和 y uw 的時候，心裡想要發的都是 uw，但由於人類發音器官的限制，你的 phoneme 發音會受到前後的 phoneme 所影響；所以，為了表達這一件事情，我們會給同樣的 phoneme 不同的 model，這個東西就叫做 tri-phone

一個 phoneme 可以拆成幾個 state，我們通常就訂成 3 個 state

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech1.png" width="60%;" /></center>
以上就是人類語言的基本構架

##### process

語音辨識的過程其實非常複雜，這裡只是講語音辨識的第一步

你首先要做的事情是把 acoustic feature(聲學特徵)轉成 state，這是一個單純的 classification 的 problem

大致過程就是在一串 wave form(聲音信號)上面取一個 window(通常不會取太大，比如 250 個 mini second 大小)，然後用 acoustic feature 來描述這個 window 裡面的特性，每隔一個時間段就取一個 window，一段聲音信號就會變成一串 vector sequence，這個就叫做 acoustic feature sequence

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech2.png" width="60%;" /></center>
你要建一個Classifier去識別acoustic feature屬於哪個state，再把state轉成phoneme，然後把phoneme轉成文字，接下來你還要考慮同音異字的問題...這裡不會詳細講述整個過程，而是想要比較一下過去在用deep learning之前和用deep learning之後，在語音辨識上的分類模型有什麼差異

##### classification

###### 傳統做法

傳統的方法叫做 HMM-GMM

GMM，即 Gaussian Mixture Model ，它假設語音里的**每一個 state 都是相互獨立的**(跟前面長頭髮的 shallow 例子很像，也是假設每種情況相互獨立)，因此屬於每個 state 的 acoustic feature 都是 stationary distribution(靜態分布)的，因此我們可以針對每一個 state 都訓練一個 GMM model 來識別

但這個方法其實不太現實，因為要列舉的 model 數目太多了，一般中英文都有 30 幾、將近 40 個 phoneme，那這邊就假設是 30 個，而在 tri-phone 裡面，每一個 phoneme 隨著 contest(上下文)的不同又有變化，假設 tri-phone 的形式是 a-b-c，那總共就有 30\*30\*30=27000 個 tri-phone，而每一個 tri-phone 又有三個 state，每一個 state 都要很用一個 GMM 來描述，那參數實在是太多了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech3.png" width="60%;" /></center>
在有deep learning之前的傳統處理方法是，讓一些不同的state共享同樣的model distribution，這件事情叫做Tied-state，其實實際操作上就把state當做pointer(指針)，不同的pointer可能會指向同樣的distribution，所以有一些state的distribution是共享的，具體哪些state共享distribution則是由語言學等專業知識決定

那這樣的處理方法太粗糙了，所以又有人提出了 subspace GMM，它裡面其實就有 modularization、有模塊化的影子，它的想法是，我們先找一個 Gaussian pool(裡面包含了很多不同的 Gaussian distribution)，每一個 state 的 information 就是一個 key，它告訴我們這個 state 要從 Gaussian pool 裡面挑選哪些 Gaussian 出來

比如有某一個 state 1，它挑第一、第三、第五個 Gaussian；另一個 state 2，它挑第一、第四、第六個 Gaussian；如果你這樣做，這些 state 有些時候就可以 share 部分的 Gaussian，那有些時候就可以完全不 share Gaussian，至於要 share 多少 Gaussian，這都是可以從 training data 中學出來的

###### 思考

HMM-GMM 的方法，默認把所有的 phone 或者 state 都看做是無關聯的，對它們分別訓練 independent model，這其實是不 efficient 的，它沒有充分利用 data 提供的信息

對人類的聲音來說，不同的 phoneme 都是由人類的發音器官所 generate 出來的，它們並不是完全無關的，下圖畫出了人類語言裡面所有的元音，這些元音的發音其實就只受到三件事情的影響：

- 舌頭的前後位置
- 舌頭的上下位置
- 嘴型

比如圖中所標英文的 5 個元音 a，e，i，o，u，當你發 a 到 e 到 i 的時候，舌頭是由下往上；而 i 跟 u，則是舌頭放在前面或放在後面的差別；在圖中同一個位置的元音，它們舌頭的位置是一樣的，只是嘴型不一樣

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech4.png" width="60%;" /></center>
###### DNN做法

如果採用 deep learning 的做法，就是去 learn 一個 deep neural network，這個 deep neural network 的 input 是一個 acoustic feature，它的 output 就是該 feature 屬於某個 state 的概率，這就是一個簡單的 classification problem

那這邊最關鍵的一點是，所有的 state 識別任務都是用同一個 DNN 來完成的；值得注意的是 DNN 並不是因為參數多取勝的，實際上在 HMM-GMM 里用到的參數數量和 DNN 其實是差不多的，區別只是 GMM 用了很多很小的 model ，而 DNN 則用了一個很大的 model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech5.png" width="60%;" /></center>
DNN把所有的state通通用同一個model來做分類，會是一種比較有效率的做法，解釋如下

我們拿一個 hidden layer 出來，然後把這個 layer 里所有 neuron 的 output 降維到 2 維得到下圖，每個點的顏色對應著 input a，e，i，o，u，神奇的事情發生了：降維圖上這 5 個元音的分布跟右上角元音位置圖的分布幾乎是一樣的

因此，DNN 並不是馬上就去檢測發音是屬於哪一個 phone 或哪一個 state，比較 lower 的 layer 會先觀察人是用什麼樣的方式在發這個聲音，人的舌頭位置應該在哪裡，是高是低，是前是後；接下來的 layer 再根據這個結果，去決定現在的發音是屬於哪一個 state 或哪一個 phone

這些 lower 的 layer 是一個人類發音方式的 detector，而所有 phone 的檢測都 share 這同一組 detector 的結果，因此最終的這些 classifier 是 share 了同一組用來 detect 發音方式的參數，這就做到了模塊化，同一個參數被更多的地方 share，因此顯得更有效率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech6.png" width="60%;" /></center>
##### result

這個時候就可以來回答之前在[8_Deep Learning]()中提到的問題了

Universality Theorem 告訴我們任何的 continuous 的 function 都可以用一層足夠寬的 neural network 來實現，在 90 年代，這是很多人放棄做 deep learning 的一個原因

但是這個理論只告訴了我們可能性，卻沒有說明這件事的效率問題；根據上面的幾個例子我們已經知道，只用一個 hidden layer 來描述 function 其實是沒有效率的；當你用 multi-layer，用 hierarchy structure 來描述 function 的時候，才會是比較有效率的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech7.png" width="60%;" /></center>
#### Analogy

下面用邏輯電路和剪窗花的例子來更形象地描述 Deep 和 shallow 的區別

##### Logic Circuit

==**邏輯電路其實可以拿來類比神經網絡**==

- 每一個邏輯門就相當於一個 neuron

- 只要兩級邏輯門就可以表示任何的 boolean function；有一個 hidden layer 的 network(input layer+hidden layer 共兩層)可以表示任何 continuous 的 function

  注：邏輯門只要根據 input 的 0、1 狀態和對應的 output 分別建立起門電路關係即可建立兩級電路

- 實際設計電路的時候，為了節約成本，會進行多級優化，建立起 hierarchy 架構，如果某一個結構的邏輯門組合被頻繁用到的話，其實在優化電路里，這個組合是可以被多個門電路共享的，這樣用比較少的邏輯門就可以完成一個電路；在 deep neural network 里，踐行 modularization 的思想，許多 neuron 作為子特徵檢測器被多個 classifier 所共享，本質上就是參數共享，就可以用比較少的參數就完成同樣的 function

  比較少的參數意味著不容易 overfitting，用比較少的 data 就可以完成同樣任務

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/circuits1.png" width="60%;" /></center>
##### 剪窗花

我們之前講過這個邏輯回歸的分類問題，可能會出現下面這種 linear model 根本就沒有辦法分類的問題，而當你加了 hidden layer 的時候，就相當於做了一個 feature transformation，把原來的 x1，x2 轉換到另外一個平面，變成 x1'、x2'

你會發現，通過這個 hidden layer 的轉換，其實就好像把原來這個平面按照對角線對折了一樣，對折後兩個藍色的點就重合在了一起，這個過程跟剪窗花很像：

- 我們在做剪窗花的時候，每次把色紙對折，就相當於把原先的這個多維空間對折了一次來提高維度
- 如果你在某個地方戳一個洞，再把色紙打開，你折了幾折，在對應的這些地方就都會有一個洞；這就相當於在折疊後的高維空間上，畫斜線的部分是某一個 class，不畫斜線的部分是另一個 class，那你在這個高維空間上的某一個點，就相當於展開後空間上的許多點，由於可以對這個空間做各種各樣複雜的對折和剪裁，所以二維平面上無論多少複雜的分類情況，經過多次折疊，不同 class 最後都可以在一個高維空間上以比較明顯的方式被分隔開來

這樣做==**既可以解決某些情況下難以分類的問題，又能夠以比較有效率的方式充分利用 data**==(比如下面這個摺紙，高維空間上的 1 個點等於二維空間上的 5 個點，相當於 1 筆 data 發揮出 5 筆 data 的作用)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/circuits2.png" width="60%;" /></center>
下面舉了一個小例子：

左邊的圖是 training data，右邊則是 1 層 hidden layer 與 3 層 hidden layer 的不同 network 的情況對比，這裡已經控制它們的參數數量趨於相同，試驗結果是，當 training data 為 10w 筆的時候，兩個 network 學到的樣子是比較接近原圖的，而如果只給 2w 筆 training data，1 層 hidden layer 的情況就完全崩掉了，而 3 層 hidden layer 的情況會比較好一些，它其實可以被看作是剪窗花的時候一不小心剪壞了，然後展開得到的結果

注：關於如何得到 model 學到的圖形，可以用固定 model 的參數，然後對 input 進行梯度下降，最終得到結果，具體方法見前幾章

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tony1.png" width="60%;" /></center>
#### End-to-end Learning

##### introduction

所謂的 End-to-end learning，指的是只給 model input 和 output，而不告訴它中間每一個 function 要怎麼分工，讓它自己去學會知道在生產線的每一站，自己應該要做什麼事情；在 DNN 里，就是疊一個很深的 neural network，每一層 layer 就是生產線上的一個站

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/end.png" width="60%;" /></center>
##### Speech Recognition

End-to-end Learning 在語音識別上體現的非常明顯

在傳統的 Speech Recognition 里，只有最後 GMM 這個藍色的 block，才是由 training data 學出來的，前面綠色的生產線部分都是由過去的「五聖先賢」手動制訂出來的，其實制訂的這些 function 非常非常的強，可以說是增一分則太肥，減一分則太瘦這樣子，以至於在這個階段卡了將近 20 年

後來有了 deep learning，我們就可以用 neural network 把 DCT、log 這些部分取代掉，甚至你從 spectrogram 開始都拿 deep neural network 取代掉，也可以得到更好的結果，如果你分析 DNN 的 weight，它其實可以自動學到要做 filter bank 這件事情(filter bank 是模擬人類的聽覺器官所制定出來的 filter)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/speech8.png" width="60%;" /></center>
那能不能夠疊一個很深很深的neural network，input直接就是time domain上的聲音信號，而output直接就是
文字，中間完全不要做feature transform之類，目前的結果是，現在machine做的事情就很像是在做Fourier transform，它學到的極限也只是做到與Fourier feature transform打平而已，或許DFT已經是信號處理的極限了

有關 End-to-end Learning 在 Image Recognition 的應用和 Speech Recognition 很像，這裡不再贅述

#### Complex Task

那 deep learning 還有什麼好處呢？

有時候我們會遇到非常複雜的 task：

- 有時候非常像的 input，它會有很不一樣的 output

  比如在做圖像辨識的時候，下圖這個白色的狗跟北極熊其實看起來是很像的，但是你的 machine 要有能力知道，看到左邊這張圖要 output 狗，看到右邊這張圖要 output 北極熊

- 有時候看起來很不一樣的 input，output 其實是一樣的

  比如下面這兩個方向上看到的火車，橫看成嶺側成峰，儘管看到的很不一樣，但是你的 machine 要有能力知道這兩個都是同一種東西

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/complex.png" width="60%;" /></center>
如果你的network只有一層的話，就只能做簡單的transform，沒有辦法把一樣的東西變得很不一樣，把不一樣的東西變得很像；如果要實現這些，就需要做很多層次的轉換，就像前面那個剪窗花的例子，在二維、三維空間上看起來很難辨別，但是到了高維空間就完全有可能把它們給辨別出來

這裡以 MNIST 手寫數字識別為例，展示一下 DNN 中，在高維空間上對這些 Complex Task 的處理能力

如果把 28\*28 個 pixel 組成的 vector 投影到二維平面上就像左上角所示，你會發現 4 跟 9 的 pixel 幾乎是疊在一起的，因為 4 跟 9 很像，都是一個圈圈再加一條線，所以如果你光看 input 的 pixel 的話，4 跟 9 幾乎是疊在一起的，你幾乎沒有辦法把它分開

但是，等到第二個、第三個 layer 的 output，你會發現 4、7、9 逐漸就被分開了，所以使用 deep learning 的 deep，這也是其中一個理由

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/task2.png" width="60%;" /></center>
#### Conclusion

Deep 總結：

- 考慮 input 之間的內在關聯，所有的 class 用同一個 model 來做分類
- modularization 思想，複雜問題簡單化，把檢測複雜特徵的大任務分割成檢測簡單特徵的小任務
- 所有的 classifier 使用同一組參數的子特徵檢測器，共享檢測到的子特徵
- 不同的 classifier 會 share 部分的參數和 data，效率高
- 聯繫 logic circuit 和剪紙畫的例子
- 多層 hidden layer 對 complex 問題的處理上比較有優勢
