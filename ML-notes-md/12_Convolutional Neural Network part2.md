# Convolutional Neural Network part2

> 人們常常會說，deep learning 就是一個黑盒子，你 learn 完以後根本就不知道它得到了什麼，所以會有很多人不喜歡這種方法，這篇文章就講述了三個問題：What does CNN do？Why CNN？How to design CNN?

#### What does CNN learn？

##### what is intelligent

如果今天有一個方法，它可以讓你輕易地理解為什麼這個方法會下這樣的判斷和決策的話，那其實你會覺得它不夠 intelligent；它必須要是你無法理解的東西，這樣它才夠 intelligent，至少你會感覺它很 intelligent

所以，大家常說 deep learning 就是一個黑盒子，你 learn 出來以後，根本就不知道為什麼是這樣子，於是你會感覺它很 intelligent，但是其實還是有很多方法可以分析的，今天我們就來示範一下怎麼分析 CNN，看一下它到底學到了什麼

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-keras3.png" width="60%;"></center>
要分析第一個convolution的filter是比較容易的，因為第一個convolution layer裡面，每一個filter就是一個3\*3的matrix，它對應到3\*3範圍內的9個pixel，所以你只要看這個filter的值，就可以知道它在detect什麼東西，因此第一層的filter是很容易理解的

但是你比較沒有辦法想像它在做什麼事情的，是第二層的 filter，它們是 50 個同樣為 3\*3 的 filter，但是這些 filter 的 input 並不是 pixel，而是做完 convolution 再做 Max pooling 的結果，因此 filter 考慮的範圍並不是 3\*3=9 個 pixel，而是一個長寬為 3\*3，高為 25 的 cubic，filter 實際在 image 上看到的範圍是遠大於 9 個 pixel 的，所以你就算把它的 weight 拿出來，也不知道它在做什麼

##### what does filter do

那我們怎麼來分析一個 filter 它做的事情是什麼呢？你可以這樣做：

我們知道在第二個 convolution layer 裡面的 50 個 filter，每一個 filter 的 output 就是一個 11\*11 的 matrix，假設我們現在把第 k 個 filter 的 output 拿出來，如下圖所示，這個 matrix 里的每一個 element，我們叫它$a^k_{ij}$，上標 k 表示這是第 k 個 filter，下標 ij 表示它在這個 matrix 里的第 i 個 row，第 j 個 column

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/kth-filter.png" width="60%;" /></center>
接下來我們define一個$a^k$叫做**Degree of the activation of the k-th filter**，這個值表示現在的第k個filter，它有多被activate，有多被「啓動」，直觀來講就是描述現在input的東西跟第k個filter有多接近，它對filter的激活程度有多少

第 k 個 filter 被啓動的 degree $a^k$就定義成，它與 input 進行卷積所輸出的 output 里所有 element 的 summation，以上圖為例，就是這 11\*11 的 output matrix 里所有元素之和，用公式描述如下：

$$
a^k=\sum\limits^{11}_{i=1}\sum\limits^{11}_{j=1} a^k_{ij}
$$

也就是說，我們 input 一張 image，然後把這個 filter 和 image 進行卷積所 output 的 11\*11 個值全部加起來，當作現在這個 filter 被 activate 的程度

接下來我們要做的事情是這樣子，我們想要知道第 k 個 filter 的作用是什麼，那我們就要找一張 image，這張 image 可以讓第 k 個 filter 被 activate 的程度最大；於是我們現在要解的問題是，找一個 image x，它可以讓我們定義的 activation 的 degree $a^k$最大，即：

$$
x^*=\arg \max\limits_x a^k
$$

之前我們求 minimize 用的是 gradient descent，那現在我們求 Maximum 用 gradient ascent(梯度上升法)就可以做到這件事了

仔細一想這個方法還是頗為神妙的，因為我們現在是把 input x 作為要找的參數，對它去用 gradient descent 或 ascent 進行 update，原來在 train CNN 的時候，input 是固定的，model 的參數是要用 gradient descent 去找出來的；但是現在這個立場是反過來的，在這個 task 裡面 model 的參數是固定的，我們要用 gradient ascent 去 update 這個 x，讓它可以使 degree of activation 最大

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/gradient-ascent.png" width="60%;" /></center>
上圖就是得到的結果，50個filter理論上可以分別找50張image使對應的activation最大，這裡僅挑選了其中的12張image作為展示，這些image有一個共同的特徵，它們裡面都是一些**反復出現的某種texture(紋路)**，比如說第三張image上布滿了小小的斜條紋，這意味著第三個filter的工作就是detect圖上有沒有斜條紋，要知道現在每個filter檢測的都只是圖上一個小小的範圍而已，所以圖中一旦出現一個小小的斜條紋，這個filter就會被activate，相應的output也會比較大，所以如果整張image上布滿這種斜條紋的話，這個時候它會最興奮，filter的activation程度是最大的，相應的output值也會達到最大

因此每個 filter 的工作就是去 detect 某一種 pattern，detect 某一種線條，上圖所示的 filter 所 detect 的就是不同角度的線條，所以今天 input 有不同線條的話，某一個 filter 會去找到讓它興奮度最高的匹配對象，這個時候它的 output 就是最大的

##### what does neuron do

我們做完 convolution 和 max pooling 之後，會將結果用 Flatten 展開，然後丟到 Fully connected 的 neural network 裡面去，之前已經搞清楚了 filter 是做什麼的，那我們也想要知道在這個 neural network 里的每一個 neuron 是做什麼的，所以就對剛才的做法如法炮製

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/neuron-do.png" width="60%;" /></center>
我們定義第j個neuron的output就是$a_j$，接下來就用gradient ascent的方法去找一張image x，把它丟到neural network裡面就可以讓$a_j$的值被maximize，即：
$$
x^*=\arg \max\limits_x a^j
$$
找到的結果如上圖所示，同理這裡僅取出其中的9張image作為展示，你會發現這9張圖跟之前filter所觀察到的情形是很不一樣的，剛才我們觀察到的是類似紋路的東西，那是因為每個filter考慮的只是圖上一部分的vision，所以它detect的是一種texture；但是在做完Flatten以後，每一個neuron不再是只看整張圖的一小部分，它現在的工作是看整張圖，所以對每一個neuron來說，讓它最興奮的、activation最大的image，不再是texture，而是一個完整的圖形

##### what about output

接下來我們考慮的是 CNN 的 output，由於是手寫數字識別的 demo，因此這裡的 output 就是 10 維，我們把某一維拿出來，然後同樣去找一張 image x，使這個維度的 output 值最大，即

$$
x^*=\arg \max_x y^i
$$

你可以想象說，既然現在每一個 output 的每一個 dimension 就對應到一個數字，那如果我們去找一張 image x，它可以讓對應到數字 1 的那個 output layer 的 neuron 的 output 值最大，那這張 image 顯然應該看起來會像是數字 1，你甚至可以期待，搞不好用這個方法就可以讓 machine 自動畫出數字

但實際上，我們得到的結果是這樣子，如下圖所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-output.png" width="60%;" /></center>
上面的每一張圖分別對應著數字0-8，你會發現，可以讓數字1對應neuron的output值最大的image其實長得一點也不像1，就像是電視機壞掉的樣子，為了驗證程序有沒有bug，這裡又做了一個實驗，把上述得到的image真的作為testing data丟到CNN裡面，結果classify的結果確實還是認為這些image就對應著數字0-8

所以今天這個 neural network，它所學到的東西跟我們人類一般的想象認知是不一樣的

那我們有沒有辦法，讓上面這個圖看起來更像數字呢？想法是這樣的，我們知道一張圖是不是一個數字，它會有一些基本的假設，比如這些 image，你不知道它是什麼數字，你也會認為它顯然就不是一個 digit，因為人類手寫出來的東西就不是長這個樣子的，所以我們要對這個 x 做一些 regularization，我們要對找出來的 x 做一些 constraint(限制約束)，我們應該告訴 machine 說，雖然有一些 x 可以讓你的 y 很大，但是它們不是數字

那我們應該加上什麼樣的 constraint 呢？最簡單的想法是說，畫圖的時候，白色代表的是有墨水、有筆畫的地方，而對於一個 digit 來說，整張 image 上塗白的區域是有限的，像上面這些整張圖都是白白的，它一定不會是數字

假設 image 里的每一個 pixel 都用$x_{ij}$表示，我們把所有 pixel 值取絕對值並求和，也就是$\sum\limits_{i,j}|x_{ij}|$，這一項其實就是之前提到過的 L1 的 regularization，再用$y^i$減去這一項，得到

$$
x^*=\arg \max\limits_x (y^i-\sum\limits_{i,j} |x_{ij}|)
$$

這次我們希望再找一個 input x，它可以讓$y^i$最大的同時，也要讓$|x_ij|$的 summation 越小越好，也就是說我們希望找出來的 image，大部分的地方是沒有塗顏色的，只有少數數字筆畫在的地方才有顏色出現

加上這個 constraint 以後，得到的結果會像下圖右側所示一樣，已經隱約有些可以看出來是數字的形狀了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/L1.png" width="60%;" /></center>
如果再加上一些額外的constraint，比如你希望相鄰的pixel是同樣的顏色等等，你應該可以得到更好的結果

#### Deep Dream

其實，這就是 Deep Dream 的精神，Deep Dream 是說，如果你給 machine 一張 image，它會在這個 image 裡面加上它看到的東西

怎麼做這件事情呢？你就找一張 image 丟到 CNN 裡面去，然後你把某一個 convolution layer 裡面的 filter 或是 fully connected layer 里的某一個 hidden layer 的 output 拿出來，它其實是一個 vector；接下來把本來是 positive 的 dimension 值調大，negative 的 dimension 值調小，也就是讓正的更正，負的更負，然後把它作為新的 image 的目標

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-dream.png" width="60%;" /></center>
這裡就是把3.9、2.3的值調大，-1.5的值調小，總體來說就是使它們的絕對值變大，然後用gradient descent的方法找一張image x，讓它通過這個hidden layer後的output就是你調整後的target，這麼做的目的就是，**讓CNN誇大化它看到的東西**——make CNN exaggerates what is sees

也就是說，如果某個 filter 有被 activate，那你讓它被 activate 的更劇烈，CNN 可能本來看到了某一樣東西，那現在你就讓它看起來更像原來看到的東西，這就是所謂的**誇大化**

如果你把上面這張 image 拿去做 Deep Dream 的話，你看到的結果就會像下面這個樣子

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-dream2.png" width="60%;" /></center>

就好像背後有很多念獸，要凝才看得到，比如像上圖右側那一隻熊，它原來是一個石頭，對機器來說，它看這張圖的時候，本來就覺得這個石頭有點像熊，所以你就更強化這件事，讓它看起來真的就變成了一隻熊，這個就是 Deep Dream

#### Deep Style

Deep Dream 還有一個進階的版本，就叫做 Deep Style，如果今天你 input 一張 image，Deep Style 做的事情就是讓 machine 去修改這張圖，讓它有另外一張圖的風格，如下所示

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-style.png" width="60%;" /></center>

實際上機器做出來的效果驚人的好，具體的做法參考 reference：[A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

這裡僅講述 Deep Style 的大致思路，你把原來的 image 丟給 CNN，得到 CNN filter 的 output，代表這樣 image 裡面有什麼樣的 content，然後你把吶喊這張圖也丟到 CNN 裡面得到 filter 的 output，注意，我們並不在於一個 filter output 的 value 到底是什麼，一個單獨的數字並不能代表任何的問題，我們真正在意的是，filter 和 filter 的 output 之間的 correlation，這個 correlation 代表了一張 image 的 style

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/deep-style2.png" width="60%;" /></center>
接下來你就再用一個CNN去找一張image，**這張image的content像左邊的圖片**，比如這張image的filter output的value像左邊的圖片；同時讓**這張image的style像右邊的圖片**，所謂的style像右邊的圖片是說，這張image output的filter之間的correlation像右邊這張圖片

最終你用 gradient descent 找到一張 image，同時可以 maximize 左邊的 content 和右邊的 style，它的樣子就像上圖左下角所示

#### More Application——Playing Go

##### What does CNN do in Playing Go

CNN 可以被運用到不同的應用上，不只是影像處理，比如出名的 alphaGo

想要讓 machine 來下圍棋，不見得要用 CNN，其實一般 typical 的 neural network 也可以幫我們做到這件事情

你只要 learn 一個 network，也就是找一個 function，它的 input 是棋盤當前局勢，output 是你下一步根據這個棋盤的盤勢而應該落子的位置，這樣其實就可以讓 machine 學會下圍棋了，所以用 fully connected 的 feedforward network 也可以做到讓 machine 下圍棋這件事情

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/play-go.png" width="60%;" /></center>
也就是說，你只要告訴它input是一個19\*19的vector，vector的每一個dimension對應到棋盤上的某一個位置，如果那一個位置有一個黑子的話，就是1，如果有一個白子的話，就是-1，反之呢，就是0，所以如果你把棋盤描述成一個19\*19的vector，丟到一個fully connected的feedforward network里，output也是19\*19個dimension ，每一個dimension對應到棋盤上的一個位置，那machine就可以學會下圍棋了

但實際上如果我們採用 CNN 的話，會得到更好的 performance，我們之前舉的例子都是把 CNN 用在圖像上面，也就是 input 是一個 matrix，而棋盤其實可以很自然地表示成一個 19\*19 的 matrix，那對 CNN 來說，就是直接把它當成一個 image 來看待，然後再 output 下一步要落子的位置，具體的 training process 是這樣的：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/go-process.png" width="60%;" /></center>
你就蒐集很多棋譜，比如說上圖這個是進藤光和社青春的棋譜，初手下在5之五，次手下在天元，然後再下在5之五，接下來你就告訴machine說，看到落子在5之五，CNN的output就是天元的地方是1，其他的output是0；看到5之五和天元都有子，那你的output就是5之五的地方是1，其他都是0

上面是 supervised 的部分，那其實呢 AlphaGo 還有 reinforcement learning 的部分，這個後面的章節會講到

##### Why CNN for Playing Go

自從 AlphaGo 用了 CNN 以後，大家都覺得好像 CNN 應該很厲害，所以有時候如果你沒有用 CNN 來處理問題，人家就會來問你；比如你去面試的時候，你的碩士論文裡面沒有用 CNN 來處理問題，口試的人可能不知道 CNN 是什麼 ，但是他就會問你說為什麼不用 CNN 呢，CNN 不是比較強嗎？這個時候如果你真的明白了為什麼要用 CNN，什麼時候才要用 CNN 這個問題，你就可以直接給他懟回去

那什麼時候我們可以用 CNN 呢？你要有 image 該有的那些特性，也就是上一篇文章開頭所說的，根據觀察到的三個 property，我們才設計出了 CNN 這樣的 network 架構：

- **Some patterns are much smaller than the whole image**
- **The same patterns appear in different regions**
- **Subsampling the pixels will not change the object**

CNN 能夠應用在 Alpha-Go 上，是因為圍棋有一些特性和圖像處理是很相似的

在 property 1，有一些 pattern 是比整張 image 要小得多，在圍棋上，可能也有同樣的現象，比如下圖中一個白子被 3 個黑子圍住，這個叫做吃，如果下一個黑子落在白子下面，就可以把白子提走；只有另一個白子接在下面，它才不會被提走

那現在你只需要看這個小小的範圍，就可以偵測這個白子是不是屬於被叫吃的狀態，你不需要看整個棋盤，才知道這件事情，所以這件事情跟 image 有著同樣的性質；在 AlphaGo 裡面，它第一個 layer 其實就是用 5\*5 的 filter，顯然做這個設計的人，覺得圍棋上最基本的 pattern 可能都是在 5\*5 的範圍內就可以被偵測出來

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/why-cnn-go.png" width="60%;" /></center>
在property 2，同樣的pattern可能會出現在不同的region，在圍棋上也可能有這個現象，像這個叫吃的pattern，它可以出現在棋盤的左上角，也可以出現在右下角，它們都是叫吃，都代表了同樣的意義，所以你可以用同一個detector，來處理這些在不同位置的同樣的pattern

所以對圍棋來說呢，它在第一個 observation 和第二個 observation 是有這個 image 的特性的，但是，讓我們沒有辦法想通的地方，就是第三點

##### Max Pooling for Alpha Go？——read alpha-go paper

我們可以對一個 image 做 subsampling，你拿掉奇數行、偶數列的 pixel，把 image 變成原來的 1/4 的大小也不會影響你看這張圖的樣子，基於這個觀察才有了 Max pooling 這個 layer；但是，對圍棋來說，它可以做這件事情嗎？比如說，你對一個棋盤丟掉奇數行和偶數列，那它還和原來是同一個函式嗎？顯然不是的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/go-property3.png" width="60%;" /></center>
如何解釋在棋盤上使用Max Pooling這件事情呢？有一些人覺得說，因為AlphaGo使用了CNN，它裡面有可能用了Max pooling這樣的構架，所以，或許這是它的一個弱點，你要是針對這個弱點攻擊它，也許就可以擊敗它

AlphaGo 的 paper 內容不多，只有 6 頁左右，它只說使用了 CNN，卻沒有在正文裡面仔細地描述它的 CNN 構架，但是在這篇 paper 長長附錄里，其實是有描述 neural network structure 的，如上圖所示

它是這樣說的，input 是一個 19\*19\*48 的 image，其中 19\*19 是棋盤的格局，對 Alpha 來說，每一個位置都用 48 個 value 來描述，這是因為加上了 domain knowledge，它不只是描述某位置有沒有白子或黑子，它還會觀察這個位置是不是處於叫吃的狀態等等

先用一個 hidden layer 對 image 做 zero padding，也就是把原來 19\*19 的 image 外圍補 0，讓它變成一張 23\*23 的 image，然後使用 k 個 5\*5 的 filter 對該 image 做 convolution，stride 設為 1，activation function 用的是 ReLU，得到的 output 是 21\*21 的 image；接下來使用 k 個 3\*3 的 filter，stride 設為 1，activation function 還是使用 ReLU，...

你會發現這個 AlphaGo 的 network structure 一直在用 convolution，其實**根本就沒有使用 Max Pooling**，原因並不是疏失了什麼之類的，而是根據圍棋的特性，我們本來就不需要在圍棋的 CNN 裡面，用 Max pooling 這樣的構架

舉這個例子是為了告訴大家：

==**neural network 架構的設計，是應用之道，存乎一心**==

#### More Application——Speech、Text

##### Speech

CNN 也可以用在很多其他的 task 裡面，比如語音處理上，我們可以把一段聲音表示成 spectrogram，spectrogram 的橫軸是時間，縱軸則是這一段時間里聲音的頻率

下圖中是一段「你好」的音頻，偏紅色代表這段時間里該頻率的 energy 是比較大的，也就對應著「你」和「好」這兩個字，也就是說 spectrogram 用顏色來描述某一個時刻不同頻率的能量

我們也可以讓機器把這個 spectrogram 就當作一張 image，然後用 CNN 來判斷說，input 的這張 image 對應著什麼樣的聲音信號，那通常用來判斷結果的單位，比如 phoneme，就是類似音標這樣的單位

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-speech.png" width="60%;" /></center>
這邊比較神奇的地方就是，當我們把一段spectrogram當作image丟到CNN裡面的時候，在語音上，我們通常只考慮在frequency(頻率)方向上移動的filter，我們的filter就像上圖這樣，是長方形的，它的寬就跟image的寬是一樣的，並且**filter只在Frequency即縱坐標的方向上移動，而不在時間的序列上移動**

這是因為在語音裡面，CNN 的 output 後面都還會再接別的東西，比如接 LSTM 之類，它們都已經有考慮 typical 的 information，所以你在 CNN 裡面再考慮一次時間的 information 其實沒有什麼特別的幫助，但是為什麼在頻率上 的 filter 有幫助呢？

我們用 CNN 的目的是為了用同一個 filter 把相同的 pattern 給 detect 出來，在聲音訊號上，雖然男生和女生說同樣的話看起來這個 spectrogram 是非常不一樣的，但實際上他們的不同只是表現在一個頻率的 shift 而已(整體在頻率上的位移)，男生說的「你好」跟女生說的「你好」，它們的 pattern 其實是一樣的，比如 pattern 是 spectrogram 變化的情形，男生女生的聲音的變化情況可能是一樣的，它們的差別可能只是所在的頻率範圍不同而已，所以 filter 在 frequency 的 direction 上移動是有效的

所以，這又是另外一個例子，當你把 CNN 用在一個 Application 的時候呢，你永遠要想一想這個 Application 的特性是什麼，根據這個特性你再去 design network 的 structure，才會真正在理解的基礎上去解決問題

##### Text

CNN 也可以用在文字處理上，假設你的 input 是一個 word sequence，你要做的事情是讓 machine 偵測這個 word sequence 代表的意思是 positive 的還是 negative 的

首先你把這個 word sequence 裡面的每一個 word 都用一個 vector 來表示，vector 代表的這個 word 本身的 semantic (語義)，那如果兩個 word 本身含義越接近的話，它們的 vector 在高維的空間上就越接近，這個東西就叫做 word embedding

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-text.png" width="60%;" /></center>
把一個sentence裡面所有word的vector排在一起，它就變成了一張image，你把CNN套用到這個image上，那filter的樣子就是上圖藍色的matrix，它的高和image的高是一樣的，然後把filter沿著句子里詞彙的順序來移動，每個filter移動完成之後都會得到一個由內積結果組成的vector，不同的filter就會得到不同的vector，接下來做Max pooling，然後把Max pooling的結果丟到fully connected layer裡面，你就會得到最後的output

與語音處理不同的是，**在文字處理上，filter 只在時間的序列(按照 word 的順序)上移動，而不在這個 embedding 的 dimension 上移動**；因為在 word embedding 裡面，不同 dimension 是 independent 的，它們是相互獨立的，不會出現有兩個相同的 pattern 的情況，所以在這個方向上面移動 filter，是沒有意義的

所以這又是另外一個例子，雖然大家覺得 CNN 很 powerful，你可以用在各個不同的地方，但是當你應用到一個新的 task 的時候，你要想一想這個新的 task 在設計 CNN 的構架的時候，到底該怎麼做

#### conclusion

本文的重點在於 CNN 的 theory base，也就是 What is CNN？What does CNN do？Why CNN？總結起來就是三個 property、兩個架構和一個理念，這也是使用 CNN 的條件基礎：

##### 三個 property

- **Some patterns are much smaller than the whole image** ——property 1
- **The same patterns appear in different regions** ——property 2
- **Subsampling the pixels will not change the object** ——property 3

##### 兩個架構

convolution 架構：針對 property 1 和 property 2

max pooling 架構：針對 property 3

##### 一個理念

針對不同的 application 要設計符合它特性的 network structure，而不是生硬套用，這就是 CNN 架構的設計理念：

==**應用之道，存乎一心**==
