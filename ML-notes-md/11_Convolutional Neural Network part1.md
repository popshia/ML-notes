# Convolutional Neural network(part 1)

> CNN 常常被用在影像處理上，它的 theory base 就是三個 property，和兩個架構
> convolution 架構：針對 property 1 和 property 2
> max pooling 架構：針對 property 3

#### Why CNN for Image？

##### CNN V.s. DNN

我們當然可以用一般的 neural network 來做影像處理，不一定要用 CNN，比如說，你想要做圖像的分類，那你就去 train 一個 neural network，它的 input 是一張圖片，你就用裡面的 pixel 來表示這張圖片，也就是一個很長很長的 vector，而 output 則是由圖像類別組成的 vector，假設你有 1000 個類別，那 output 就有 1000 個 dimension

但是，我們現在會遇到的問題是這樣子：實際上，在 train neural network 的時候，我們會有一種期待說，在這個 network structure 裡面的每一個 neuron，都應該代表了一個最基本的 classifier；事實上，在文獻上，根據訓練的結果，也有很多人得到這樣的結論，舉例來說，下圖中：

- 第一個 layer 的 neuron，它就是最簡單的 classifier，它做的事情就是 detect 有沒有綠色出現、有沒有黃色出現、有沒有斜的條紋出現等等
- 那第二個 layer，它做的事情是 detect 更複雜的東西，根據第一個 layer 的 output，它如果看到直線橫線，就是窗框的一部分；如果看到棕色的直條紋就是木紋；看到斜條紋加灰色的，這個有可能是很多東西，比如說，輪胎的一部分等等
- 再根據第二個 hidden layer 的 output，第三個 hidden layer 會做更複雜的事情，比如它可以知道說，當某一個 neuron 看到蜂巢，它就會被 activate；當某一個 neuron 看到車子，它就會被 activate；當某一個 neuron 看到人的上半身，它就會被 activate 等等

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/neuron-classifier.png" width="60%;"></center>
那現在的問題是這樣子：**當我們直接用一般的fully connected的feedforward network來做圖像處理的時候，往往會需要太多的參數**

舉例來說，假設這是一張 100\*100 的彩色圖片，它的分辨率才 100\*100，那這已經是很小張的 image 了，然後你需要把它拉成一個 vector，總共有 100\*100\*3 個 pixel(如果是彩色的圖的話，每個 pixel 其實需要 3 個 value，即 RGB 值來描述它的)，把這些加起來 input vectot 就已經有三萬維了；如果 input vector 是三萬維，又假設 hidden layer 有 1000 個 neuron，那僅僅是第一層 hidden layer 的參數就已經有 30000\*1000 個了，這樣就太多了

所以，**CNN 做的事情其實是，來簡化這個 neural network 的架構，我們根據自己的知識和對圖像處理的理解，一開始就把某些實際上用不到的參數給過濾掉**，我們一開始就想一些辦法，不要用 fully connected network，而是用比較少的參數，來做圖像處理這件事情，所以 CNN 其實是比一般的 DNN 還要更簡單的

雖然 CNN 看起來，它的運作比較複雜，但事實上，它的模型比 DNN 還要更簡單，我們就是用 prior knowledge，去把原來 fully connected 的 layer 裡面的一些參數拿掉，就變成 CNN

##### Three Property for CNN theory base

為什麼我們有可能把一些參數拿掉？為什麼我們有可能只用比較少的參數就可以來做圖像處理這件事情？下面列出三個對影像處理的觀察：(**這也是 CNN 架構提出的基礎所在！！！**)

###### Some patterns are much smaller than the whole image

在影像處理裡面，如果在 network 的第一層 hidden layer 里，那些 neuron 要做的事情是偵測有沒有一種東西、一種 pattern(圖案樣式)出現，那大部分的 pattern 其實是比整張 image 要小的，所以對一個 neuron 來說，想要偵測有沒有某一個 pattern 出現，它其實並不需要看整張 image，只需要看這張 image 的一小部分，就可以決定這件事情了

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pattern.png" width="60%;"></center>
舉例來說，假設現在我們有一張鳥的圖片，那第一層hidden layer的某一個neuron的工作是，檢測有沒有鳥嘴的存在(你可能還有一些neuron偵測有沒有鳥嘴的存在、有一些neuron偵測有沒有爪子的存在、有一些neuron偵測有沒有翅膀的存在、有沒有尾巴的存在，之後合起來，就可以偵測，圖片中有沒有一隻鳥)，那它其實並不需要看整張圖，因為，其實我們只要給neuron看這個小的紅色槓槓裡面的區域，它其實就可以知道說，這是不是一個鳥嘴，對人來說也是一樣，只要看這個小的區域你就會知道說這是鳥嘴，所以，**每一個neuron其實只要連接到一個小塊的區域就好，它不需要連接到整張完整的圖，因此也對應著更少的參數**

###### The same patterns appear in different regions

同樣的 pattern，可能會出現在 image 的不同部分，但是它們有同樣的形狀、代表的是同樣的含義，因此它們也可以用同樣的 neuron、同樣的參數，被同一個 detector 檢測出來

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/pattern-region.png" width="60%;"></center>
舉例來說，上圖中分別有一個處於左上角的鳥嘴和一個處於中央的鳥嘴，但你並不需要訓練兩個不同的detector去專門偵測左上角有沒有鳥嘴和中央有沒有鳥嘴這兩件事情，這樣做太冗余了，我們要cost down(降低成本)，我們並不需要有兩個neuron、兩組不同的參數來做duplicate(重復一樣)的事情，所以**我們可以要求這些功能幾乎一致的neuron共用一組參數，它們share同一組參數就可以幫助減少總參數的量**

###### Subsampling the pixels will not change the object

我們可以對一張 image 做 subsampling(二次抽樣)，假如你把它奇數行、偶數列的 pixel 拿掉，image 就可以變成原來的十分之一大小，而且並不會影響人對這張 image 的理解，對你來說，下面兩張大小不一的 image 看起來不會有什麼太大的區別，你都可以識別裡面有什麼物件，因此 subsampling 對圖像辨識來說，可能是沒有太大的影響的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/subsamp.png" width="60%;"></center>
所以，**我們可以利用subsampling這個概念把image變小，從而減少需要用到的參數量**

#### The whole CNN structure

整個 CNN 的架構是這樣的：

首先，input 一張 image 以後，它會先通過 Convolution 的 layer，接下來做 Max Pooling 這件事，然後再去做 Convolution，再做 Maxi Pooling...，這個 process 可以反復進行多次(重復次數需要事先決定)，這就是 network 的架構，就好像 network 有幾層一樣，你要做幾次 convolution，做幾次 Max Pooling，在定這個 network 的架構時就要事先決定好

當你做完先前決定的 convolution 和 max pooling 的次數後，你要做的事情是 Flatten，做完 flatten 以後，你就把 Flatten output 丟到一般的 Fully connected network 裡面去，最終得到影像辨識的結果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/whole-cnn.png" width="60%;"></center>
我們基於之前提到的三個對影像處理的觀察，設計了CNN這樣的架構，第一個是要偵測一個pattern，你不需要看整張image，只要看image的一個小部分；第二個是同樣的pattern會出現在一張圖片的不同區域；第三個是我們可以對整張image做subsampling

那**前面這兩個 property，是用 convolution 的 layer 來處理的；最後這個 property，是用 max pooling 來處理的**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/property.png" width="60%;"></center>
#### Convolution

假設現在我們 network 的 input 是一張 6\*6 的 image，圖像是黑白的，因此每個 pixel 只需要用一個 value 來表示，而在 convolution layer 裡面，有一堆 Filter，這邊的每一個 Filter，其實就等同於是 Fully connected layer 里的一個 neuron

##### Property 1

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/filter.png" width="60%;"></center>
每一個Filter其實就是一個matrix，這個matrix裡面每一個element的值，就跟那些neuron的weight和bias一樣，是network的parameter，它們具體的值都是通過Training data學出來的，而不是人去設計的

所以，每個 Filter 裡面的值是什麼，要做什麼事情，都是自動學習出來的，上圖中每一個 filter 是 3\*3 的 size，意味著它就是在偵測一個 3\*3 的 pattern，**當它偵測的時候，並不會去看整張 image，它只看一個 3\*3 範圍內的 pixel，就可以判斷某一個 pattern 有沒有出現，這就考慮了 property 1**

##### Property 2

這個 filter 是從 image 的左上角開始，做一個 slide window，每次向右挪動一定的距離，這個距離就叫做 stride，由你自己設定，每次 filter 停下的時候就跟 image 中對應的 3\*3 的 matrix 做一個內積(相同位置的值相乘並累計求和)，這裡假設 stride=1，那麼我們的 filter 每次移動一格，當它碰到 image 最右邊的時候，就從下一行的最左邊開始重復進行上述操作，經過一整個 convolution 的 process，最終得到下圖所示的紅色的 4\*4 matrix

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/filter1.png" width="60%;"></center>
觀察上圖中的Filter1，它斜對角的地方是1,1,1，所以它的工作就是detect有沒有連續的從左上角到右下角的1,1,1出現在這個image裡面，檢測到的結果已在上圖中用藍線標識出來，此時filter得到的卷積結果的左上和左下得到了最大的值，這就代表說，該filter所要偵測的pattern出現在image的左上角和左下角

**同一個 pattern 出現在 image 左上角的位置和左下角的位置，並不需要用到不同的 filter，我們用 filter1 就可以偵測出來，這就考慮了 property 2**

##### Feature Map

在一個 convolution 的 layer 裡面，它會有一打 filter，不一樣的 filter 會有不一樣的參數，但是這些 filter 做卷積的過程都是一模一樣的，你把 filter2 跟 image 做完 convolution 以後，你就會得到另外一個藍色的 4\*4 matrix，那這個藍色的 4\*4 matrix 跟之前紅色的 4\*4matrix 合起來，就叫做**Feature Map(特徵映射)**，有多少個 filter，對應就有多少個映射後的 image

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/filter2.png" width="60%;"></center>
CNN對**不同scale的相同pattern的處理**上存在一定的困難，由於現在每一個filter size都是一樣的，這意味著，如果你今天有同一個pattern，它有不同的size，有大的鳥嘴，也有小的鳥嘴，CNN並不能夠自動處理這個問題；DeepMind曾經發過一篇paper，上面提到了當你input一張image的時候，它在CNN前面，再接另外一個network，這個network做的事情是，它會output一些scalar，告訴你說，它要把這個image的裡面的哪些位置做旋轉、縮放，然後，再丟到CNN裡面，這樣你其實會得到比較好的performance

#### Colorful image

剛才舉的例子是黑白的 image，所以你 input 的是一個 matrix，如果今天是彩色的 image 會怎麼樣呢？我們知道彩色的 image 就是由 RGB 組成的，所以一個彩色的 image，它就是好幾個 matrix 疊在一起，是一個立方體，如果我今天要處理彩色的 image，要怎麼做呢？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/rgb.png" width="60%;"></center>
這個時候你的filter就不再是一個matrix了，它也會是一個立方體，如果你今天是RGB這三個顏色來表示一個pixel的話，那你的input就是3\*6\*6，你的filter就是3\*3\*3，你的filter的高就是3，你在做convolution的時候，就是把這個filter的9個值跟這個image裡面的9個值做內積，可以想象成filter的每一層都分別跟image的三層做內積，得到的也是一個三層的output，每一個filter同時就考慮了不同顏色所代表的channel

#### Convolution V.s. Fully connected

##### filter 是特殊的」neuron「

接下來要講的是，convolution 跟 fully connected 有什麼關係，你可能覺得說，它是一個很特別的 operation，感覺跟 neural network 沒半毛錢關係，其實，它就是一個 neural network

convolution 這件事情，其實就是 fully connected 的 layer 把一些 weight 拿掉而已，下圖中綠色方框標識出的 feature map 的 output，其實就是 hidden layer 的 neuron 的 output

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/convolution-fully.png" width="60%;"></center>
接下來我們來解釋這件事情：

如下圖所示，我們在做 convolution 的時候，把 filter 放在 image 的左上角，然後再去做 inner product，得到一個值 3；這件事情等同於，我們現在把這個 image 的 6\*6 的 matrix 拉直變成右邊這個用於 input 的 vector，然後，你有一個紅色的 neuron，這些 input 經過這個 neuron 之後，得到的 output 是 3

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/filter-neuron1.png" width="60%;"></center>
##### 每個「neuron」只檢測image的部分區域

那這個 neuron 的 output 怎麼來的呢？這個 neuron 實際上就是由 filter 轉化而來的，我們把 filter 放在 image 的左上角，此時 filter 考慮的就是和它重合的 9 個 pixel，假設你把這一個 6\*6 的 image 的 36 個 pixel 拉成直的 vector 作為 input，那這 9 個 pixel 分別就對應著右側編號 1，2，3 的 pixel，編號 7，8，9 的 pixel 跟編號 13，14，15 的 pixel

如果我們說這個 filter 和 image matrix 做 inner product 以後得到的 output 3，就是 input vector 經過某個 neuron 得到的 output 3 的話，這就代表說存在這樣一個 neuron，這個 neuron 帶 weight 的連線，就只連接到編號為 1，2，3，7，8，9，13，14，15 的這 9 個 pixel 而已，而這個 neuron 和這 9 個 pixel 連線上所標注的的 weight 就是 filter matrix 裡面的這 9 個數值

作為對比，Fully connected 的 neuron 是必須連接到所有 36 個 input 上的，但是，我們現在只用連接 9 個 input，因為我們知道要 detect 一個 pattern，不需要看整張 image，看 9 個 input pixel 就夠了，所以當我們這麼做的時候，就用了比較少的參數

##### 「neuron」之間共享參數

當我們把 filter 做 stride = 1 的移動的時候，會發生什麼事呢？此時我們通過 filter 和 image matrix 的內積得到另外一個 output 值-1，我們假設這個-1 是另外一個 neuron 的 output，那這個 neuron 會連接到哪些 input 呢？下圖中這個框起來的地方正好就對應到 pixel 2，3，4，pixel 8，9，10 跟 pixel 14，15，16

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/share-weight.png" width="60%;"></center>
你會發現output為3和-1的這兩個neuron，它們分別去檢測在image的兩個不同位置上是否存在某個pattern，因此在Fully connected layer里它們做的是兩件不同的事情，每一個neuron應該有自己獨立的weight

但是，當我們做這個 convolution 的時候，首先我們把每一個 neuron 前面連接的 weight 減少了，然後我們強迫某些 neuron(比如上圖中 output 為 3 和-1 的兩個 neuron)，它們一定要共享一組 weight，雖然這兩個 neuron 連接到的 pixel 對象各不相同，但它們用的 weight 都必須是一樣的，等於 filter 裡面的元素值，這件事情就叫做 weight share，當我們做這件事情的時候，用的參數，又會比原來更少

##### 總結

因此我們可以這樣想，有這樣一些特殊的 neuron，它們只連接著 9 條帶 weight 的線(9=3\*3 對應著 filter 的元素個數，這些 weight 也就是 filter 內部的元素值，上圖中圓圈的顏色與連線的顏色一一對應)

當 filter 在 image matrix 上移動做 convolution 的時候，每次移動做的事情實際上是去檢測這個地方有沒有某一種 pattern，對於 Fully connected layer 來說，它是對整張 image 做 detection 的，因此每次去檢測 image 上不同地方有沒有 pattern 其實是不同的事情，所以這些 neuron 都必須連接到整張 image 的所有 pixel 上，並且不同 neuron 的連線上的 weight 都是相互獨立的

==**那對於 convolution layer 來說，首先它是對 image 的一部分做 detection 的，因此它的 neuron 只需要連接到 image 的部分 pixel 上，對應連線所需要的 weight 參數就會減少；其次由於是用同一個 filter 去檢測不同位置的 pattern，所以這對 convolution layer 來說，其實是同一件事情，因此不同的 neuron，雖然連接到的 pixel 對象各不相同，但是在「做同一件事情」的前提下，也就是用同一個 filter 的前提下，這些 neuron 所使用的 weight 參數都是相同的，通過這樣一張 weight share 的方式，再次減少 network 所需要用到的 weight 參數**==

CNN 的本質，就是減少參數的過程

##### 補充

看到這裡你可能會問，這樣的 network 該怎麼搭建，又該怎麼去 train 呢？

首先，第一件事情就是這都是用 toolkit 做的，所以你大概不會自己去寫；如果你要自己寫的話，它其實就是跟原來的 Backpropagation 用一模一樣的做法，只是有一些 weight 就永遠是 0，你就不去 train 它，它就永遠是 0

然後，怎麼讓某些 neuron 的 weight 值永遠都是一樣呢？你就用一般的 Backpropagation 的方法，對每個 weight 都去算出 gradient，再把本來要 tight 在一起、要 share weight 的那些 weight 的 gradient 平均，然後，讓他們 update 同樣值就 ok 了

#### Max Pooling

##### Operation of max pooling

相較於 convolution，max pooling 是比較簡單的，它就是做 subsampling，根據 filter 1，我們得到一個 4\*4 的 matrix，根據 filter 2，你得到另外一個 4\*4 的 matrix，接下來，我們要做什麼事呢？

我們把 output 四個分為一組，每一組裡面通過選取平均值或最大值的方式，把原來 4 個 value 合成一個 value，這件事情相當於在 image 每相鄰的四塊區域內都挑出一塊來檢測，這種 subsampling 的方式就可以讓你的 image 縮小！

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/max-pooling.png" width="60%;"></center>
講到這裡你可能會有一個問題，如果取Maximum放到network裡面，不就沒法微分了嗎？max這個東西，感覺是沒有辦法對它微分的啊，其實是可以的，後面的章節會講到Maxout network，會告訴你怎麼用微分的方式來處理它

##### Convolution + Max Pooling

所以，結論是這樣的：

做完一次 convolution 加一次 max pooling，我們就把原來 6\*6 的 image，變成了一個 2\*2 的 image；至於這個 2\*2 的 image，它每一個 pixel 的深度，也就是每一個 pixel 用幾個 value 來表示，就取決於你有幾個 filter，如果你有 50 個 filter，就是 50 維，像下圖中是兩個 filter，對應的深度就是兩維

所以，這是一個新的比較小的 image，它表示的是不同區域上提取到的特徵，實際上不同的 filter 檢測的是該 image 同一區域上的不同特徵屬性，所以每一層 channel(通道)代表的是一種屬性，一塊區域有幾種不同的屬性，就有幾層不同的 channel，對應的就會有幾個不同的 filter 對其進行 convolution 操作

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/max-pool.png" width="60%;"></center>
這件事情可以repeat很多次，你可以把得到的這個比較小的image，再次進行convolution和max pooling的操作，得到一個更小的image，依次類推

有這樣一個問題：假設我第一個 convolution 有 25 個 filter，通過這些 filter 得到 25 個 feature map，然後 repeat 的時候第二個 convolution 也有 25 個 filter，那這樣做完，我是不是會得到 25^2 個 feature map？

其實不是這樣的，你這邊做完一次 convolution，得到 25 個 feature map 之後再做一次 convolution，還是會得到 25 個 feature map，因為 convolution 在考慮 input 的時候，是會考慮深度的，它並不是每一個 channel 分開考慮，而是一次考慮所有的 channel，所以，你 convolution 這邊有多少個 filter，再次 output 的時候就會有多少個 channel

因此你這邊有 25 個 channel，經過含有 25 個 filter 的 convolution 之後 output 還會是 25 個 channel，只是這邊的每一個 channel，它都是一個 cubic(立方體)，它的高有 25 個 value 那麼高

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/channel.png" width="60%;"></center>
#### Flatten

做完 convolution 和 max pooling 之後，就是 FLatten 和 Fully connected Feedforward network 的部分

Flatten 的意思是，把左邊的 feature map 拉直，然後把它丟進一個 Fully connected Feedforward network，然後就結束了，也就是說，我們之前通過 CNN 提取出了 image 的 feature，它相較於原先一整個 image 的 vetor，少了很大一部分內容，因此需要的參數也大幅度地減少了，但最終，也還是要丟到一個 Fully connected 的 network 中去做最後的分類工作

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/fatten.png" width="50%;"></center>
#### CNN in Keras

##### 內容簡介

接下來就講一下，如何用 Keras 來 implement CNN，實際上在 compile、training 和 fitting 的部分，內容跟 DNN 是一模一樣的，對 CNN 來說，唯一需要改變的是 network structure，以及 input 的 format

本來在 DNN 里，input 是一個由 image 拉直展開而成的 vector，但現在如果是 CNN 的話，它是會考慮 input image 的幾何空間的，所以不能直接 input 一個 vector，而是要 input 一個 tensor 給它(tensor 就是高維的 vector)，這裡你要給它一個三維的 vector，一個 image 的長寬各是一維，如果它是彩色的話，RGB 就是第三維，所以你要 assign 一個三維的 matrix，這個高維的 matrix 就叫做 tensor

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-keras1.png" width="60%;"></center>
那怎麼implement一個convolution的layer呢？

```python
model2.add( Convolution2D(25,3,3, input_shape=(28,28,1)) )
```

還是用`model.add`增加 CNN 的 layer，將原先的 Dense 改成 Convolution2D，參數 25 代表你有 25 個 filter，參數 3,3 代表你的 filter 都是 3\*3 的 matrix，此外你還需要告訴 model，你 input 的 image 的 shape 是什麼樣子的，假設我現在要做手寫數字識別，input 就是 28\*28 的 image，又因為它的每一個 pixel 都只有單一顏色，因此`input_shape`的值就是(28,28,1)，如果是 RGB 的話，1 就要改成 3

然後增加一層 Max Pooling 的 layer

```python
model2.add( MaxPooling2D(2,2) )
```

這裡參數(2,2)指的是，我們把通過 convolution 得到的 feature map，按照 2\*2 的方式分割成一個個區域，每次選取最大的那個值，並將這些值組成一個新的比較小的 image，作為 subsampling 的結果

##### 過程描述

- 假設我們 input 是一個 1\*28\*28 的 image

- 通過 25 個 filter 的 convolution layer 以後你得到的 output，會有 25 個 channel，又因為 filter 的 size 是 3\*3，因此如果不考慮 image 邊緣處的處理的話，得到的 channel 會是 26\*26 的，因此通過第一個 convolution 得到 25\*26\*26 的 cubic image(這裡把這張 image 想象成長寬為 26，高為 25 的 cubic 立方體)

- 接下來就是做 Max pooling，把 2\*2 的 pixel 分為一組，然後從裡面選一個最大的組成新的 image，大小為 25\*13\*13(cubic 長寬各被砍掉一半)

- 再做一次 convolution，假設這次選擇 50 個 filter，每個 filter size 是 3\*3 的話，output 的 channel 就變成有 50 個，那 13\*13 的 image，通過 3\*3 的 filter，就會變成 11\*11，因此通過第二個 convolution 得到 50\*11\*11 的 image(得到一個新的長寬為 11，高為 50 的 cubic)

- 再做一次 Max Pooling，變成 50\*50\*5

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-keras3.png" width="60%;"></center>
在第一個convolution裡面，每一個filter都有9個參數，它就是一個3\*3的matrix；但是在第二個convolution layer裡面，雖然每一個filter都是3\*3，但它其實不是3\*3個參數，因為它的input是一個25\*13\*13的cubic，這個cubic的channel有25個，所以要用同樣高度的cubic filter對它進行卷積，於是我們的filter實際上是一個25\*3\*3的cubic，所以這邊每個filter共有225個參數

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cnn-keras2.png" width="60%;"></center>
通過兩次convolution和max pooling的組合，最終的image變成了50\*5\*5的size，然後使用Flatten將這個image拉直，變成一個1250維的vector，再把它丟到一個Fully Connected Feedforward network裡面，network structure就搭建完成了

##### 一個重要的問題

看到這裡，你可能會有一個疑惑，第二次 convolution 的 input 是 25\*13\*13 的 cubic，用 50 個 3\*3 的 filter 卷積後，得到的輸出時應該是 50 個 cubic，且每個 cubic 的尺寸為 25\*11\*11，那麼 max pooling 把長寬各砍掉一半後就是 50 層 25\*5\*5 的 cubic，那 flatten 後不應該就是 50\*25\*5\*5 嗎？

其實不是這樣的，在第二次做 convolution 的時候，我們是用 25\*3\*3 的 cubic filter 對 25\*13\*13 的 cubic input 進行卷積操作的，filter 的每一層和 input cubic 中對應的每一層(也就是每一個 channel)，它們進行內積後，還要把 cubic 的 25 個 channel 的內積值進行求和，作為這個「neuron」的 output，它是一個 scalar，這個 cubic filter 對整個 cubic input 做完一遍卷積操作後，得到的是一層 scalar，然後有 50 個 cubic filter，對應著 50 層 scalar，因此最終得到的 output 是一個 50\*11\*11 的 cubic！

這裡的關鍵是 filter 和 image 都是 cubic，每個 cubic filter 有 25 層高，它和同樣有 25 層高的 cubic image 做卷積，並不是單單把每個 cubic 對應的 channel 進行內積，還會把這些內積求和！！！最終變為 1 層，因此==**兩個矩陣或者 tensor 做了卷積後，不管之前的維數如何，都會變為一個 scalor！**==，故如果有 50 個 Filter，無論 input 是什麼樣子的，最終的 output 還會是 50 層

#### Appendix：CNN in Keras2.0

這裡還是舉**手寫數字識別**的例子，將單純使用 DNN 和加上 CNN 的情況作為對比

##### code

```python
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist

# categorical_crossentropy


def load_mnist_data(number):
    # the data, shuffled and  split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist_data(10000)

    # do DNN
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=100, epochs=20)

    result_train = model.evaluate(x_train, y_train)
    print('\nTrain Acc:\n', result_train[1])

    result_test = model.evaluate(x_test, y_test)
    print('\nTest Acc:\n', result_test[1])

    # do CNN
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    model2 = Sequential()
    model2.add(Conv2D(25, (3, 3), input_shape=(
        1, 28, 28), data_format='channels_first'))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Conv2D(50, (3, 3)))
    model2.add(MaxPooling2D((2, 2)))
    model2.add(Flatten())
    model2.add(Dense(units=100, activation='relu'))
    model2.add(Dense(units=10, activation='softmax'))
    model2.summary()

    model2.compile(loss='categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    model2.fit(x_train, y_train, batch_size=100, epochs=20)

    result_train = model2.evaluate(x_train, y_train)
    print('\nTrain CNN Acc:\n', result_train[1])
    result_test = model2.evaluate(x_test, y_test)
    print('\nTest CNN Acc:\n', result_test[1])

```

##### result

###### DNN

```python
Using TensorFlow backend.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 500)               392500
_________________________________________________________________
dense_2 (Dense)              (None, 500)               250500
_________________________________________________________________
dense_3 (Dense)              (None, 500)               250500
_________________________________________________________________
dense_4 (Dense)              (None, 10)                5010
=================================================================
Total params: 898,510
Trainable params: 898,510
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
10000/10000 [==============================] - 2s 207us/step - loss: 0.4727 - acc: 0.8643
Epoch 2/20
10000/10000 [==============================] - 1s 149us/step - loss: 0.1613 - acc: 0.9521
Epoch 3/20
10000/10000 [==============================] - 2s 159us/step - loss: 0.0916 - acc: 0.9726
Epoch 4/20
10000/10000 [==============================] - 2s 173us/step - loss: 0.0680 - acc: 0.9769
Epoch 5/20
10000/10000 [==============================] - 2s 166us/step - loss: 0.0437 - acc: 0.9850
Epoch 6/20
10000/10000 [==============================] - 2s 166us/step - loss: 0.0274 - acc: 0.9921
Epoch 7/20
10000/10000 [==============================] - 2s 168us/step - loss: 0.0265 - acc: 0.9892
Epoch 8/20
10000/10000 [==============================] - 2s 161us/step - loss: 0.0240 - acc: 0.9916
Epoch 9/20
10000/10000 [==============================] - 2s 169us/step - loss: 0.0149 - acc: 0.9950
Epoch 10/20
10000/10000 [==============================] - 2s 155us/step - loss: 0.0258 - acc: 0.9933
Epoch 11/20
10000/10000 [==============================] - 2s 168us/step - loss: 0.0206 - acc: 0.9934
Epoch 12/20
10000/10000 [==============================] - 2s 161us/step - loss: 0.0132 - acc: 0.9955
Epoch 13/20
10000/10000 [==============================] - 2s 168us/step - loss: 0.0113 - acc: 0.9964
Epoch 14/20
10000/10000 [==============================] - 2s 169us/step - loss: 0.0027 - acc: 0.9991
Epoch 15/20
10000/10000 [==============================] - 2s 157us/step - loss: 6.6533e-04 - acc: 0.9999
Epoch 16/20
10000/10000 [==============================] - 1s 150us/step - loss: 1.1253e-04 - acc: 1.0000
Epoch 17/20
10000/10000 [==============================] - 2s 152us/step - loss: 8.3190e-05 - acc: 1.0000
Epoch 18/20
10000/10000 [==============================] - 2s 174us/step - loss: 6.7850e-05 - acc: 1.0000
Epoch 19/20
10000/10000 [==============================] - 2s 173us/step - loss: 5.6810e-05 - acc: 1.0000
Epoch 20/20
10000/10000 [==============================] - 2s 172us/step - loss: 4.8757e-05 - acc: 1.0000

10000/10000 [==============================] - 1s 97us/step
Train Acc: 1.0
10000/10000 [==============================] - 1s 77us/step
Test Acc: 0.9661
```

###### CNN

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 25, 26, 26)        250
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 12, 13, 26)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 10, 11, 50)        11750
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 50)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1250)              0
_________________________________________________________________
dense_5 (Dense)              (None, 100)               125100
_________________________________________________________________
dense_6 (Dense)              (None, 10)                1010
=================================================================
Total params: 138,110
Trainable params: 138,110
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
10000/10000 [==============================] - 8s 785us/step - loss: 0.6778 - acc: 0.8113
Epoch 2/20
10000/10000 [==============================] - 7s 734us/step - loss: 0.2302 - acc: 0.9349
Epoch 3/20
10000/10000 [==============================] - 8s 765us/step - loss: 0.1562 - acc: 0.9532
Epoch 4/20
10000/10000 [==============================] - 8s 760us/step - loss: 0.1094 - acc: 0.9680
Epoch 5/20
10000/10000 [==============================] - 8s 843us/step - loss: 0.0809 - acc: 0.9763
Epoch 6/20
10000/10000 [==============================] - 7s 748us/step - loss: 0.0664 - acc: 0.9810
Epoch 7/20
10000/10000 [==============================] - 8s 764us/step - loss: 0.0529 - acc: 0.9832
Epoch 8/20
10000/10000 [==============================] - 7s 747us/step - loss: 0.0370 - acc: 0.9904
Epoch 9/20
10000/10000 [==============================] - 7s 687us/step - loss: 0.0302 - acc: 0.9919
Epoch 10/20
10000/10000 [==============================] - 7s 690us/step - loss: 0.0224 - acc: 0.9940
Epoch 11/20
10000/10000 [==============================] - 7s 698us/step - loss: 0.0177 - acc: 0.9959
Epoch 12/20
10000/10000 [==============================] - 7s 690us/step - loss: 0.0154 - acc: 0.9965
Epoch 13/20
10000/10000 [==============================] - 7s 692us/step - loss: 0.0126 - acc: 0.9962
Epoch 14/20
10000/10000 [==============================] - 7s 689us/step - loss: 0.0130 - acc: 0.9966
Epoch 15/20
10000/10000 [==============================] - 7s 691us/step - loss: 0.0092 - acc: 0.9977
Epoch 16/20
10000/10000 [==============================] - 7s 691us/step - loss: 0.0067 - acc: 0.9986
Epoch 17/20
10000/10000 [==============================] - 7s 687us/step - loss: 0.0069 - acc: 0.9985
Epoch 18/20
10000/10000 [==============================] - 7s 691us/step - loss: 0.0040 - acc: 0.9995
Epoch 19/20
10000/10000 [==============================] - 7s 745us/step - loss: 0.0020 - acc: 1.0000
Epoch 20/20
10000/10000 [==============================] - 8s 782us/step - loss: 0.0014 - acc: 1.0000

10000/10000 [==============================] - 7s 657us/step
Train CNN Acc: 1.0
10000/10000 [==============================] - 5s 526us/step
Test CNN Acc: 0.98
```
