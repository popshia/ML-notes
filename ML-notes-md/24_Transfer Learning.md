# Transfer Learning

> 遷移學習，主要介紹共享 layer 的方法以及屬性降維對比的方法

#### Introduction

遷移學習，transfer learning，旨在利用一些不直接相關的數據對完成目標任務做出貢獻

##### not directly related

以貓狗識別為例，解釋「不直接相關」的含義：

- input domain 是類似的，但 task 是無關的

  比如輸入都是動物的圖像，但這些 data 是屬於另一組有關大象和老虎識別的 task

- input domain 是不同的，但 task 是一樣的

  比如 task 同樣是做貓狗識別，但輸入的是卡通類型的圖像

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/no-related.png" width="60%"/></center>

##### compare with real life

事實上，我們在日常生活中經常會使用遷移學習，比如我們會把漫畫家的生活自動遷移類比到研究生的生活

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tf-real.png" width="60%"/></center>

##### overview

遷移學習是很多方法的集合，這裡介紹一些概念：

- Target Data：和 task 直接相關的 data
- Source Data：和 task 沒有直接關係的 data

按照 labeled data 和 unlabeled data 又可以劃分為四種：

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tl-overview.png" width="60%"/></center>

### Case 1

這裡 target data 和 source data 都是帶有標籤的：

- target data：$(x^t,y^t)$，作為有效數據，通常量是很少的

  如果 target data 量非常少，則被稱為 one-shot learning

- source data：$(x^s, y^s)$，作為不直接相關數據，通常量是很多的

#### Model Fine-tuning

##### Introduction

模型微調的基本思想：用 source data 去訓練一個 model，再用 target data 對 model 進行微調(fine tune)

所謂「微調」，類似於 pre-training，就是把用 source data 訓練出的 model 參數當做是參數的初始值，再用 target data 繼續訓練下去即可，但當直接相關的數據量非常少時，這種方法很容易會出問題

所以訓練的時候要小心，有許多技巧值得注意

##### Conservation Training

如果現在有大量的 source data，比如在語音識別中有大量不同人的聲音數據，可以拿它去訓練一個語音識別的神經網絡，而現在你擁有的 target data，即特定某個人的語音數據，可能只有十幾條左右，如果直接拿這些數據去再訓練，肯定得不到好的結果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tf-ct.png" width="60%"/></center>

此時我們就需要在訓練的時候加一些限制，讓用 target data 訓練前後的 model 不要相差太多：

- 我們可以讓新舊兩個 model 在看到同一筆 data 的時候，output 越接近越好
- 或者讓新舊兩個 model 的 L2 norm 越小越好，參數盡可能接近
- 總之讓兩個 model 不要相差太多，防止由於 target data 的訓練導致過擬合

注：這裡的限制就類似於 regularization

##### Layer Transfer

現在我們已經有一個用 source data 訓練好的 model，此時把該 model 的某幾個 layer 拿出來複製到同樣大小的新 model 里，接下來**只**用 target data 去訓練余下的沒有被複製到的 layer

這樣做的好處是 target data 只需要考慮 model 中非常少的參數，這樣就可以避免過擬合

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tl-lt.png" width="60%"/></center>

這個對部分 layer 進行遷移的過程，就體現了遷移學習的思想，接下來要面對的問題是，哪些 layer 需要被複製遷移，哪些不需要呢？

值得注意的是，在不同的 task 上，需要被複製遷移的 layer 往往是不一樣的：

- 在語音識別中，往往遷移的是最後幾層 layer，再重新訓練與輸入端相鄰的那幾層

  由於口腔結構不同，同樣的發音方式得到的發音是不一樣的，NN 的前幾層會從聲音信號里提取出發音方式，再用後幾層判斷對應的詞彙，從這個角度看，NN 的後幾層是跟特定的人沒有關係的，因此可做遷移

- 在圖像處理中，往往遷移的是前面幾層 layer，再重新訓練後面的 layer

  CNN 在前幾層通常是做最簡單的識別，比如識別是否有直線斜線、是否有簡單的幾何圖形等，這些 layer 的功能是可以被遷移到其它 task 上通用的

- 主要還是具體問題具體分析

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/tl-lt2.png" width="60%"/></center>

#### Multitask Learning

##### Introduction

fine-tune 僅考慮在 target data 上的表現，而多任務學習，則是同時考慮 model 在 source data 和 target data 上的表現

如果兩個 task 的輸入特徵類似，則可以用同一個神經網絡的前幾層 layer 做相同的工作，到後幾層再分方向到不同的 task 上，這樣做的好處是前幾層得到的 data 比較多，可以被訓練得更充分

有時候 task A 和 task B 的輸入輸出都不相同，但中間可能會做一些類似的處理，則可以讓兩個神經網絡共享中間的幾層 layer，也可以達到類似的效果

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/multi-task.png" width="60%"/></center>

注意，以上方法要求不同的 task 之間要有一定的共性，這樣才有共用一部分 layer 的可能性

##### Multilingual Speech Recognition

多任務學習在語音識別上比較有用，可以同時對法語、德語、西班牙語、意大利語訓練一個 model，它們在前幾層 layer 上共享參數，而在後幾層 layer 上擁有自己的參數

在機器翻譯上也可以使用同樣的思想，比如訓練一個同時可以中翻英和中翻日的 model

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/multi-task-speech.png" width="60%"/></center>

注意到，屬於同一個語系的語言翻譯，比如歐洲國家的語言，幾乎都是可以做遷移學習的；而語音方面則可遷移的範圍更廣

下圖展示了只用普通話的語音數據和加了歐洲話的語音數據之後得到的錯誤率對比，其中橫軸為使用的普通話數據量，縱軸為錯誤率，可以看出使用了遷移學習後，只需要原先一半的普通話語音數據就可以得到幾乎一樣的準確率

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/multi-task-speech2.png" width="60%"/></center>

#### Progressive Neural Network

如果兩個 task 完全不相關，硬是把它們拿來一起訓練反而會起到負面效果

而在 Progressive Neural Network 中，每個 task 對應 model 的 hidden layer 的輸出都會被接到後續 model 的 hidden layer 的輸入上，這樣做的好處是：

- task 2 的 data 並不會影響到 task 1 的 model，因此 task 1 一定不會比原來更差
- task 2 雖然可以借用 task 1 的參數，但可以將之直接設為 0，最糟的情況下就等於沒有這些參數，也不會對本身的表現產生影響

- task 3 也做一樣的事情，同時從 task 1 和 task 2 的 hidden layer 中得到信息

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/multi-task-pro.png" width="60%"/></center>

### Case 2

這裡 target data 不帶標籤，而 source data 帶標籤：

- target data：$(x^t)$

- source data：$(x^s, y^s)$

#### Domain-adversarial Training

如果 source data 是有 label 的，而 target data 是沒有 label 的，該怎麼處理呢？

比如 source data 是 labeled MNIST 數字集，而 target data 則是加了顏色和背景的 unlabeled 數字集，雖然都是做數字識別，但兩者的情況是非常不匹配的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/domain-adversarial.png" width="60%"/></center>

這個時候一般會把 source data 當做訓練集，而 target data 當做測試集，如果不管訓練集和測試集之間的差異，直接訓練一個普通的 model，得到的結果準確率會相當低

實際上，神經網絡的前幾層可以被看作是在抽取 feature，後幾層則是在做 classification，如果把用 MNIST 訓練好的 model 所提取出的 feature 做 t-SNSE 降維後的可視化，可以發現 MNIST 的數據特徵明顯分為紫色的十團，分別代表 10 個數字，而作為測試集的數據卻是擠成一團的紅色點，因此它們的特徵提取方式根本不匹配

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/domain-adversarial2.png" width="60%"/></center>

所以我們希望前面的特徵提取器(feature extractor)可以把 domain 的特性去除掉，不再使紅點與藍點分成兩群，而是讓它們都混在一起

這裡採取的做法是，在特徵提取器(feature extractor)之後接一個域分類器(domain classifier)，以便分類出這些提取到的 feature 是來自於 MNIST 的數據集還是來自於 MNIST-M 的數據集，這個生成+辨別的架構與 GAN 非常類似

只不過在這裡，feature extractor 可以通過把 feature 全部設為 0，很輕易地騙過 domain classifier，因此還需要給 feature classifier 增加任務的難度，它不只要騙過 domain classifier，還要同時滿足 label predictor 的需求

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/domain-adversarial3.png" width="60%"/></center>

此時通過特徵提取器得到的 feature 不僅可以消除不同 domain 的特性，還要保留原先 digit 的特性，既可以區分不同類別的數字集，又可以正確地識別出不同的數字

通常神經網絡的參數都是朝著最小化 loss 的目標共同前進的，但在這個神經網絡里，三個組成部分的參數各懷鬼胎：

- 對 Label predictor，要把不同數字的分類準確率做的越高越好
- 對 Domain classifier，要正確地區分某張 image 是屬於哪個 domain
- 對 Feature extractor，要提高 Label predictor 的準確率，但要降低 Domain classifier 的準確率

這裡可以看出，Feature extractor 和 Domain classifier 的目標是相反的，要做到這一點，只需要在兩者之間加一層梯度反轉的 layer 即可，當 NN 做 backward 的時候，兩者的參數更新往相反的方向走

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/domain-adversarial4.png" width="60%"/></center>

注意到，Domain classifier 只能接受到 Feature extractor 給到的特徵信息，而無法直接看到圖像的樣子，因此它最後一定會鑒別失敗，所以如何提高 Domain classifier 的能力，讓它經過一番奮力掙扎之後才犧牲是很重要的，如果它一直很弱，就無法把 Feature extractor 的潛能激發到極限

#### Zero-shot Learning

同樣是 source data 有 label，target data 沒有 label 的情況，但在 Zero-shot Learning 中的定義更嚴格一些，它假設 source 和 target 是兩個完全不同的 task，數據完全不相關

在語音識別中，經常會遇到這個問題，畢竟詞彙千千萬萬，總有一些詞彙是訓練時不曾遇到過的，它的處理方法是不要直接將識別的目標定成 word，而是定成 phoneme(音素)，再建立文字與 phoneme 之間的映射表即可

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/zero-shot.png" width="60%"/></center>

在圖像處理中，我們可以把每個類別都用其屬性表示，並且要具備獨一無二的屬性，在數據集中把每種動物按照特性劃分，比如是否毛茸茸、有幾只腳等，在訓練的時候我們不直接去判斷類別，而是去判斷該圖像的屬性，再根據這些屬性去找到最契合的類別即可

有時候屬性的維數也很大，以至於我們對屬性要做 embedding 的降維映射，同樣的，還要把訓練集中的每張圖像都通過某種轉換投影到 embedding space 上的某個點，並且要保證屬性投影的$g(y^i)$和對應圖像投影的$f(x^i)$越接近越好，這裡的$f(x^n)$和$g(y^n)$可以是兩個神經網絡

當遇到新的圖像時，只需要將其投影到相同的空間，即可判斷它與哪個屬性對應的類別更接近

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/zero-shot2.png" width="60%"/></center>

但如果我們根本就無法找出每個動物的屬性$y^i$是什麼，那該怎麼辦？可以使用 word vector，比如直接從維基百科上爬取圖像對應的文字描述，再用 word vector 降維提取特徵，映射到同樣的空間即可

以下這個 loss function 存在些問題，它會讓 model 把所有不同的 x 和 y 都投影到同一個點上：

$$
f^*,g^*=\arg \min\limits_{f,g} \sum\limits_n ||f(x^n)-g(y^n)||_2
$$

類似用 t-SNE 的思想，我們既要考慮同一對$x^n$和$y^n$距離要接近，又要考慮不屬於同一對的$x^n$與$y^m$距離要拉大(這是前面的式子沒有考慮到的)，於是有：

$$
f^*,g^*=\arg \min\limits_{f,g} \sum\limits_n \max(0, k-f(x^n)\cdot g(y^n)+\max\limits_{m\ne n} f(x^n)\cdot g(y^m))
$$
