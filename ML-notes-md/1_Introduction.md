# Introduction

> define a set of function(model) -> goodness of function -> pick the best function 

### Learning Map

下圖中，同樣的顏色指的是同一個類型的事情

藍色方塊指的是scenario，即學習的情境。通常學習的情境是我們沒有辦法控制的，比如做reinforcement Learning是因為我們沒有data、沒有辦法來做supervised Learning的情況下才去做的。如果有data，supervised Learning當然比reinforcement Learning要好；因此手上有什麼樣的data，就決定你使用什麼樣的scenario

紅色方塊指的是task，即要解決的問題。你要解的問題，隨著你要找的function的output的不同，有輸出scalar的regression、有輸出options的classification、有輸出structured object的structured Learning...

綠色的方塊指的是model，即用來解決問題的模型(function set)。在這些task裡面有不同的model，也就是說，同樣的task，我們可以用不同的方法來解它，比如linear model、Non-linear model(deep Learning、SVM、decision tree、K-NN...)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/learningMap.png" alt="learning map" width="60%;" /></center>

#### Supervised Learning(監督學習)

supervised learning 需要大量的training data，這些training data告訴我們說，一個我們要找的function，它的input和output之間有什麼樣的關係

而這種function的output，通常被叫做label(標籤)，也就是說，我們要使用supervised learning這樣一種技術，我們需要告訴機器，function的input和output分別是什麼，而這種output通常是通過人工的方式標注出來的，因此稱為人工標注的label，它的缺點是需要大量的人工effort

##### Regression(回歸)

regression是machine learning的一個task，特點是==通過regression找到的function，它的輸出是一個scalar數值==

比如PM2.5的預測，給machine的training data是過去的PM2.5資料，而輸出的是對未來PM2.5的預測**數值**，這就是一個典型的regression的問題

##### Classification(分類)

regression和classification的區別是，我們要機器輸出的東西的類型是不一樣的，在regression里機器輸出的是scalar，而classification又分為兩類：

###### Binary Classification(二元分類)

在binary classification里，我們要機器輸出的是yes or no，是或否

比如G-mail的spam filtering(垃圾郵件過濾器)，輸入是郵件，輸出是該郵件是否是垃圾郵件

###### Multi-class classification(多元分類)

在multi-class classification里，機器要做的是選擇題，等於給他數個選項，每一個選項就是一個類別，它要從數個類別裡面選擇正確的類別

比如document classification(新聞文章分類)，輸入是一則新聞，輸出是這個新聞屬於哪一個類別(選項)

##### model(function set) 選擇模型

在解任務的過程中，第一步是要選一個function的set，選不同的function set，會得到不同的結果；而選不同的function set就是選不同的model，model又分為很多種：

* Linear Model(線性模型)：最簡單的模型

* Non-linear Model(非線性模型)：最常用的模型，包括：

    * **deep learning**

        如alpha-go下圍棋，輸入是當前的棋盤格局，輸出是下一步要落子的位置；由於棋盤是19\*19的，因此可以把它看成是一個有19\*19個選項的選擇題

    * **SVM**

    * **decision tree**

    * **K-NN**

#### Semi-supervised Learning(半監督學習)

舉例：如果想要做一個區分貓和狗的function

手頭上有少量的labeled data，它們標注了圖片上哪只是貓哪只是狗；同時又有大量的unlabeled data，它們僅僅只有貓和狗的圖片，但沒有標注去告訴機器哪只是貓哪只是狗

在Semi-supervised Learning的技術裡面，這些沒有labeled的data，對機器學習也是有幫助的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/semi-supervised-Learning.png" alt="semi-supervised" width="60%;" /></center>

#### Transfer Learning(遷移學習)

假設一樣我們要做貓和狗的分類問題

我們也一樣只有少量的有labeled的data；但是我們現在有大量的不相干的data(不是貓和狗的圖片，而是一些其他不相干的圖片)，在這些大量的data裡面，它可能有label也可能沒有label

Transfer Learning要解決的問題是，這一堆不相干的data可以對結果帶來什麼樣的幫助

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/transfer-Learning.png" alt="transfer" width="60%;" /></center>

#### Unsupervised Learning(無監督學習)

區別於supervised learning，unsupervised learning希望機器學到無師自通，在完全沒有任何label的情況下，機器到底能學到什麼樣的知識

舉例來說，如果我們給機器看大量的文章，機器看過大量的文章之後，它到底能夠學到什麼事情？它能不能學會每個詞彙的意思？

學會每個詞彙的意思可以理解為：我們要找一個function，然後把一個詞彙丟進去，機器要輸出告訴你說這個詞彙是什麼意思，也許他用一個向量來表示這個詞彙的不同的特性，不同的attribute

又比如，我們帶機器去逛動物園，給他看大量的動物的圖片，對於unsupervised learning來說，我們的data中只有給function的輸入的大量圖片，沒有任何的輸出標注；在這種情況下，機器該怎麼學會根據testing data的輸入來自己生成新的圖片？

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/unsupervised-Learning.png" width="60%;" /></center>

#### Structured Learning(結構化學習)

在structured Learning里，我們要機器輸出的是，一個有結構性的東西

在分類的問題中，機器輸出的只是一個選項；在structured類的problem裡面，機器要輸出的是一個複雜的物件

舉例來說，在語音識別的情境下，機器的輸入是一個聲音信號，輸出是一個句子；句子是由許多詞彙拼湊而成，它是一個有結構性的object

或者說機器翻譯、人臉識別(標出不同的人的名稱)

比如**GAN**也是structured Learning的一種方法

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/structured-Learning.png" alt="structured" width="60%;" /></center>

#### Reinforcement Learning(強化學習)

**Supervised Learning**：我們會告訴機器正確的答案是什麼 ，其特點是**Learning from teacher**

* 比如訓練一個聊天機器人，告訴他如果使用者說了「Hello」，你就說「Hi」；如果使用者說了「Bye bye」，你就說「Good bye」；就好像有一個家教在它的旁邊手把手地教他每一件事情

**Reinforcement Learning**：我們沒有告訴機器正確的答案是什麼，機器最終得到的只有一個分數，就是它做的好還是不好，但他不知道自己到底哪裡做的不好，他也沒有正確的答案；很像真實社會中的學習，你沒有一個正確的答案，你只知道自己是做得好還是不好。其特點是**Learning from critics**

* 比如訓練一個聊天機器人，讓它跟客人直接對話；如果客人勃然大怒把電話掛掉了，那機器就學到一件事情，剛才做錯了，它不知道自己哪裡做錯了，必須自己回去反省檢討到底要如何改進，比如一開始不應該打招呼嗎？還是中間不能罵臟話之類的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/reinforcement-Learning.png" alt="reinforcement" width="60%;" /></center>

再拿下棋這件事舉例，supervised Learning是說看到眼前這個棋盤，告訴機器下一步要走什麼位置；而reinforcement Learning是說讓機器和對手互弈，下了好幾手之後贏了，機器就知道這一局棋下的不錯，但是到底哪一步是贏的關鍵，機器是不知道的，他只知道自己是贏了還是輸了

其實Alpha Go是用supervised Learning+reinforcement Learning的方式去學習的，機器先是從棋譜學習，有棋譜就可以做supervised的學習；之後再做reinforcement Learning，機器的對手是另外一台機器，Alpha Go就是和自己下棋，然後不斷的進步