# Where does the error come from?

#### Review

之前有提到說，不同的 function set，也就是不同的 model，它對應的 error 是不同的；越複雜的 model，也許 performance 會越差，所以今天要討論的問題是，這個 error 來自什麼地方

- error due to ==**bias**==
- error due to ==**variance**==

瞭解 error 的來源其實是很重要的，因為我們可以針對它挑選適當的方法來 improve 自己的 model，提高 model 的準確率，而不會毫無頭緒

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/estimator.png" style="width:60%;" /></center>
#### 抽樣分布

##### $\widehat{y}$和$y^*$ 真值和估測值

$\widehat{y}$表示那個真正的 function，而$f^*$表示這個$\widehat{f}$的估測值 estimator

就好像在打靶，$\widehat{f}$是靶的中心點，收集到一些 data 做 training 以後，你會得到一個你覺得最好的 function 即$f^*$，這個$f^*$落在靶上的某個位置，它跟靶中心有一段距離，這段距離就是由 Bias 和 variance 決定的

bias：偏差；variance：方差 -> 實際上對應著物理實驗中系統誤差和隨機誤差的概念，假設有 n 組數據，每一組數據都會產生一個相應的$f^*$，此時 bias 表示所有$f^*$的平均落靶位置和真值靶心的距離，variance 表示這些$f^*$的集中程度

##### 抽樣分布的理論(概率論與數理統計)

假設獨立變量為 x(這裡的 x 代表每次獨立地從不同的 training data 里訓練找到的$f^*$)，那麼

總體期望$E(x)=u$ ；總體方差$Var(x)=\sigma^2$

###### 用樣本均值$\overline{x}$估測總體期望$u$

由於我們只有有限組樣本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

樣本均值$\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$ ；樣本均值的期望$E(\overline{x})=E(\frac{1}{N}\sum\limits_{i=1}^{N}x^i)=u$ ; 樣本均值的方差$Var(\overline{x})=\frac{\sigma^2}{N}$

**樣本均值 $\overline{x}$的期望是總體期望$u$**，也就是說$\overline{x}$是按概率對稱地分布在總體期望$u$的兩側的；而$\overline{x}$分布的密集程度取決於 N，即數據量的大小，如果 N 比較大，$\overline{x}$就會比較集中，如果 N 比較小，$\overline{x}$就會以$u$為中心分散開來

綜上，==樣本均值$\overline{x}$以總體期望$u$為中心對稱分布，可以用來估測總體期望$u$==

###### 用樣本方差$s^2$估測總體方差$\sigma^2$

由於我們只有有限組樣本 $Sample \ N \ points:\{x^1,x^2,...,x^N\}$，故

樣本均值$\overline{x}=\frac{1}{N}\sum\limits_{i=1}^{N}x^i$ ；樣本方差$s^2=\frac{1}{N-1}\sum\limits_{i=1}^N(x^i-\overline{x})^2$ ；樣本方差的期望$E(s^2)=\sigma^2$ ； 樣本方差的方差$Var(s^2)=\frac{2\sigma^4}{N-1}$

**樣本方差$s^2$的期望是總體方差$\sigma^2$**，而$s^2$分布的密集程度也取決於 N

同理，==樣本方差$s^2$以總體方差$\sigma^2$為中心對稱分布，可以用來估測總體方差$\sigma^2$==

##### 回到 regression 的問題上來

現在我們要估測的是靶的中心$\widehat{f}$，每次 collect data 訓練出來的$f^*$是打在靶上的某個點；產生的 error 取決於：

- 多次實驗得到的$f^*$的期望$\overline{f}$與靶心$\widehat{f}$之間的 bias——$E(f^*)$，可以形象地理解為瞄准的位置和靶心的距離的偏差
- 多次實驗的$f^*$之間的 variance——$Var(f^*)$，可以形象地理解為多次打在靶上的點的集中程度

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bias-variance.png" /></center>
說到這裡，可能會產生一個疑惑：我們之前不就只做了一次實驗嗎？我們就collect了十筆data，然後training出來了一個$f^*$，然後就結束了。那怎麼找很多個$f^*$呢？怎麼知道它的bias和variance有多大呢？

##### $f^*$取決於 model 的複雜程度以及 data 的數量

假設這裡有多個平行宇宙，每個空間里都在用 10 只寶可夢的 data 去找$f^*$，由於不同宇宙中寶可夢的 data 是不同的，因此即使使用的是同一個 model，最終獲得的$f^*$都會是不同的

於是我們做 100 次相同的實驗，把這 100 次實驗找出來的 100 條$f^*$的分布畫出來

###### $f^*$的 variance 取決於 model 的複雜程度和 data 的數量

$f^*$的 variance 是由 model 決定的，一個簡單的 model 在不同的 training data 下可以獲得比較穩定分布的$f^*$，而複雜的 model 在不同的 training data 下的分布比較雜亂(如果 data 足夠多，那複雜的 model 也可以得到比較穩定的分布)

如果採用比較簡單的 model，那麼每次在不同 data 下的實驗所得到的不同的$f^*$之間的 variance 是比較小的，就好像說，你在射擊的時候，每次擊中的位置是差不多的，就如同下圖中的 linear model，100 次實驗找出來的$f^*$都是差不多的

但是如果 model 比較複雜，那麼每次在不同 data 下的實驗所得到的不同的$f^*$之間的 variance 是比較大的，它的散布就會比較開，就如同下圖中含有高次項的 model，每一條$f^*$都長得不太像，並且散布得很開

> 那為什麼比較複雜的 model，它的散布就比較開呢？比較簡單的 model，它的散布就比較密集呢？

原因其實很簡單，其實前面在講 regularization 正規化的時候也提到了部分原因。簡單的 model 實際上就是沒有高次項的 model，或者高次項的系數非常小的 model，這樣的 model 表現得相當平滑，受到不同的 data 的影響是比較小的

舉一個很極端的例子，我們的整個 model(function set)裡面，就一個 function：f=c，這個 function 只有一個常數項，因此無論 training data 怎麼變化，從這個最簡單的 model 里找出來的$f^*$都是一樣的，它的 variance 就是等於 0

###### $f^*$的 bias 只取決於 model 的複雜程度

bias 是說，我們把所有的$f^*$平均起來得到$E(f^*)=\overline{f^*}$，這個$\overline{f^*}$與真值$\widehat{f}$有多接近

當然這裡會有一個問題是說，總體的真值$\widehat{f}$我們根本就沒有辦法知道，因此這裡只是假定了一個$\widehat{f}$

下面的圖示中，**紅色**線條部分代表 5000 次實驗分別得到的$f^*$，**黑色**線條部分代表真實值$\widehat{f}$，**藍色**線條部分代表 5000 次實驗得到的$f^*$的平均值$\overline{f}$

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/5000-tests.png" style="width:60%;" /></center>
根據上圖我們發現，當model比較簡單的時候，每次實驗得到的$f^*$之間的variance會比較小，這些$f^*$會穩定在一個範圍內，但是它們的平均值$\overline{f}$距離真實值$\widehat{f}$會有比較大的偏差；而當model比較複雜的時候，每次實驗得到的$f^*$之間的variance會比較大，實際體現出來就是每次重新實驗得到的$f^*$都會與之前得到的有較大差距，但是這些差距較大的$f^*$的平均值$\overline{f}$卻和真實值$\widehat{f}$比較接近

上圖分別是含有一次項、三次項和五次項的 model 做了 5000 次實驗後的結果，你會發現 model 越複雜，比如含有 5 次項的 model 那一幅圖，每一次實驗得到的$f^*$幾乎是雜亂無章，遍布整幅圖的；但是他們的平均值卻和真實值$\widehat{f}$吻合的很好。也就是說，複雜的 model，單次實驗的結果是沒有太大參考價值的，但是如果把考慮多次實驗的結果的平均值，也許會對最終的結果有幫助

注：這裡的單次實驗指的是，用一組 training data 訓練出 model 的一組有效參數以構成$f^*$(每次獨立實驗使用的 training data 都是不同的)

###### 因此：

- 如果是一個比較簡單的 model，那它有比較小的 variance 和比較大的 bias。就像下圖中左下角的打靶模型，每次實驗的$f^*$都比較集中，但是他們平均起來距離靶心會有一段距離(比較適合實驗次數少甚至只有單次實驗的情況)
- 如果是一個比較複雜的 model，每次實驗找出來的$f^*$都不一樣，它有比較大的 variance 但是卻有比較小的 bias。就像下圖中右下角的打靶模型，每次實驗的$f^*$都比較分散，但是他們平均起來的位置與靶心比較接近(比較適合多次實驗的情況)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/model-bias.png" style="width:60%;" /></center>
###### 為什麼會這樣？

實際上我們的 model 就是一個 function set，當你定好一個 model 的時候，實際上就已經定好這個 function set 的範圍了，那個最好的 function 只能從這個 function set 裡面挑出來

如果是一個簡單的 model，它的 function set 的 space 是比較小的，這個範圍可能根本就沒有包含你的 target；如果這個 function set 沒有包含 target，那麼不管怎麼 sample，平均起來永遠不可能是 target(這裡的 space 指上圖中左下角那個被 model 圈起來的空間)

如果這個 model 比較複雜，那麼這個 model 所代表的 function set 的 space 是比較大的(簡單的 model 實際上就是複雜 model 的子集)，那它就很有可能包含 target，只是它沒有辦法找到那個 target 在哪，因為你給的 training data 不夠，你給的 training data 每一次都不一樣，所以他每一次找出來的$f^*$都不一樣，但是如果他們是散布在這個 target 附近的，那平均起來，實際上就可以得到和 target 比較接近的位置(這裡的 space 指上圖中右下角那個被 model 圈起來的空間)

#### Bias vs Variance

由前面的討論可知，比較簡單的 model，variance 比較小，bias 比較大；而比較複雜的 model，bias 比較小，variance 比較大

##### bias 和 variance 對 error 的影響

因此下圖中(也就是之前我們得到的從最高項為一次項到五次項的五個 model 的 error 表現)，綠色的線代表 variance 造成的 error，紅色的線代表 bias 造成的 error，藍色的線代表這個 model 實際觀測到的 error

$error_{實際}=error_{variance}+error_{bias}——藍線為紅線和綠線之和$

可以發現，隨著 model 的逐漸複雜：

- bias 逐漸減小，bias 所造成的 error 也逐漸下降，也就是打靶的時候瞄得越來越准，體現為圖中的紅線
- variance 逐漸變大，variance 所造成的 error 也逐漸增大，也就是雖然瞄得越來越准，但是每次射出去以後，你的誤差是越來越大的，體現為圖中的綠線
- 當 bias 和 variance 這兩項同時被考慮的時候，得到的就是圖中的藍線，也就是實際體現出來的 error 的變化；實際觀測到的 error 先是減小然後又增大，因此實際 error 為最小值的那個點，即為 bias 和 variance 的 error 之和最小的點，就是表現最好的 model
- ==**如果實際 error 主要來自於 variance 很大，這個狀況就是 overfitting 過擬合；如果實際 error 主要來自於 bias 很大，這個狀況就是 underfitting 欠擬合**==(可以理解為，overfitting 就是過分地包圍了靶心所在的 space，而 underfitting 則是還未曾包圍到靶心所在的 space)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/bias-vs-variance.png" style="width:60%;"></center>
這就是為什麼我們之前要先計算出每一個model對應的error(每一個model都有唯一對應的$f^*$，因此也有唯一對應的error)，再挑選error最小的model的原因，只有這樣才能綜合考慮bias和variance的影響，找到一個實際error最小的model

##### 必須要知道自己的 error 主要來自於哪裡

###### 你現在的問題是 bias 大，還是 variance 大？

當你自己在做 research 的時候，你必須要搞清楚，手頭上的這個 model，它目前主要的 error 是來源於哪裡；你覺得你現在的問題是 bias 大，還是 variance 大

你應該先知道這件事情，你才能知道你的 future work，你要 improve 你的 model 的時候，你應該要走哪一個方向

###### 那怎麼知道現在是 bias 大還是 variance 大呢？

- 如果 model 沒有辦法 fit training data 的 examples，代表 bias 比較大，這時是 underfitting

  形象地說，就是該 model 找到的$f^*$上面並沒有 training data 的大部分樣本點，如下圖中的 linear model，我們只是 example 抽樣了這幾個藍色的樣本點，而這個 model 甚至沒有 fit 這少數幾個藍色的樣本點(這幾個樣本點沒有在$f^*$上)，代表說這個 model 跟正確的 model 是有一段差距的，所以這個時候是 bias 大的情況，是 underfitting

- 如果 model 可以 fit training data，在 training data 上得到小的 error，但是在 testing data 上，卻得到一個大的 error，代表 variance 比較大，這時是 overfitting

###### 如何針對性地處理 bias 大 or variance 大的情況呢？

遇到 bias 大或 variance 大的時候，你其實是要用不同的方式來處理它們

1、**如果 bias 比較大**

bias 大代表，你現在這個 model 裡面可能根本沒有包含你的 target，$\widehat{f}$可能根本就不在你的 function set 里

對於 error 主要來自於 bias 的情況，是由於該 model(function set)本來就不好，collect 更多的 data 是沒有用的，必須要從 model 本身出發

- redesign，重新設計你的 model

  - 增加更多的 features 作為 model 的 input 輸入變量

    比如 pokemon 的例子里，只考慮進化前 cp 值可能不夠，還要考慮 hp 值、species 種類...作為 model 新的 input 變量

  - 讓 model 變得更複雜，增加高次項

    比如原本只是 linear model，現在考慮增加二次項、三次項...

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/large-bias.png" style="width:60%;" /></center>
2、**如果variance比較大**

- 增加 data
  - 如果是 5 次式，找 100 個$f^*$，每次實驗我們只用 10 只寶可夢的數據訓練 model，那我們找出來的 100 個$f^*$的散布就會像下圖一樣雜亂無章；但如果每次實驗我們用 100 只寶可夢的數據訓練 model，那我們找出來的 100 個$f^*$的分布就會像下圖所示一樣，非常地集中
  - 增加 data 是一個很有效控制 variance 的方法，假設你 variance 太大的話，collect data 幾乎是一個萬能丹一樣的東西，並且它不會傷害你的 bias
  - 但是它存在一個很大的問題是，實際上並沒有辦法去 collect 更多的 data
  - 如果沒有辦法 collect 更多的 data，其實有一招，根據你對這個問題的理解，自己去 generate 更多「假的」data
    - 比如手寫數字識別，因為每個人手寫數字的角度都不一樣，那就把所有 training data 裡面的數字都左轉 15°，右轉 15°
    - 比如做火車的影像辨識，只有從左邊開過來的火車影像資料，沒有從右邊開過來的火車影像資料，該怎麼辦？實際上可以把每張圖片都左右顛倒，就 generate 出右邊的火車數據了，這樣就多了一倍 data 出來
    - 比如做語音辨識的時候，只有男生說的「你好」，沒有女生說的「你好」，那就用男生的聲音用一個變聲器把它轉化一下，這樣男女生的聲音就可以互相轉化，這樣 data 就可以多出來
    - 比如現在你只有錄音室里錄下的聲音，但是 detection 實際要在真實場景下使用的，那你就去真實場景下錄一些噪音加到原本的聲音里，就可以 generate 出符合條件的 data 了
- Regularization(正規化)
  - 就是在 loss function 裡面再加一個與 model 高次項系數相關的 term，它會希望你的 model 里高次項的參數越小越好，也就是說希望你今天找出來的曲線越平滑越好；這個新加的 term 前面可以有一個 weight，代表你希望你的曲線有多平滑
  - 下圖中 Regularization 部分，左邊第一幅圖是沒有加 regularization 的 test；第二幅圖是加了 regularization 後的情況，一些怪怪的、很不平滑的曲線就不會再出現，所有曲線都集中在比較平滑的區域；第三幅圖是增加 weight 的情況，讓曲線變得更平滑
  - 加了 regularization 以後，因為你強迫所有的曲線都要比較平滑，所以這個時候也會讓你的 variance 變小；但 regularization 是可能會傷害 bias 的，因為它實際上調整了 function set 的 space 範圍，變成它只包含那些比較平滑的曲線，這個縮小的 space 可能沒有包含原先在更大 space 內的$\widehat{f}$，因此傷害了 bias，所以當你做 regularization 的時候，需要調整 regularization 的 weight，在 variance 和 bias 之間取得平衡

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/large-variance.png" style="width:60%;"/></center>
注：variance比較大的case，加以圖例解釋如下：(假設這裡我們無法獲得更多的data)

1、藍色區域代表最初的情況，此時 model 比較複雜，function set 的 space 範圍比較大，包含了 target 靶心，但由於 data 不夠，$f^*$比較分散，variance 比較大

2、紅色區域代表進行 regularization 之後的情況，此時 model 的 function set 範圍被縮小成只包含平滑的曲線，space 減小，variance 當然也跟著變小，但這個縮小後的 space 實際上並沒有包含原先已經包含的 target 靶心，因此該 model 的 bias 變大

3、橙色區域代表增大 regularization 的 weight 的情況，增大 weight 實際上就是放大 function set 的 space，慢慢調整至包含 target 靶心，此時該 model 的 bias 變小，而相較於一開始的 case，由於限定了曲線的平滑度(由 weight 控制平滑度的閾值)，該 model 的 variance 也比較小

實際上，通過 regularization 優化 model 的過程就是上述的 1、2、3 步驟，不斷地調整 regularization 的 weight，使 model 的 bias 和 variance 達到一個最佳平衡的狀態(可以通過 error 來評價狀態的好壞，weight 需要慢慢調參)

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/regularization-illustration.png" style="width:60%;"/></center>
#### Model Selection

我們現在會遇到的問題往往是這樣：我們有很多個 model 可以選擇，還有很多參數可以調，比如 regularization 的 weight，那通常我們是在 bias 和 variance 之間做一些 trade-off 權衡

我們希望找一個 model，它 variance 夠小，bias 也夠小，這兩個合起來給我們最小的 testing data 的 error

##### 但是以下這些事情，是你不應該做的：

你手上有 training set，有 testing set，接下來你想知道 model1、model2、model3 裡面，應該選哪一個 model，然後你就分別用這三個 model 去訓練出$f_1^*,f_2^*,f_3^*$，然後把它 apply 到 testing set 上面，分別得到三個 error 為 0.9，0.7，0.5，這裡很直覺地會認為是 model3 最好

但是現在可能的問題是，這個 testing set 是你自己手上的 testing set，是你自己拿來衡量 model 好壞的 testing set，真正的 testing set 是你沒有的；注意到你自己手上的這筆 testing set，它有自己的一個 bias(這裡的 bias 跟之前提到的略有不同，可以理解為自己的 testing data 跟實際的 testing data 會有一定的偏差存在)

所以你今天那這個 testing set 來選擇最好的 model 的時候，它在真正的 testing set 上不見得是最好的 model，通常是比較差的，所以你實際得到的 error 是會大於你在自己的 testing set 上估測到的 0.5

以 PM2.5 預測為例，提供的數據分為 training set，public testing set 和 private testing set 三部分，其中 public 的 testing set 是供你測試自己的 model 的，private 的 testing data 是你暫且未知的真正測試數據，現在你的 model3 在 public testing set 上的 error 為 0.5，已經成功 beat baseline，但是在 private 的 testing set 上，你的 model3 也許根本就沒有 beat the baseline，反而是 model1 和 model2 可能會表現地更好

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/model-selection.png" style="width:60%;"/></center>
##### 怎樣做才是可靠的呢？

###### training data 分成 training set 和 validation set

你要做的事情是，把你的 training set 分成兩組：

- 一組是真正拿來 training model 的，叫做 training set(訓練集)
- 另外一組不拿它來 training model，而是拿它來選 model，叫做 validation set(驗證集)

==先在 training set 上找出每個 model 最好的 function $f^*$，然後用 validation set 來選擇你的 model==

也就是說，你手頭上有 3 個 model，你先把這 3 個 model 用 training set 訓練出三個$f^*$，接下來看一下它們在 validation set 上的 performance

假設現在 model3 的 performance 最好，那你可以直接把這個 model3 的結果拿來 apply 在 testing data 上

如果你擔心現在把 training set 分成 training 和 validation 兩部分，感覺 training data 變少的話，可以這樣做：已經從 validation 決定 model3 是最好的 model，那就定住 model3 不變(function 的表達式不變)，然後用全部的 data 在 model3 上面再訓練一次(使用全部的 data 去更新 model3 表達式的參數)

這個時候，如果你把這個訓練好的 model 的$f^*$apply 到 public testing set 上面，你可能會得到一個大於 0.5 的 error，雖然這麼做，你得到的 error 表面上看起來是比較大的，但是**這個時候你在 public set 上的 error 才能夠真正反映你在 private set 上的 error**

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/cross-validation.png" style="width:60%;"/></center>
###### 考慮真實的測試集

實際上是這樣一個關係：

> training data(訓練集) -> 自己的 testing data(測試集) -> 實際的 testing data
> (該流程沒有考慮自己的 testing data 的 bias)

> training set(部分訓練集) -> validation set(部分驗證集) -> 自己的 testing data(測試集) -> 實際的 testing data
> (該流程使用自己的 testing data 和 validation 來模擬 testing data 的 bias 誤差，可以真實地反映出在實際的 data 上出現的 error)

###### 真正的 error

當你得到 public set 上的 error 的時候(儘管它可能會很大)，不建議回過頭去重新調整 model 的參數，因為當你再回去重新調整什麼東西的時候，你就又會把 public testing set 的 bias 給考慮進去了，這就又回到了第一種關係，即圍繞著有偏差的 testing data 做 model 的優化

這樣的話此時你在 public set 上看到的 performance 就沒有辦法反映實際在 private set 上的 performance 了，因為你的 model 是針對 public set 做過優化的，雖然 public set 上的 error 數據看起來可能會更好看，但是針對實際未知的 private set，這個「優化」帶來的可能是反作用，反而會使實際的 error 變大

當然，你也許幾乎沒有辦法忍住不去做這件事情，在發 paper 的時候，有時候你會 propose 一個方法，那你要 attach 在 benchmark 的 corpus，如果你在 testing set 上得到一個差的結果，你也幾乎沒有辦法把持自己不回頭去調一下你的 model，你肯定不會只是寫一個 paper 說這個方法不 work 這樣子(滑稽

因此這裡只是說，你要 keep in mind，如果在那個 benchmark corpus 上面所看到的 testing 的 performance，它的 error，肯定是大於它在 real 的 application 上應該有的值

比如說你現在常常會聽到說，在 image lab 的那個 corpus 上面，error rate 都降到 3%，那個是超越人類了，但是真的是這樣子嗎？已經有這麼多人玩過這個 corpus，已經有這麼多人告訴你說前面這些方法都不 work，他們都幫你挑過 model 了，你已經用「testing」 data 調過參數了，所以如果你把那些 model 真的 apply 到現實生活中，它的 error rate 肯定是大於 3%的

###### 如何劃分 training set 和 validation set？

那如果 training set 和 validation set 分壞了怎麼辦？如果 validation 也有怪怪的 bias，豈不是對結果很不利？那你要做下面這件事情：

==**N-flod Cross Validation**==

如果你不相信某一次分 train 和 validation 的結果的話，那你就分很多種不同的樣子

比如說，如果你做 3-flod 的 validation，意思就是你把 training set 分成三份，你每一次拿其中一份當做 validation set，另外兩份當 training；分別在每個情境下都計算一下 3 個 model 的 error，然後計算一下它的 average error；然後你會發現在這三個情境下的 average error，是 model1 最好

然後接下來，你就把用整個完整的 training data 重新訓練一遍 model1 的參數；然後再去 testing data 上 test

原則上是，如果你少去根據 public testing set 上的 error 調整 model 的話，那你在 private testing set 上面得到的 error 往往是比較接近 public testing set 上的 error 的

<center><img src="https://gitee.com/Sakura-gh/ML-notes/raw/master/img/n-flod-cross-validation.png" style="width:60%;"/></center>
#### 總結conclusion

1、一般來說，error 是 bias 和 variance 共同作用的結果

2、model 比較簡單和比較複雜的情況：

- 當 model 比較簡單的時候，variance 比較小，bias 比較大，此時$f^*$會比較集中，但是 function set 可能並沒有包含真實值$\widehat{f}$；此時 model 受 bias 影響較大
- 當 model 比較複雜的時候，bias 比較小，variance 比較大，此時 function set 會包含真實值$\widehat{f}$，但是$f^*$會比較分散；此時 model 受 variance 影響較大

3、區分 bias 大 or variance 大的情況

- 如果連採樣的樣本點都沒有大部分在 model 訓練出來的$f^*$上，說明這個 model 太簡單，bias 比較大，是欠擬合

- 如果樣本點基本都在 model 訓練出來的$f^*$上，但是 testing data 上測試得到的 error 很大，說明這個 model 太複雜，variance 比較大，是過擬合

4、bias 大 or variance 大的情況下該如何處理

- 當 bias 比較大時，需要做的是重新設計 model，包括考慮添加新的 input 變量，考慮給 model 添加高次項；然後對每一個 model 對應的$f^*$計算出 error，選擇 error 值最小的 model(隨 model 變複雜，bias 會減小，variance 會增加，因此這裡分別計算 error，取兩者平衡點)

- 當 variance 比較大時，一個很好的辦法是增加 data(可以憑借經驗自己 generate data)，當 data 數量足夠時，得到的$f^*$實際上是比較集中的；如果現實中沒有辦法 collect 更多的 data，那麼就採用 regularization 正規化的方法，以曲線的平滑度為條件控制 function set 的範圍，用 weight 控制平滑度閾值，使得最終的 model 既包含$\widehat{f}$，variance 又不會太大

5、如何選擇 model

- 選擇 model 的時候呢，我們手頭上的 testing data 與真實的 testing data 之間是存在偏差的，因此我們要將 training data 分成 training set 和 validation set 兩部分，經過 validation 挑選出來的 model 再用全部的 training data 訓練一遍參數，最後用 testing data 去測試 error，這樣得到的 error 是模擬過 testing bias 的 error，與實際情況下的 error 會比較符合
