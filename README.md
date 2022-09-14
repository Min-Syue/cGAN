 # **Conditional GAN (CGAN)**

## CGAN是什麼?

>Conditional GAN 在基於 GAN 下，增加輸入條件，改善原本 GAN 無法生成特定圖片的問題。 <br>
>因此 cGAN 將原本 GAN 的損失函數修改如下：<br>
>
>$$\underset{G}{\min} \ \underset{D}{\max} \ V(D,G) = \mathbb{E}\_{x \sim p_{data}(x)} \left[ \log \ D(x|y) \right] + \mathbb{E}\_{z \sim p_{z}(z)} \left[\log \ (1 - D(x|y)) \right]$$
>
>可以注意到cGAN損失函數裡的 Discriminator 修改為 $D(x|y)$ (在 GAN 或 DCGAN 中，為 $D(x)$ )，而此一改動代表著不僅有 noise 作為 Generator 的輸入，加入了 label 來共同當成 Generator 的輸入， Discriminator 也是如此，因此改動後的流程圖如下：
>
>![](https://www.mathworks.com/help/examples/nnet/win64/TrainConditionalGenerativeAdversarialNetworkCGANExample_02.png) <br>
>圖片取自 [這裡](https://www.mathworks.com/help/deeplearning/ug/train-conditional-generative-adversarial-network.html)

## 生成結果
> ### Mnist 結果
>![](https://github.com/Min-Syue/cGAN/blob/main/cGAN_50epochs.gif)
