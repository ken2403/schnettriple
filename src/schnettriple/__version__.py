__version_info__ = (1, 0, 0)
__version__ = ".".join(map(str, __version_info__))
"""
(0.1.5)
- 3体のfilterをconcatinateせずに、2体のfileterと別々にconvolutionする
- reshapeとaggを2体と3体とで別々に行う。
- 学習の際のparameterを小さくする(featuresとnum_gaussinas)ことでモデルサイズを抑える
- filter networkの2個目の層にもactivationを追加
- zetaをregister_bufferに入れる
(0.2.0)
- mappingをangularに名前を変え、CFconvではなく、SchNetTripleのなかで行うように変更(勾配が消失してしまう)
(0.2.2)
- cos_thetaの計算を外に出す
(0.3.0)
- positionの入力を一個にするために、r_doubleをr_ijk[0]で代用
- neighbors_k の効果を足し算して追加
(0.3.1)
- positionの入力を一個にするために、r_doubleをr_ijk[0]で代用
- neighbors_k の効果を足し算して追加
- torch.reshape → torch.viewに変更(viewはメモリーを共有)
(0.4.0)
- triple_distance_expansion()(GassianSmearing)のcentered=Falseとして、expのマイナス乗を避ける
(0.4.1)
- double filter のconvolutionをなくす
- torch.gatherをするときに、neighborの距離に応じた重み付けを行う
(0.4.2)
- double filterを元に戻す
- doubleとtiripleの間に一層全結合層を足す
- hyper parameterのnum_gaussians=30の方が良い
(0.4.3)
- loss functionに正則化項を加えて学習できるようにscriptを修正
(0.4.4)
- cfconv でdoubleからtripleへの全結合層なくす
- r_jkを使ってtriple distributionを求める
- pbcのreshapeのところは、neighbor_jのみの和を取る形に変更
- neighborのindexが-1とならないようにloaderを修正
(0.4.5)
- gussian expand した距離の値にneighbor maskを適用する
(1.0.0)
- change the use of triple featrues.
"""
