__version_info__ = (3, 1, 1)
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
- add polynomial cutoff layer
(1.0.1)
- modify angular distribution; not using torch.arccos()
(1.0.2)
- add NanStoppingHook
- optimization of training process
(1.0.3)
- add script_utils.py and change the usage of script(only from json)
(1.0.4)
- multiple with node embedding of neihgbor j and neighbor k.
(1.0.5)
- add feture weighting to schnettriple(substute for residual networks)
(1.0.6)
- change the location of applying the cutoff function.
- add residual net in cfconv
(1.0.7)
- remove residual net in cfconv
(1.0.8)
- remove feature weighting in last residual networks.

(2.0.0)
- separate double and triple interaction blocks.
(2.1.0)
- cutoff apply after filter generator
- add node j and node k in CFconvTriple
(2.1.1)
- change angular term to BF descriptor
(2.1.2)
- cutoff apply before filter generator
- change angular term to ANI-1
(2.1.3)
- add trainble_gaussian parameter
(3.1.0)
- change outputmodule; weight value for triple is 1.0e-5.
(3.1.1)
- remove weight init for triple (it's not meaningful)
- add triple_ijk before concatinate with atom embeddings
- simply concatinate angular featrues to atom embeddings
- add trainable_theta parameter.
"""
