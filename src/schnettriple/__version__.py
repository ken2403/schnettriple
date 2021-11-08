__version_info__ = (0, 3, 33)
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
"""
__version__ = ".".join(map(str, __version_info__))
