__version_info__ = (0, 2, 1)
"""
(0.1.5)
- 3体のfilterをconcatinateせずに、2体のfileterと別々にconvolutionする
- reshapeとaggを2体と3体とで別々に行う。
- 学習の際のparameterを小さくする(featuresとnum_gaussinas)ことでモデルサイズを抑える
- filter networkの2個目の層にもactivationを追加
- zetaをregister_bufferに入れる
(0.2.0)
- mappingをangularに名前を変え、CFconvではなく、SchNetTripleのなかで行うように変更(勾配が消失してしまう)
"""
__version__ = ".".join(map(str, __version_info__))
