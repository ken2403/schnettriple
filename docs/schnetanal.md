# SchNetでの解析の流れ

1. データの準備<br>  
    *.xyz* 形式のファイルを準備<br>

2. データを変換<br>
    データをASEのデータベースの形式に変換する.<br>
    `$ spk_parse.py <.xyzのファイルのパス> <新たに.dbファイルを作るパス>`<br>

3. モデルの学習<br>
    1. 学習したモデルを保存するディレクトリを作る<br>
    2. モデルのハイパーパラメータを決めたJSONファイルを, *args.json* として同じディレクトリ中に用意する.(既にあるファイルをコピーして利用すると良い)<br>
    3. 計算開始<br>
       `$ spk_run.py from_json <modeldir>/args.json`

4. 結果の解析<br>
    1. 学習したモデルを利用してvalidationする(これにもGPU必要).結果は *< modeldir>/evaluation.txt* に記録される.<br>
       `$ spk_run eval <datapath> <modeldir> [--split {train validation test}] [--cuda]`<br>
       GPUのメモリーを超えてしまう場合は,バッチサイズをオプション引数で`--batch_size <batch>`のように指定してevaluationする.<br>
       得られる結果はMAEとRMSEなので,個々の値に対する予測結果は別途計算する必要がある.<br>

    2. 学習の結果は *< modeldir>/log/log.csv* にあるので,読み込んで解析する.<br>
       このファイルも同様にloss,MAE,RMSEの値のみが記録されている.<br>

    3. 個々のデータに対する予測値の計算は,以下の流れで行う.<br>

        - 学習したモデルは *< modeldir>/best_model* に記録されている.<br>
            ```python
            # モデルの読み込み
            import torch
            best_model = torch.load('<modeldir>/best_model')
            ```
            
        - test,test,validationのsplitの情報は *< modeldir>/split.npz* に記録されている.<br>
            ```python
            # npzファイルの読み込み
            import numpy as np
            split = np.load('<modeldir>/split.npz')
            # 複数のnp.arrayが格納されている
            >>> split.files 
            ['train_idx', 'val_idx', 'test_idx', 'mean', 'stddev'] 

            # validationのためのデータのindexがnp.array形式で格納されている
            split['val_idx'] 
            ```
        - databaseの読み込み<br>
            datasetは**schnetpack.AtomsData**オブジェクトになる.<br>
            ```python
            import schnetpack
            # AtomsDataオブジェクトとしてデータベースを読み込む必要がある
            dataset = schnetpack.AtomsData('dbファイルのパス')
            # 計算に利用可能なpropertiesを持つ属性
            >>> dataset.available_properties
            ['forces', 'energy']

            # 各データは辞書形式で格納されている.各値はtorchのtensorの形式になっている.
            >>> for k, v in dataset[0].items():
                    print('-- {key}:'.format(key=k), v.shape)
            -- forces: torch.Size([31, 3])
            -- energy: torch.Size([1])
            -- _atomic_numbers: torch.Size([31])
            -- _positions: torch.Size([31, 3])
            -- _cell: torch.Size([3, 3])
            -- _neighbors: torch.Size([31, 30])
            -- _cell_offset: torch.Size([31, 30, 3])
            -- _idx: torch.Size([1])
            ```
        - schnetpack.AtomsConverterによってschentの入力になるようにdatabaseを変換<br>
            AtomsConverterは**ase.atoms.Atoms**オブジェクトを,schnetの入力として適切な形式(dictで情報を持つ)に変換する.<br>
            ```python
            device = 'cpu'
            # converterオブジェクトの生成
            converter = spk.data.AtomsConverter(device=device)

            at, props = dataset.get_properties(idx=3)
            # atはase.atoms.Atomオブジェクト,propsは格子の情報を持ったdictオブジェクト
            # atを変換してschnetの入力とする
            inputs = converter(at)

            # apply model
            pred = best_model(inputs)

            # propsもpredもtorch.Tensor形式なので適宜変換して表示している
            print('Truth:', props['energy'].cpu().numpy()[0])
            print('Prediction:', pred['energy'].detach().cpu().numpy()[0,0])
            ```




