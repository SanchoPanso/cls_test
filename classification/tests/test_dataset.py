import pytest
import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from train.datasets import ImageDataset, TitsSizeDataset


def test_delete_badlist():
    mydict = [
        {'path': '1.png', 'a': 2,    'b': 3,    'c': 4},
        {'path': '2.png', 'a': 200,  'b': 300,  'c': 400},
        {'path': '3.png', 'a': 2000, 'b': 3000, 'c': 4000 }
    ]
    df = pd.DataFrame(mydict)
    
    df1 = ImageDataset.delete_badlist(df, [])
    assert len(df1) == 3
    
    df1 = ImageDataset.delete_badlist(df, ['-1.png'])
    assert len(df1) == 3
    
    df1 = ImageDataset.delete_badlist(df, ['1.png'])
    assert len(df1) == 2
    
    df1 = ImageDataset.delete_badlist(df, ['1.png', '2.png', '3.png'])
    assert len(df1) == 0


def test_tits_size_data():
    mydict = [
        {'path': '1.png', 'cls1': 1,},
        {'path': '2.png', 'cls1': 0,},
        {'path': '3.png', 'cls1': 0,},
        {'path': '4.png', 'cls1': 0,},
        {'path': '5.png', 'cls1': 0,},
    ]
    df = pd.DataFrame(mydict)
    ds = TitsSizeDataset(df, None, str(Path(__file__).parent / 'test_dataset'))
    data = ds.data
    assert data.columns.tolist() == ['path', 'cls1', 'trash_bg', 'trash_male', 'trash_female']
    assert data['path'].tolist() == ['1.png'] + ['2.png'] * 2 + ['3.png'] * 2 + ['4.png'] * 3 + ['5.png']
    assert data['trash_bg'].tolist() == [0, 1, 0, 1, 0, 1, 0, 0, 1]
    

if __name__ == '__main__':
    sys.exit(pytest.main([__file__]))
