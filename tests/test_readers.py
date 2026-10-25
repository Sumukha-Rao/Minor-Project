import json
import pandas as pd
import pytest
from promptguard.data import read_table


@pytest.mark.parametrize('suffix',['csv','json','jsonl'])
def test_local_readers_preserve_payload(tmp_path,suffix):
    d=pd.DataFrame({'text':['[SYS]\nDo NOT strip: <tag> & {x}!','normal text'],'label':['injection','benign']})
    path=tmp_path/('data.'+suffix)
    if suffix=='csv':d.to_csv(path,index=False)
    else:d.to_json(path,orient='records',lines=suffix=='jsonl')
    result=read_table(path)
    assert result.text.tolist()==d.text.tolist()
    assert result.label.tolist()==d.label.tolist()
