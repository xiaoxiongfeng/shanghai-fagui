import json
import os

from jina import Document, DocumentArray, Executor, Flow, requests


def load_data(data_fn='toy-data/case_parse_10.json'):
    with open(data_fn, 'r') as f:
        for l in f:
            doc = Document(json.loads(l))
            doc.id = doc.tags['_id']
            doc.tags.pop('_id')
            try:
                doc.text = doc.tags['_source']['title']
                yield doc
            except KeyError as e:
                continue


class SentenceSegmenter(Executor):
    @requests(on=['/index', '/search'])
    def segment(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for s_idx, s in enumerate(doc.text.split('\n')):
                for ss_idx, ss in enumerate(s.split('。')):
                    ss = ss.strip()
                    if not ss:
                        continue
                    _chunk = Document(
                        text=ss,
                        parent_id=doc.id,
                        location=[s_idx, ss_idx])
                    doc.chunks.append(_chunk)


def check_index_resp(resp):
    for d in resp.data.docs:
        doc = Document(d)
        print(f'id: {doc.id}')
        print(f'+- chunks: {len(doc.chunks)}')
        print(f'+- emb: {doc.embedding.shape if doc.embedding else None}')


f = Flow.load_config('flows/index.yml')

with f:
    f.post(on='/index', inputs=load_data, on_done=check_index_resp)