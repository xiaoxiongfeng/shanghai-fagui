import json
import os

from jina import Document, DocumentArray, Executor, Flow, requests

os.environ['JINA_DUMP_PATH'] = './workspace/dump'


def load_data(data_fn='toy-data/case_parse_10.json'):
    counter = 0
    with open(data_fn, 'r') as f:
        for l in f:
            doc = Document(json.loads(l))
            try:
                doc.text = doc.tags['_source']['title']
                doc.id = doc.tags['_id']
                counter += 1
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
        return DocumentArray([d for d in docs if d.chunks])


class ReplaceMatchId(Executor):
    @requests(on='/search')
    def replace(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for m in doc.matches:
                print(f'vector match: {m}')
                m.id = m.parent_id


def check_index_resp(resp):
    for d in resp.data.docs:
        doc = Document(d)
        print(f'id: {doc.id}')
        print(f'+- chunks: {len(doc.chunks)}')
        print(f'+- emb: {doc.embedding.shape if doc.embedding is not None else None}')


f_index = Flow.load_config('flows/index.yml')
f_query = Flow.load_config('flows/query.yml')

with f_index:
    f_index.post(on='/index', inputs=load_data, on_done=check_index_resp)
    f_index.post(
        on='/dump', target_peapod='indexer', parameters={'dump_path': os.environ.get('JINA_DUMP_PATH'), 'shards': 1})

with f_query:
    results = f_query.post(
        on='/search', inputs=load_data, parameters={'top_k': 3}, return_results=True)
    print(f'result: {results[0].docs[0]}')