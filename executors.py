


from typing import List, Dict
from itertools import groupby

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore
import re

from zhon.hanzi import punctuation as chinese_punctuation  # 中文标点符号
import string

# import pkuseg

# seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能

chi_punc = '|'.join([c for c in chinese_punctuation])

# 方法2：在切分后，过滤掉split返回的list中的空字符串
# filter_data()函数的功能是：对于一个由string组成的list [str1, str2, str3, ......]，过滤掉那些空字符串''、特殊字符串'\n'，并返回过滤后的新list
def not_break(sen):
    return sen != '\n' and sen != '\u3000' and sen != '' and not sen.isspace()


def filter_data(ini_data):
    # ini_data是由句子组成的string
    new_data = list(filter(not_break, [data.strip() for data in ini_data]))
    return new_data


# f = open('toy-data/key_dics.txt', 'r')
# key_list = [s.strip() for s in f.readlines()]


class IndexSentenceSegmenter(Executor):
    @requests(on='/index')
    def segment(self, docs: DocumentArray, **kwargs):
        for doc in docs:

            try:
                # content chunk
                #for s_idx, s in enumerate(doc.text.split('\n')):
                 #   s = s.strip()
                  #  if not s:
                   #     continue
                    #if len(s) > 65:
                     #   s = s[:64]
                  #  _chunk = Document(
                   #         text=s,
                    #        parent_id=doc.id,
                     #       location=[s_idx],
                      #      modality='content',
                       # )
                    #doc.chunks.append(_chunk)

                # paras chunk
                if doc.tags['_source']['paras']:
                    for tag in doc.tags['_source']['paras']:
                        try:
                            if not tag['content']:
                                continue
                            if len(tag['content']) > 65:
                                tag['content'] = tag['content'][:64]
                            _chunk = Document(
                                text=tag['content'], parent_id=doc.id, modality='paras'
                            )
                            doc.chunks.append(_chunk)
                        except KeyError as e:
                            continue

                # title chunk
                for s_idx, s in enumerate(doc.tags['_source']['title'].split('\n')):
                    s_list = filter_data(re.split(r'' + ("[" + chi_punc + "]"), s))
                    for ss_idx, ss in enumerate(s_list):
                        ss = ss.strip()
                        if not ss:
                            continue
                        if len(ss) > 65:
                            ss = ss[:64]
                        _chunk = Document(
                            text=ss,
                            parent_id=doc.id,
                            location=[s_idx, ss_idx],
                            modality='title',
                        )
                        doc.chunks.append(_chunk)

                # causes chunk
                if doc.tags['_source']['causes']:
                    print('causes')
                    for cause in doc.tags['_source']['causes']:
                        if not cause:
                            continue
                        _chunk = Document(
                            text=cause, parent_id=doc.id, modality='causes'
                        )
                        doc.chunks.append(_chunk)
                # court chunk
                if doc.tags['_source']['court']:
                    print('court')
                    court_ = doc.tags['_source']['court']
                    print(court_)
                    
                    #if not court_:
                     #   continue
                    _chunk = Document(
                            text=court_, parent_id=doc.id, modality='court'
                            )
                    doc.chunks.append(_chunk)
                    print('add court_')


            except KeyError as e:
                continue

        return DocumentArray([d for d in docs if d.chunks])


class QuerySentenceSegmenter(Executor):
    @requests(on='/search')
    def segment(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for s_idx, s in enumerate(doc.text.split('\n')):
                #s_list = filter_data(re.split(r'' + ("[" + chi_punc + "]"), s))
                #for ss_idx, ss in enumerate(s_list):
                    s = s.strip()
                    if not s:
                        continue
                    if len(s) > 65:
                        s = s[:64]
                    _chunk = Document(
                        text=s + '.....................', parent_id=doc.id, location=[s_idx]
                    )
                    doc.chunks.append(_chunk)
        return DocumentArray([d for d in docs if d.chunks])


class AggregateRanker(Executor):
    def __init__(
        self,
        default_top_k: int = 5,
        default_traversal_paths: List[str] = ['r'],
        metric: str = 'cosine',
        is_distance: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.default_top_k = default_top_k
        self.default_traversal_paths = default_traversal_paths
        self.metric = metric
        if is_distance:
            self.distance_mult = 1
        else:
            self.distance_mult = -1

    @requests(on='/search')
    def rank(self, docs: DocumentArray, parameters: Dict = None, **kwargs):

        top_k = int(parameters.get('top_k', self.default_top_k))

        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )

        for doc in docs.traverse_flat(traversal_paths):
            matches_of_chunks = []
            for chunk in doc.chunks:
                matches_of_chunks.extend(chunk.matches)

            groups = groupby(
                sorted(matches_of_chunks, key=lambda d: d.parent_id),
                lambda d: d.parent_id,
            )
            for key, group in groups:
                chunk_match_list = list(group)

                for m in chunk_match_list:
                    if m.modality == 'content':
                        m.scores[self.metric].value *= 1
                    if m.modality == 'causes':
                        m.scores[self.metric].value *= 1
                    if m.modality == 'paras':
                        print('paras')
                        m.scores[self.metric].value *= 1
                    if m.modality == 'title':
                        m.scores[self.metric].value *= 1
                    if m.modality == 'court':
                        print('court')
                        m.scores[self.metric].value *= 1

                chunk_match_list.sort(
                    key=lambda m: self.distance_mult * m.scores[self.metric].value
                )

                operands = []
                for m in chunk_match_list:
                    o = NamedScore(
                            op_name=f'{m.location[0]}' if m.location else '',
                            value=m.scores[self.metric].value,
                            ref_id=m.parent_id,
                            description=f'{m.text} ',
                        )
                    operands.append(o)

                
                match = chunk_match_list[0]
                match.id = chunk_match_list[0].parent_id
                match.scores[self.metric].set_attrs(operands=operands)
                doc.matches.append(match)

            doc.matches.sort(
                key=lambda d: self.distance_mult * d.scores[self.metric].value
            )
            doc.matches = doc.matches[:top_k]


            # trim `chunks` and `tags`
            doc.pop('chunks', 'tags')


class RemoveTags(Executor):
    @requests
    def remove(self, docs, **kwargs):
        for d in docs:
            # continue
            d.pop('tags')


class DebugExecutor(Executor):
    def __init__(self, metric: str = 'l2', *args, **kwargs):
        super().__init__(**kwargs)
        from jina.logging.logger import JinaLogger

        self.logger = JinaLogger(self.__class__.__name__)
        self.metric = metric

    @requests
    def debug(self, docs, **kwargs):
        for i, d in enumerate(docs):
            print(f'[{i}] chunks: {len(d.chunks)}')
            for j, c in enumerate(d.chunks):
                if j >= 3:
                    print(f'\t ...')
                    assert c.embedding.shape == (768,)
                else:
                    print(f'\t emb shape: {c.embedding.shape}')
                print(
                    f'modality: {c.matches[0].parent_id} - {c.matches[0].modality} - {c.matches[0].scores[self.metric].value}'
                )
                print(
                    f'[{i} - {j}]: {len(c.matches)} - [{" ".join([str(m.scores[self.metric].value) for m in c.matches])}]'
                )
