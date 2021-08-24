


from typing import List, Dict
from itertools import groupby

from jina import Document, DocumentArray, Executor, requests
from jina.types.score import NamedScore
import re

import pkuseg

from zhon.hanzi import punctuation as chinese_punctuation  # 中文标点符号
chi_punc = '|'.join([c for c in chinese_punctuation])

# 方法2：在切分后，过滤掉split返回的list中的空字符串
# filter_data()函数的功能是：对于一个由string组成的list [str1, str2, str3, ......]，过滤掉那些空字符串''、特殊字符串'\n'，并返回过滤后的新list
def not_break(sen):
    return sen != '\n' and sen != '\u3000' and sen != '' and not sen.isspace()


def filter_data(ini_data):
    # ini_data是由句子组成的string
    new_data = list(filter(not_break, [data.strip() for data in ini_data]))
    return new_data


class IndexSentenceSegmenter(Executor):
    @requests(on='/index')
    def segment(self, docs: DocumentArray, **kwargs):
        if not docs:
            return
        for doc in docs:
            if '_source' not in doc.tags:
                continue
            # paras chunk, we need to filter the paras based on types
            paras = doc.tags['_source'].get('paras', None)
            if paras:
                for para in paras:
                    if not para['content']:
                        continue
                    if para['tag'] in ['书记员', '审判人员', 'judges', 'basics', 'party_info', '附', 'defendant_plea', 'plaintiff_claims', '审判法官', 'court_consider', '当事人信息', '基础信息', '本院认为', '裁判结果', '调解结果']:
                        continue
                    for p_idx, p in enumerate(para['content'].split('\n')):
                        s_list = filter_data(re.split(r'' + ('[。]'), p))
                        for s_idx, s in enumerate(s_list):
                            s = s.strip()
                            _chunk = Document(
                                text=s[-64:], parent_id=doc.id, modality='paras', location=[p_idx, s_idx])
                            _chunk.tags['para_tag'] = para['tag']
                            doc.chunks.append(_chunk)

            # title chunk
            title = doc.tags['_source'].get('title', None)
            if title:
                title_list = title.split('\n')
                title_text = ''.join(title_list)
                _chunk = Document(
                    text=title_text[-64:],
                    parent_id=doc.id,
                    modality='title',
                )
                for s_idx, s in enumerate(title_list):
                    s_list = filter_data(re.split(r'' + ('[' + chi_punc + ' ' + ']'), s))
                    if len(s_list) <= 1:
                        continue
                    for s_idx, s in enumerate(s_list):
                        s = s.strip()
                        if not s:
                            continue
                        if s == _chunk.text:
                            continue
                        _chunk_chunk = Document(
                            text=s[-64:],
                            parent_id=_chunk.id,
                            location=[s_idx, s_idx],
                            modality='title_subsentence',
                        )
                        _chunk.chunks.append(_chunk_chunk)
                doc.chunks.append(_chunk)

            # causes chunk
            causes = doc.tags['_source'].get('causes', None)
            if causes:
                for cause in causes:
                    if cause in ['其他', ]:
                       continue
                    if not cause:
                        continue
                    _chunk = Document(
                        text=cause, parent_id=doc.id, modality='causes'
                    )
                    doc.chunks.append(_chunk)

            # docType chunk
            doc_type = doc.tags['_source'].get('docType', None)
            if doc_type:
                _chunk = Document(
                    text=doc_type, parent_id=doc.id, modality='docType'
                )
                doc.chunks.append(_chunk)

            # topCause chunk
            top_cause = doc.tags['_source'].get('topCause', None)
            if top_cause:
                _chunk = Document(
                    text=top_cause, parent_id=doc.id, modality='topCause'
                )
                doc.chunks.append(_chunk)

            # topCause+docType chunk
            if top_cause and doc_type:
                _chunk = Document(
                    text=top_cause+doc_type, parent_id=doc.id, modality='topCause_docType'
                )
                doc.chunks.append(_chunk)

            # trialRound chunk
            trial_round = doc.tags['_source'].get('trialRound', None)
            if trial_round in ['其他', ]:
                trial_round = None
            if trial_round:
                _chunk = Document(
                    text=trial_round, parent_id=doc.id, modality='trialRound'
                )
                doc.chunks.append(_chunk)

            # trialRound+docType
            if trial_round and doc_type:
                _chunk = Document(
                    text=trial_round+doc_type,
                    parent_id=doc.id,
                    modality='trialRound_docType'
                )
                doc.chunks.append(_chunk)

            # topCause+docType chunk
            if trial_round and top_cause and doc_type:
                _chunk = Document(
                    text=trial_round + top_cause + doc_type,
                    parent_id=doc.id,
                    modality='trialRound_topCause_docType'
                )
                doc.chunks.append(_chunk)

            # court chunk
            if doc.tags['_source']['court']:
                court_ = doc.tags['_source']['court']
                _chunk = Document(
                        text=court_, parent_id=doc.id, modality='court'
                        )
                doc.chunks.append(_chunk)

        return DocumentArray([d for d in docs if d.chunks])


class QuerySentenceSegmenter(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seg = pkuseg.pkuseg(model_name='news', postag=False)

    @requests(on='/search')
    def segment(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            for s_idx, s in enumerate(doc.text.split(' ')):
                s = s.strip()
                if not s:
                    continue
                _chunk = Document(
                    text=s[:64], parent_id=doc.id, location=[s_idx]
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
                        m.scores[self.metric].value *= 1
                    if m.modality == 'title':
                        m.scores[self.metric].value *= 1
                    if m.modality == 'court':
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
