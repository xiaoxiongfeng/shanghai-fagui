from jina import Document, DocumentArray, Executor, requests


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


class Ranker(Executor):
    @requests(on='/search')
    def rank(self, docs: DocumentArray, **kwargs):
        from collections import defaultdict
        for doc in docs:
            result = defaultdict(list)
            for m in doc.matches:
                result[m.parent_id].append(m)
            ranked_matches = []
            for m_id, m_list in result.items():
                sorted_list = sorted(m_list, key=lambda x: x.scores['similarity'].value, reverse=True)
                match = sorted_list[0]
                match.id = m_id
                match.pop('embedding')
                ranked_matches.append(match)
            doc.matches = ranked_matches
            doc.pop('chunks', 'tags')


class RemoveTags(Executor):
    @requests
    def remove(self, docs, **kwargs):
        for d in docs:
            d.pop('tags')
