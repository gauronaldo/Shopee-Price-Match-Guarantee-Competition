"""Training-only character TF-IDF features for noisy marketplace titles."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

from shopee_match.evaluation.protocol import CorpusItem, Ranking, ScoredCandidate

SparseVector = dict[int, float]


def normalize_title(value: str) -> str:
    """Normalize presentation noise while preserving identity-critical letters and digits."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def char_word_ngrams(value: str, minimum: int, maximum: int) -> Counter[str]:
    """Generate character n-grams within padded words, matching char_wb semantics."""
    counts: Counter[str] = Counter()
    for word in normalize_title(value).split():
        padded = f" {word} "
        for size in range(minimum, maximum + 1):
            counts.update(padded[index : index + size] for index in range(len(padded) - size + 1))
    return counts


@dataclass(frozen=True)
class CharTfidfModel:
    vocabulary: dict[str, int]
    idf: tuple[float, ...]
    ngram_range: tuple[int, int]

    @classmethod
    def fit(
        cls,
        items: tuple[CorpusItem, ...],
        ngram_range: tuple[int, int],
        max_features: int,
    ) -> CharTfidfModel:
        """Fit vocabulary and IDF from the training split only."""
        document_frequency: Counter[str] = Counter()
        for item in items:
            document_frequency.update(char_word_ngrams(item.title, *ngram_range).keys())
        selected = sorted(document_frequency, key=lambda gram: (-document_frequency[gram], gram))[
            :max_features
        ]
        vocabulary = {gram: index for index, gram in enumerate(selected)}
        document_count = len(items)
        idf = tuple(
            math.log((1 + document_count) / (1 + document_frequency[gram])) + 1 for gram in selected
        )
        return cls(vocabulary, idf, ngram_range)

    def transform_one(self, title: str) -> SparseVector:
        counts = char_word_ngrams(title, *self.ngram_range)
        weighted = {
            index: (1 + math.log(count)) * self.idf[index]
            for gram, count in counts.items()
            if (index := self.vocabulary.get(gram)) is not None
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        return {index: value / norm for index, value in weighted.items()} if norm else {}

    def similarity(self, left: str, right: str) -> float:
        """Compute cosine similarity in the train-fitted sparse feature space."""
        left_vector = self.transform_one(left)
        right_vector = self.transform_one(right)
        if len(left_vector) > len(right_vector):
            left_vector, right_vector = right_vector, left_vector
        return sum(value * right_vector.get(index, 0.0) for index, value in left_vector.items())

    def rank(self, items: tuple[CorpusItem, ...], top_k: int) -> Ranking:
        """Retrieve against the full supplied split with a sparse inverted index."""
        return self.rank_queries(items, items, top_k)

    def rank_queries(
        self,
        items: tuple[CorpusItem, ...],
        queries: tuple[CorpusItem, ...],
        top_k: int,
    ) -> Ranking:
        """Retrieve a bounded query subset against a full label-blind corpus."""
        vectors = {item.posting_id: self.transform_one(item.title) for item in items}
        postings: dict[int, list[tuple[str, float]]] = defaultdict(list)
        for posting_id, vector in vectors.items():
            for feature, weight in vector.items():
                postings[feature].append((posting_id, weight))
        all_ids = sorted(vectors)
        ranking: Ranking = {}
        for query in sorted(queries, key=lambda item: item.posting_id):
            query_id = query.posting_id
            query_vector = self.transform_one(query.title)
            scores: dict[str, float] = defaultdict(float)
            for feature, query_weight in query_vector.items():
                for candidate_id, candidate_weight in postings[feature]:
                    if candidate_id != query_id:
                        scores[candidate_id] += query_weight * candidate_weight
            ordered = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]
            selected = {posting_id for posting_id, _score in ordered}
            if len(ordered) < min(top_k, len(all_ids) - 1):
                ordered.extend(
                    (posting_id, 0.0)
                    for posting_id in all_ids
                    if posting_id != query_id and posting_id not in selected
                )
            ranking[query_id] = [
                ScoredCandidate(posting_id, float(score)) for posting_id, score in ordered[:top_k]
            ]
        return ranking
