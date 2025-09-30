import re

import pytest
import torch

from ..core.intents import TextClassifier


# Dummy ClassificationResult for type checking (since it's not in the provided code)
class ClassificationResult:
    def __init__(
        self, categories, matched_patterns, semantic_matches, processing_time_ms
    ):
        self.categories = categories
        self.matched_patterns = matched_patterns
        self.semantic_matches = semantic_matches
        self.processing_time_ms = processing_time_ms


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """Patch SentenceTransformer and util.cos_sim to avoid heavy model loading and torch ops."""

    class DummyModel:
        def encode(
            self,
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ):
            # Return a tensor of shape [N, 384] for N prompts/texts
            if isinstance(texts, str):
                return torch.ones(1, 384)
            return torch.ones(len(texts), 384)

    monkeypatch.setattr("intents.SentenceTransformer", lambda *a, **kw: DummyModel())
    monkeypatch.setattr("intents.util.cos_sim", lambda a, b: torch.ones(1, b.shape[0]))

    # Patch TextEmbedder to avoid dependency
    class DummyEmbedder:
        def get_text_embedding(self, text):
            return torch.ones(1, 384)

    monkeypatch.setattr("intents.TextEmbedder", lambda model: DummyEmbedder())


@pytest.mark.parametrize(
    "text,expected_category,expected_match",
    [
        ("Please share the HAR file", "ask_for_diagnostic_data", True),
        ("What is your goal?", "ask_for_goal", True),
        ("Can you provide steps to reproduce?", "ask_for_repro", True),
        ("Which device and OS are you using?", "ask_for_context", True),
        ("Try this workaround", "propose_fix", True),
        ("This is unrelated to any category", "ask_for_diagnostic_data", False),
        ("How were you trying to implement this?", "ask_for_goal", True),
    ],
    ids=[
        "har-diagnostic-data",
        "goal-detection",
        "repro-steps",
        "context-device-os",
        "propose-fix",
        "no-match",
        "goal-detection-alt",
    ],
)
def test_apply_regex_rules(text, expected_category, expected_match):
    # Arrange

    classifier = TextClassifier()

    # Act

    hits, matched_patterns = classifier._apply_regex_rules(text)

    # Assert

    assert hits[expected_category] is expected_match
    if expected_match:
        assert len(matched_patterns[expected_category]) > 0
    else:
        assert matched_patterns[expected_category] == []


@pytest.mark.parametrize(
    "text,regex_hits,expected_semantic,expected_threshold",
    [
        (
            "Please share the HAR file",
            {
                "ask_for_diagnostic_data": False,
                "ask_for_goal": False,
                "ask_for_repro": False,
                "ask_for_context": False,
                "propose_fix": False,
            },
            True,
            0.5,
        ),
        (
            "Unrelated text",
            {
                "ask_for_diagnostic_data": False,
                "ask_for_goal": False,
                "ask_for_repro": False,
                "ask_for_context": False,
                "propose_fix": False,
            },
            True,
            0.5,
        ),
        (
            "",
            {
                "ask_for_diagnostic_data": False,
                "ask_for_goal": False,
                "ask_for_repro": False,
                "ask_for_context": False,
                "propose_fix": False,
            },
            True,
            0.5,
        ),
    ],
    ids=[
        "semantic-match-har",
        "semantic-match-unrelated",
        "semantic-match-empty",
    ],
)
def test_apply_semantic_matching(
    text, regex_hits, expected_semantic, expected_threshold
):
    # Arrange

    classifier = TextClassifier(similarity_threshold=expected_threshold)
    # Patch _semantic_embeddings to a tensor of shape [5, 384]
    classifier._semantic_embeddings = torch.ones(5, 384)

    # Act

    result = classifier._apply_semantic_matching(text, regex_hits.copy())

    # Assert

    assert isinstance(result, dict)
    assert set(result.keys()) == set(classifier._semantic_prompts.keys())
    for v in result.values():
        assert isinstance(v, float)


def test_apply_semantic_matching_no_embeddings():
    # Arrange

    classifier = TextClassifier()
    classifier._semantic_embeddings = None
    regex_hits = {k: False for k in classifier._semantic_prompts.keys()}

    # Act

    result = classifier._apply_semantic_matching("any text", regex_hits.copy())

    # Assert

    assert all(v == 0.0 for v in result.values())


@pytest.mark.parametrize(
    "text,expected_categories",
    [
        ("Please share the HAR file", {"ask_for_diagnostic_data": True}),
        ("What is your goal?", {"ask_for_goal": True}),
        ("Can you provide steps to reproduce?", {"ask_for_repro": True}),
        ("Which device and OS are you using?", {"ask_for_context": True}),
        ("Try this workaround", {"propose_fix": True}),
        ("This is unrelated to any category", {}),
        ("", {}),
        ("   ", {}),
    ],
    ids=[
        "detect-har",
        "detect-goal",
        "detect-repro",
        "detect-context",
        "detect-fix",
        "detect-none",
        "detect-empty",
        "detect-whitespace",
    ],
)
def test_detect(text, expected_categories):
    # Arrange

    classifier = TextClassifier()

    # Act

    result = classifier.detect(text)

    # Assert

    for category, expected in expected_categories.items():
        assert result[category] is expected
    # All other categories should be False
    for category in classifier._rules:
        if category not in expected_categories:
            assert result[category] is False


@pytest.mark.parametrize(
    "text,expected_any_match",
    [
        ("Please share the HAR file", True),
        ("What is your goal?", True),
        ("Can you provide steps to reproduce?", True),
        ("Which device and OS are you using?", True),
        ("Try this workaround", True),
        ("This is unrelated to any category", False),
        ("", False),
        ("   ", False),
    ],
    ids=[
        "classify-har",
        "classify-goal",
        "classify-repro",
        "classify-context",
        "classify-fix",
        "classify-none",
        "classify-empty",
        "classify-whitespace",
    ],
)
def test_classify(text, expected_any_match):
    # Arrange

    classifier = TextClassifier()

    # Act

    result = classifier.classify(text)

    # Assert

    assert hasattr(result, "categories")
    assert hasattr(result, "matched_patterns")
    assert hasattr(result, "semantic_matches")
    assert hasattr(result, "processing_time_ms")
    assert isinstance(result.categories, dict)
    assert isinstance(result.matched_patterns, dict)
    assert isinstance(result.semantic_matches, dict)
    assert isinstance(result.processing_time_ms, float)
    assert any(result.categories.values()) is expected_any_match


def test_classify_empty_and_whitespace():
    # Arrange

    classifier = TextClassifier()

    # Act

    result_empty = classifier.classify("")
    result_ws = classifier.classify("   ")

    # Assert

    assert all(v is False for v in result_empty.categories.values())
    assert all(v is False for v in result_ws.categories.values())
    assert all(isinstance(v, list) for v in result_empty.matched_patterns.values())
    assert all(isinstance(v, list) for v in result_ws.matched_patterns.values())
    assert all(isinstance(v, float) for v in result_empty.semantic_matches.values())
    assert all(isinstance(v, float) for v in result_ws.semantic_matches.values())
    assert result_empty.processing_time_ms == 0.0
    assert result_ws.processing_time_ms == 0.0


def test_batch_classify():
    """Tests that batch_classify returns a list of ClassificationResult objects for each input text.

    This function verifies that the batch classification method processes multiple texts and returns the expected result structure for each.
    """
    # Arrange

    classifier = TextClassifier()
    texts = [
        "Please share the HAR file",
        "What is your goal?",
        "Unrelated text",
        "",
    ]

    # Act

    results = classifier.batch_classify(texts)

    # Assert

    assert isinstance(results, list)
    assert len(results) == len(texts)
    for res in results:
        assert hasattr(res, "categories")
        assert hasattr(res, "matched_patterns")
        assert hasattr(res, "semantic_matches")
        assert hasattr(res, "processing_time_ms")


def test_get_stats():
    # Arrange

    classifier = TextClassifier()

    # Act

    stats = classifier.get_stats()

    # Assert

    assert isinstance(stats, dict)
    assert "categories" in stats
    assert "total_patterns" in stats
    assert "similarity_threshold" in stats
    assert "model_loaded" in stats
    assert "cache_size" in stats
    assert stats["categories"] == list(classifier._rules.keys())
    assert stats["total_patterns"] == sum(len(v) for v in classifier._rules.values())
    assert stats["similarity_threshold"] == classifier.similarity_threshold
    assert stats["model_loaded"] is True


def test_compile_patterns():
    # Arrange

    classifier = TextClassifier()

    # Act

    compiled = classifier._compile_patterns()

    # Assert

    assert isinstance(compiled, dict)
    for category, patterns in compiled.items():
        assert isinstance(patterns, list)
        for pat in patterns:
            assert isinstance(pat, re.Pattern)


def test_model_property_loads_and_caches(monkeypatch):
    # Arrange

    classifier = TextClassifier()
    # Remove model to force reload
    classifier._model = None
    called = {}

    class DummyModel:
        def encode(self, *a, **kw):
            called["encode"] = True
            return torch.ones(5, 384)

    monkeypatch.setattr("intents.SentenceTransformer", lambda *a, **kw: DummyModel())

    # Act

    model = classifier.model

    # Assert

    assert model is not None
    assert hasattr(classifier, "_model")
    assert classifier._model is model


def test_cached_text_embedding_cache(monkeypatch):
    # Arrange

    classifier = TextClassifier()
    # Patch embedder to count calls
    calls = []

    class DummyEmbedder:
        def get_text_embedding(self, text):
            calls.append(text)
            return torch.ones(1, 384)

    classifier.embedder = DummyEmbedder()

    # Act

    emb1 = classifier._cached_text_embedding(1, "foo")
    emb2 = classifier._cached_text_embedding(1, "foo")
    classifier._cached_text_embedding(1, "bar")

    # Assert

    assert torch.equal(emb1, emb2)
    assert torch.equal(emb1, torch.ones(1, 384))
    assert len(calls) == 2  # Only two unique calls


def test_get_text_embedding_delegates():
    # Arrange

    classifier = TextClassifier()
    called = {}

    class DummyEmbedder:
        def get_text_embedding(self, text):
            called["text"] = text
            return torch.ones(1, 384)

    classifier.embedder = DummyEmbedder()

    # Act

    emb = classifier._get_text_embedding("hello")

    # Assert

    assert torch.equal(emb, torch.ones(1, 384))
    assert called["text"] == "hello"
