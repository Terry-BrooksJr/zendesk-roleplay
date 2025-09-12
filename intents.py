import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Union
import time

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, util

_CLASSIFER_INSTANCE = None
@dataclass
class ClassificationResult:
    """Stores the results of text classification for intent detection.

    This dataclass contains detected categories, matched patterns, semantic similarity scores, and processing time.

    Attributes:
        categories (Dict[str, bool]): Mapping of category names to detection results.
        matched_patterns (Dict[str, List[str]]): Patterns matched for each category.
        semantic_matches (Dict[str, float]): Semantic similarity scores for each category.
        processing_time_ms (float): Time taken to process the classification in milliseconds.
    """

    categories: Dict[str, bool]
    matched_patterns: Dict[str, List[str]]
    semantic_matches: Dict[str, float]
    processing_time_ms: float

class TextEmbedder:
    _MODEL_REGISTRY = {}

    @classmethod
    def register_model(cls, model_id, model):
        cls._MODEL_REGISTRY[model_id] = model

    def __init__(self, model):
        self.model = model
        self.model_id = id(model)
        TextEmbedder.register_model(self.model_id, model)

    def get_text_embedding(self, text: Union[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(text, torch.Tensor):
            return text
        model = TextEmbedder._MODEL_REGISTRY.get(self.model_id)
        return model.encode([text], convert_to_tensor=True, normalize_embeddings=True)
    
class TextClassifier:
    """Classifies user text into intent categories using regex and semantic similarity.

    This class detects user intents by applying regex rules and semantic similarity, supporting both single and batch classification.

    Attributes:
        similarity_threshold (float): The threshold for semantic similarity matching.
    """
    # Rule definitions
    _rules = {
            "ask_for_diagnostic_data": [
                r"\bhar\b",
                r"network\s+log",
                r"devtools?",
                r"fiddler",
                r"wireshark",
                r"postman",
                r"curl",
                r"http\s+trace",
                r"trace\s+log",
                r"capture\s+log",
                r"screen\s+recording",
                r"screencast",
                r"video\s+recording",
                r"video\s+clip",
                r"screenshot",
                r"screen\s+shot",
                r"network\s+tab",
                r"traceback",
                r"preserve\s+log",
                r"export\s+.*har",
                r"console\s+log", 
            ],
            "ask_for_goal": [
                r"\bgoal\b",
                r"what.*trying.*(do|achieve)",
                r"outcome",
                r"objective",  
                r"purpose",
                r"aim",
                r"intend(ed)?\s+result",
                r"desired\s+state",
                r"expect(ed)?\s+result",
                r"want(ed)?\s+to\s+(do|achieve)",
                r"looking\s+to\s+(do|achieve)",
                r"need(ed)?\s+to\s+(do|achieve)",
                r"wish(ed)?\s+to\s+(do|achieve)",
                r"trying\s+to\s+(do|achieve)",
            ],
            "ask_for_repro": [
                r"repro(duce|duction)?",
                r"\bsteps\b",
                r"how.*reproduce",
                r"recreate.*issue",
                r"replicate.*issue",
                r"cannot\s+reproduce",
                r"unable\s+to\s+reproduce",
                r"works?\s+for\s+me",
                r"did\s+not\s+see",
            ],
            "ask_for_context": [
                r"device",
                r"ipad",
                r"ios",
                r"version",
                r"browser",
                r"safari",
                r"chrome",
                r"firefox",
                r"LTS\sversion",
                r"\bLTS\b",
                r"edge",
                r"android",
                r"app",
                r"When\sdid\s(these|this)\s(start|begin)",
                r"did\s(this|these)\s(start|begin)",
                r"did\s(this|these)\s(change|stop)",
                r"os",
                r"windows",
                r"mac(os)?",
                r"linux",
                r"mobile",
                r"tablet",
                r"desktop",
                r"implementation\s+detail",
                r"webview",
                r"file\s?(type|format)",
                r"operating\s+system",  
            ],
            "propose_fix": [
                r"try",
                r"suggest",
                r"workaround",
                r"\bfix\b",
                r"solution",
                r"change",
                r"switch",
                r"configure",
                r"encode",
                r"recommend",
                r"alternative",
                r"adjust",
                r"patch",
            ],
        }
    _semantic_prompts = {
            "ask_for_diagnostic_data": "please share the HAR or network logs from devtools",
            "ask_for_goal": "what are you trying to achieve overall",
            "ask_for_repro": "please provide steps to reproduce the issue",
            "ask_for_context": "ask device os browser app webview and file-type details",
            "propose_fix": "propose a likely resolution or workaround",
        }
    def __init__(self, similarity_threshold: float = 0.62):
        self._model: Optional[SentenceTransformer] = None
        self.embedder:TextEmbedder = TextEmbedder(self.model)
        self.similarity_threshold: float = similarity_threshold
        self._compiled_patterns: Dict[str, List[re.Pattern]] = self._compile_patterns()
        self._semantic_embeddings:Optional[List[torch.Tensor]] = None
        

        logger.info(f"TextClassifier initialized with {len(type(self)._rules)} categories")

    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compiles regex patterns for each intent category.

        This method prepares and returns a dictionary mapping each category to its list of compiled regex patterns.

        Returns:
            Dict[str, List[re.Pattern]]: Compiled regex patterns for each category.
        """
        return {
           category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
           for category, patterns in type(self)._rules.items()
       }
    @property
    def model(self) -> SentenceTransformer:
        """Loads and returns the sentence transformer model.

        This property initializes the model if it is not already loaded and precomputes semantic prompt embeddings.
        
        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            logger.info("Loading sentence transformer model...")
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._precompute_semantic_embeddings()
            logger.debug("Precomputed semantic prompt embeddings")
            # Pre-compute semantic prompt embeddings

            logger.info("Model loaded successfully")
        return self._model

    def _precompute_semantic_embeddings(self):
        prompts = list(type(self)._semantic_prompts.values())
        if not prompts:
            logger.warning("No semantic prompts defined for pre-computation")
            self._semantic_embeddings = None
            return
        # Encode all prompts at once -> shape [N, D]
        self._semantic_embeddings = self.model.encode(
            prompts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.debug(f"Pre-computed embeddings for {len(prompts)} semantic prompts")

    @lru_cache(maxsize=1000)
    def _cached_text_embedding(self, model_id: int, text: str):
        """Caches and returns the embedding for the given text and model.

        This function retrieves the model by its ID and encodes the input text, caching the result for efficiency.

        Args:
            model_id: The unique identifier for the sentence transformer model.
            text: The input text to encode.

        Returns:
            The tensor embedding of the input text.
        """
        return self.embedder.get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> torch.Tensor:
        return self.embedder.get_text_embedding(text)


    def _apply_regex_rules(
        self, text: str
    ) -> tuple[Dict[str, bool], Dict[str, List[str]]]:
        """Applies regex rules to the input text to detect intent categories.

        This method checks the input text against precompiled regex patterns for each intent category and returns which categories matched and which patterns were found.

        Args:
            text: The input text to classify.

        Returns:
            A tuple containing:
                - A dictionary mapping each category to a boolean indicating if it matched.
                - A dictionary mapping each category to a list of matched regex patterns.
        """
        text_lower = text.lower()
        hits = {category: False for category in type(self)._rules}
        matched_patterns = {category: [] for category in type(self)._rules}

        for category, compiled_patterns in self._compiled_patterns.items():
            for pattern in compiled_patterns:
                if pattern.search(text_lower):
                    hits[category] = True
                    matched_patterns[category].append(pattern.pattern)

        regex_matches = sum(hits.values())
        logger.debug(f"Regex matching found {regex_matches} category matches")
        return hits, matched_patterns

    def _apply_semantic_matching(
        self, text: str, regex_hits: Dict[str, bool]
    ) -> Dict[str, float]:
        """Apply semantic similarity matching for categories not matched by regex."""
        semantic_matches = {}

        # Ensure semantic embeddings exist; if not, skip gracefully
        if self._semantic_embeddings is None:
            for category in type(self)._semantic_prompts.keys():
                semantic_matches[category] = 0.0
            return semantic_matches

        # Obtain normalized text embedding as [1, D]
        text_embedding = self._get_text_embedding(text)

        # Compute cosine similarities -> [1, N] -> squeeze to [N]
        similarities = util.cos_sim(text_embedding, self._semantic_embeddings).squeeze(0)

        semantic_found = 0
        for i, category in enumerate(type(self)._semantic_prompts.keys()):
            similarity = float(similarities[i].item())
            semantic_matches[category] = similarity

            # Only apply semantic match if regex didn't already match and similarity is high enough
            if not regex_hits[category] and similarity > self.similarity_threshold:
                regex_hits[category] = True
                semantic_found += 1
                logger.debug(f"Semantic match for '{category}': {similarity:.3f}")

        if semantic_found > 0:
            logger.debug(f"Semantic matching found {semantic_found} additional matches")

        return semantic_matches

    def detect(self, text: str) -> Dict[str, bool]:
        """
        Simplified interface for backward compatibility.

        Args:
            text: Input text to classify

        Returns:
            Dictionary mapping category names to boolean detection results
        """
        result = self.classify(text)
        return result.categories

    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using both regex patterns and semantic similarity.

        Args:
            text: Input text to classify

        Returns:
            ClassificationResult with detailed classification information
        """

        start_time = time.perf_counter()

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return ClassificationResult(
                categories={k: False for k in type(self)._rules},
                matched_patterns={k: [] for k in type(self)._rules},
                semantic_matches={k: 0.0 for k in type(self)._rules},
                processing_time_ms=0.0,
            )

        logger.debug(
            f"Classifying text: '{text[:100]}{'...' if len(text) > 100 else ''}'"
        )

        # Apply regex rules first
        regex_hits, matched_patterns = self._apply_regex_rules(text)

        # Apply semantic matching for unmatched categories
        semantic_matches = self._apply_semantic_matching(text, regex_hits)

        processing_time = (time.perf_counter() - start_time) * 1000

        total_matches = sum(regex_hits.values())
        logger.info(
            f"Classification complete: {total_matches}/{len(type(self)._rules)} categories matched ({processing_time:.1f}ms)"
        )

        return ClassificationResult(
            categories=regex_hits,
            matched_patterns=matched_patterns,
            semantic_matches=semantic_matches,
            processing_time_ms=processing_time,
        )

    def batch_classify(self, texts: List[str]) -> List[ClassificationResult]:
        """
        Classify multiple texts efficiently.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResult objects
        """
        logger.info(f"Starting batch classification of {len(texts)} texts")
        return [self.classify(text) for text in texts]

    def get_stats(self) -> Dict:
        """Get classifier statistics and configuration."""
        return {
            "categories": list(type(self)._rules.keys()),
            "total_patterns": sum(len(patterns) for patterns in type(self)._rules.values()),
            "similarity_threshold": self.similarity_threshold,
            "model_loaded": self._model is not None,
            "cache_size": (
                self._cached_text_embedding.cache_info() if hasattr(self, "_cached_text_embedding") else None
            ),
        }


def detect(text: str) -> ClassificationResult:
    """Backward compatible function interface."""
    if _CLASSIFER_INSTANCE is None:
        _CLASSIFER_INSTANCE = TextClassifier()
    detected_intentions = _CLASSIFER_INSTANCE.classify(text)


# Example usage
if __name__ == "__main__":
    classifier = TextClassifier()

    # Test cases
    test_texts = [
        "Can you please share the HAR file?",
        "What are you trying to achieve with this?",
        "Please provide steps to reproduce",
        "What browser and device are you using?",
        "Try switching to a different configuration",
        "This is unrelated text about weather",
        "I cannot reproduce the issue on my end",
        "Here is a workaround that might help",
        "Could you share a screenshot of the error?",
        "What is your goal with this task?",
        "Steps to reproduce the bug are as follows...",
        "I suggest you try updating your browser",
        "It's broken",
    ]
    results = classifier.batch_classify(test_texts)
    for text, result in zip(test_texts, results):
        matches = [k for k, v in result.categories.items() if v]
        print(f"Text: {text}")
        print(f"Matches: {matches}")
        print(f"Time: {result.processing_time_ms:.1f}ms")
        print("-" * 50)
    # for text in test_texts:
    #     result = classifier.classify(text)
    #     matches = [k for k, v in result.categories.items() if v]
    #     print(f"Text: {text}")
    #     print(f"Matches: {matches}")
    #     print(f"Time: {result.processing_time_ms:.1f}ms")
    #     print("-" * 50)

    print("\nClassifier stats:")
    print(classifier.get_stats())
