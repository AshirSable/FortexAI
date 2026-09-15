import re
import json
import nltk
from dataclasses import dataclass
from typing import List, Optional
import warnings
from pathlib import Path
from nltk.tokenize import sent_tokenize
import dataclasses
from concurrent.futures import ProcessPoolExecutor
import os

nltk.download("punkt_tab")
nltk.download("punkt")


@dataclass
class EMPIRICAL_ATTACK_TERMS:
    exfilteration: List[str]
    subversion: List[str]
    exploitation: List[str]
    instruction_override: List[str]
    guise_framing: List[str]

    def to_dict(self):
        return self.__dict__

    @classmethod
    def default(cls):
        return cls(
            exfilteration=["reveal", "disclose", "divulge", "expose", "secret"],
            subversion=["trick", "deceive", "coerce", "manipulate", "force the system"],
            exploitation=[
                "exploit",
                "vulnerabilit",
                "inject",
                "override",
                "admin",
                "config",
            ],
            instruction_override=[
                "ignore previous",
                "ignore all previous",
                "ignore the previous",
            ],
            guise_framing=[
                "hidden within",
                "under the guise",
                "not supposed to share",
                "programmed to",
            ],
        )

    def __getitem__(self, name: str, /) -> List[str]:
        return self.__dict__[name]

    @classmethod
    def from_dict(cls, data: dict):
        incoming_data_keys = set(data.keys())
        valid_field_names = {f.name for f in dataclasses.fields(cls)}
        unknown_keys = incoming_data_keys - valid_field_names

        if unknown_keys:
            warnings.warn(
                f"Found Unknown terms: {unknown_keys}, they wont be added here",
                UserWarning,
            )

        valid_keys = incoming_data_keys & valid_field_names

        keys_not_specified = valid_field_names - valid_keys

        defaults = cls.default()
        final_value = {vk: data[vk] for vk in valid_keys}

        for kv in keys_not_specified:
            final_value[kv] = getattr(defaults, kv)

        return cls(**final_value)


def structural_fn_worker(args: tuple) -> List[float]:
    text, term_list, code_pattern_str = args

    code_pattern = re.compile(code_pattern_str, re.I)

    text_lower = text.lower()

    n_words = len(text.split())
    n_characters = len(text)
    n_sentences = max(len(sent_tokenize(text)), 1)

    exfil_terms, subversion_terms = term_list[0], term_list[1]

    has_code = bool(code_pattern.search(text))
    has_intent_vocab = any(
        term in text.lower()
        for terms in [exfil_terms, subversion_terms]
        for term in terms
    )
    code_inject_intent = int(has_code and has_intent_vocab)

    emp_counts = [sum(text_lower.count(term) for term in terms) for terms in term_list]

    return [
        n_words,
        n_characters / max(n_words, 1),
        n_words / n_sentences,
        code_inject_intent,
    ] + emp_counts


class StructuralExtractor:
    def __init__(
        self,
        _empirical_attack_terms: Optional[dict[str, List[str]]] = None,
        _code_injection_pattern: Optional[str] = None,
    ):
        self.empirical_attack_terms = (
            EMPIRICAL_ATTACK_TERMS.from_dict(_empirical_attack_terms)
            if _empirical_attack_terms
            else EMPIRICAL_ATTACK_TERMS.default()
        )
        code_injection = (
            _code_injection_pattern
            or r"(exec\(|execute\(|hacksystem\(|function\(\)|console\.log\(|for\(let|=>\s*\{|lambda\s+\w+)"
        )
        self.code_injection_pattern = re.compile(code_injection, re.I)

    def calculate_n_words(self, text: str) -> int:
        return len(text.split())

    def calculate_n_characters(self, text: str) -> int:
        return len(text)

    def calculate_n_sentences(self, text: str) -> int:
        return max(len(sent_tokenize(text)), 1)

    @property
    def _term_lists(self) -> List[List[str]]:
        return [
            self.empirical_attack_terms.exfilteration,
            self.empirical_attack_terms.subversion,
            self.empirical_attack_terms.exploitation,
            self.empirical_attack_terms.instruction_override,
            self.empirical_attack_terms.guise_framing,
        ]

    def calculate_empirical_term_counts(self, text: str) -> List[int]:
        text_lower = text.lower()

        return [
            sum(text_lower.count(term) for term in terms) for terms in self._term_lists
        ]

    def calculate_code_injection_with_intent(self, text: str) -> int:
        has_code = bool(self.code_injection_pattern.search(text))

        has_intent_vocab = any(
            term in text.lower()
            for terms in [
                self.empirical_attack_terms.exfilteration,
                self.empirical_attack_terms.subversion,
            ]
            for term in terms
        )

        return int(has_code and has_intent_vocab)

    def structural_fn(self, text: str) -> List[float]:
        n_words = self.calculate_n_words(text)
        n_sentences = self.calculate_n_sentences(text)
        n_characters = self.calculate_n_characters(text)
        code_inject_intent = self.calculate_code_injection_with_intent(text)
        n_emp_tc = self.calculate_empirical_term_counts(text)

        final_list = [
            n_words,
            n_characters / n_words,
            n_words / n_sentences,
            code_inject_intent,
        ]

        final_list.extend(n_emp_tc)

        return final_list

    def structural_fn_batch(self, texts: List[str]) -> List[List[float]]:
        code_pattern_str = self.code_injection_pattern.pattern
        term_list = self._term_lists
        args = [(text, term_list, code_pattern_str) for text in texts]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            return list(executor.map(structural_fn_worker, args, chunksize=500))

    @classmethod
    def from_json(cls, file_path: Path | str):
        with open(file_path, "r") as f:
            j = json.load(f)

        return cls(
            _empirical_attack_terms=j.get("empirical_attack_terms"),
            _code_injection_pattern=j.get("code_injection_pattern"),
        )

    def save_json(self, output_path: Path | str):
        with open(output_path, "w") as f:
            json.dump(
                {
                    "empirical_attack_terms": self.empirical_attack_terms.to_dict(),
                    "code_injection_pattern": self.code_injection_pattern.pattern,
                },
                f,
            )
