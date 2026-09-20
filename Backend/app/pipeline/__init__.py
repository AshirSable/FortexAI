# THOUGHT: I need to cover 4 main phases, give them an order, and also keep an log of each phase initializiation
# therefore i need to keep a log at each given session, which can be traced back
# at the same time i need to keep a small cache of outputs given by each phase, and keep the next phase ready
# what else i need to make sure is that we know when it is an attack at some phase i stop the pipeline

# THOUGHT: Lets make an enum which would indicate attack and benign first
# another enum that would indicate the phase
# a struct that would keep this enum with confidence and a record of the phase
# another struct which would keep the previous struct with the input data...
import time
from abc import ABC, abstractmethod
from dataclasses import replace

from app.type_store import (
    Ok,
    Phase,
    PhaseInput,
    Result,
    SuccessForReview,
    SuccessReturn,
    Verdict,
)
from app.type_store._error import PhaseError


class Pipeline:
    """
    Runs the detection stages one at a time, in the order given, and stops at
    the first stage whose verdict is NOT `undetermined`.

    This is the paper's D(x) = d_k(x): the first stage in the cascade that is
    willing to make a call wins, and every stage before it just passed.

    We take the list of phases in the constructor instead of building it
    ourselves, because each stage module (semantic_search.py, autoencoder.py,
    ...) imports PipelinePhase from this file - if this file also imported
    those stage modules, we'd get a circular import. Whoever wires the
    pipeline together (main.py) builds the phase list.
    """

    def __init__(self, phases: list["PipelinePhase"]):
        if not phases:
            raise ValueError("Pipeline needs at least one phase to run")
        self.phases = phases

    def run(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        current_input = input
        last_result = None

        # measured from the moment the input arrives here to the moment a
        # verdict is produced - this is "detection time" for the caller
        # (main.py), not raw HTTP time. Whichever stage resolves the verdict,
        # the latency stamped on the result is the sum of every stage that
        # ran before it plus its own time - e.g. if bert resolves it, that's
        # semantic_search + autoencoder + bert added together, since they ran
        # one after another and the clock never stopped in between.
        start_time = time.perf_counter()

        for phase in self.phases:
            result = phase.verdict(current_input)

            if result.is_err():
                # a stage broke - stop here and hand the error back, instead
                # of guessing what the broken stage would have said.
                return result

            success = result.unwrap()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            success = replace(success, latency_ms=elapsed_ms)
            result = Ok(success)
            last_result = result

            # TODO: The data should pass if auto-encoder detects it as an attack, as then it would be passed to the further stages
            # The pipeline should stop if auto-encoder calls it normal

            # TODO: If the models ahead of the Auto-encoder classify a text as normal, it must be saved in the allowed prompts
            # we need to do an analysis comparing latency vs storage for deciding in which phase we should save in the allowed prompts database
            if success.verdict != Verdict.undetermined:
                # this stage made a call - the cascade stops here.
                # BUG: The cascade stopping criteria as not as discussed
                # It should stop when malicious in stage "semantic search", "bert", "llm judge"
                # it should stop when benign in auto encoder
                return result

            # this stage had no opinion - remember its result (now carrying
            # its own cumulative latency) and move on to the next stage,
            # carrying the prior results forward so later stages can see
            # what already ran.
            prior_results = dict(current_input._prior_results)
            prior_results[phase.phase] = success
            current_input = replace(current_input, _prior_results=prior_results)

        # every single stage came back undetermined. This shouldn't happen
        # once llm_judge (the last stage) is wired to always decide, but we
        # handle it instead of crashing: just return the last undetermined
        # result we got (phases is never empty, so we always have one).
        return last_result


class PipelinePhase(ABC):
    def __init__(self, phase: Phase):
        self.phase: Phase = phase

    @property
    def name(self):
        return self.phase.name

    def __repr__(self) -> str:
        return f"<{self.name} pipeline-phase>"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def verdict(self, input: PhaseInput) -> Result[SuccessReturn, PhaseError]:
        raise NotImplementedError("method not implemented yet")

    @abstractmethod
    def verdict_with_data(
        self, input: PhaseInput
    ) -> Result[SuccessForReview, PhaseError]:
        raise NotImplementedError("method not implemented yet")
