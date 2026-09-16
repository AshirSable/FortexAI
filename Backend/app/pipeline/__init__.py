# THOUGHT: I need to cover 4 main phases, give them an order, and also keep an log of each phase initializiation
# therefore i need to keep a log at each given session, which can be traced back
# at the same time i need to keep a small cache of outputs given by each phase, and keep the next phase ready
# what else i need to make sure is that we know when it is an attack at some phase i stop the pipeline

# THOUGHT: Lets make an enum which would indicate attack and benign first
# another enum that would indicate the phase
# a struct that would keep this enum with confidence and a record of the phase
# another struct which would keep the previous struct with the input data...
from abc import ABC, abstractmethod

from app.type_store import Phase, PhaseInput, Result, SuccessForReview, SuccessReturn

from app.type_store._error import PhaseError


class Pipeline: ...


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
