from typing import Type
import numpy as np

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractModelFeature
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

from sledge.autoencoder.preprocessing.features.map_id_feature import MapID, MAP_NAME_TO_ID


class MapIDTargetBuilder(AbstractTargetBuilder):
    def __init__(self) -> None:
        pass

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "map_id"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return MapID

    def get_targets(self, scenario: AbstractScenario) -> MapID:
        """Inherited, see superclass."""
        id = np.array(MAP_NAME_TO_ID[scenario.map_api.map_name], dtype=np.int64)
        return MapID(id)
