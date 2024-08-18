from dataclasses import dataclass
from typing import Any, Dict, Tuple
from PIL import ImageColor

from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType


@dataclass
class Color:
    """Dataclass for storing hex colors."""

    hex: str

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """
        :return: hex color as RBG tuple
        """
        return ImageColor.getcolor(self.hex, "RGB")

    @property
    def rgba(self) -> Tuple[int, int, int]:
        """
        :return: hex color as RBGA tuple
        """
        return ImageColor.getcolor(self.hex, "RGBA")


BLACK: Color = Color("#000000")
WHITE: Color = Color("#FFFFFF")
LIGHT_GREY: Color = Color("#D3D3D3")

TAB_10: Dict[int, Color] = {
    0: Color("#1f77b4"),  # blue
    1: Color("#ff7f0e"),  # orange
    2: Color("#2ca02c"),  # green
    3: Color("#d62728"),  # red
    4: Color("#9467bd"),  # violet
    5: Color("#8c564b"),  # brown
    6: Color("#e377c2"),  # pink
    7: Color("#7f7f7f"),  # grey
    8: Color("#bcbd22"),  # yellow
    9: Color("#17becf"),  # cyan
}


NEW_TAB_10: Dict[int, str] = {
    0: Color("#4e79a7"),  # blue
    1: Color("#f28e2b"),  # orange
    2: Color("#e15759"),  # red
    3: Color("#76b7b2"),  # cyan
    4: Color("#59a14f"),  # green
    5: Color("#edc948"),  # yellow
    6: Color("#b07aa1"),  # violet
    7: Color("#ff9da7"),  # pink-ish
    8: Color("#9c755f"),  # brown
    9: Color("#bab0ac"),  # grey
}


ELLIS_5: Dict[int, str] = {
    0: Color("#DE7061"),  # red
    1: Color("#B0E685"),  # green
    2: Color("#4AC4BD"),  # cyan
    3: Color("#E38C47"),  # orange
    4: Color("#699CDB"),  # blue
}


SLEDGE_ELEMENTS: Dict[SemanticMapLayer, Color] = {
    "lines": Color("#666666"),
    "vehicles": ELLIS_5[4],
    "pedestrians": NEW_TAB_10[6],
    "static_objects": NEW_TAB_10[5],
    "green_lights": TAB_10[2],
    "red_lights": TAB_10[3],
    "ego": ELLIS_5[0],
}

MAP_LAYER_CONFIG: Dict[SemanticMapLayer, Any] = {
    SemanticMapLayer.LANE: {
        "fill_color": LIGHT_GREY,
        "fill_color_alpha": 1.0,
        "line_color": LIGHT_GREY,
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.WALKWAYS: {
        "fill_color": "#d4d19e",
        "fill_color_alpha": 1.0,
        "line_color": "#d4d19e",
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.CARPARK_AREA: {
        "fill_color": Color("#b9d3b4"),
        "fill_color_alpha": 1.0,
        "line_color": Color("#b9d3b4"),
        "line_color_alpha": 0.0,
        "line_width": 0.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.PUDO: {
        "fill_color": Color("#AF75A7"),
        "fill_color_alpha": 0.3,
        "line_color": Color("#AF75A7"),
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.INTERSECTION: {
        "fill_color": Color("#D3D3D3"),
        "fill_color_alpha": 1.0,
        "line_color": Color("#D3D3D3"),
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.STOP_LINE: {
        "fill_color": Color("#FF0101"),
        "fill_color_alpha": 0.0,
        "line_color": Color("#FF0101"),
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.CROSSWALK: {
        "fill_color": NEW_TAB_10[6],
        "fill_color_alpha": 0.3,
        "line_color": NEW_TAB_10[6],
        "line_color_alpha": 0.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.ROADBLOCK: {
        "fill_color": Color("#0000C0"),
        "fill_color_alpha": 0.2,
        "line_color": Color("#0000C0"),
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
    SemanticMapLayer.BASELINE_PATHS: {
        "line_color": Color("#666666"),
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "--",
        "zorder": 1,
    },
    SemanticMapLayer.LANE_CONNECTOR: {
        "line_color": Color("#CBCBCB"),
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 1,
    },
}

AGENT_CONFIG: Dict[TrackedObjectType, Any] = {
    TrackedObjectType.VEHICLE: {
        "fill_color": ELLIS_5[4],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.PEDESTRIAN: {
        "fill_color": NEW_TAB_10[6],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.BICYCLE: {
        "fill_color": ELLIS_5[3],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.TRAFFIC_CONE: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.BARRIER: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.CZONE_SIGN: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.GENERIC_OBJECT: {
        "fill_color": NEW_TAB_10[5],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
    TrackedObjectType.EGO: {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": "black",
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "-",
        "zorder": 2,
    },
}

TRAFFIC_LIGHT_CONFIG: Dict[TrafficLightStatusType, Any] = {
    TrafficLightStatusType.RED: {
        "line_color": TAB_10[3],
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "--",
        "zorder": 1,
    },
    TrafficLightStatusType.GREEN: {
        "line_color": TAB_10[2],
        "line_color_alpha": 1.0,
        "line_width": 1.0,
        "line_style": "--",
        "zorder": 1,
    },
}

TRAJECTORY_CONFIG: Dict[str, Any] = {
    "human": {
        "fill_color": NEW_TAB_10[4],
        "fill_color_alpha": 1.0,
        "line_color": NEW_TAB_10[4],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
    "agent": {
        "fill_color": ELLIS_5[0],
        "fill_color_alpha": 1.0,
        "line_color": ELLIS_5[0],
        "line_color_alpha": 1.0,
        "line_width": 2.0,
        "line_style": "-",
        "marker": "o",
        "marker_size": 5,
        "marker_edge_color": "black",
        "zorder": 3,
    },
}
