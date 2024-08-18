from typing import Any, Dict

from nuplan.common.maps.abstract_map import SemanticMapLayer

PLOT_PALETTE: Dict[str, str] = {
    "font_lavender_gray": "#adb9e3",
    "font_white": "#c9eaf1",
    "background_white": "#fafafa",
    "background_black": "#000000",
    "chart_green": "#00FF73",
    "chart_yellow": "#E2FF1A",
}

default_data_table: Dict[str, Any] = {"row_height": 80}

default_multi_choice_style: Dict[str, Any] = {"option_limit": 10}

default_spinner_style: Dict[str, Any] = {"width": 300, "low": 1}
default_div_style: Dict[str, Any] = {"margin": (5, 5, 5, 30), "width": 800}
base_tab_style: Dict[str, Any] = {
    "plot_sizes": (350, 300),
    "plot_frame_sizes": (1200, 1200),
    "search_criteria_sizes": (80, 80),
}

simulation_tile_style: Dict[str, Any] = {
    "figure_sizes": (500, 500),
    "render_figure_sizes": (1500, 1500),
    "figure_margins": [5, 40, 0, 30],
    "figure_title_text_font_size": "10pt",
    "video_button_margins": [5, 40, 5, 35],  # Top, right, bottom, left
    "frame_control_button_margins": [5, 19, 5, 35],  # Top, right, bottom, left
    "slider_margins": [5, 40, 0, 30],
    "background_color": "#FFFFFF",
    "mission_goal_color": "#00FF00",
    "mission_goal_alpha": 0.0,
    "mission_goal_line_width": 2,
    "decimal_points": 2,
}

render_style: Dict[str, Any] = {
    "figure_sizes": (500, 500),
    "render_figure_sizes": (500, 500),
    "figure_margins": [5, 40, 0, 30],
    "figure_title_text_font_size": "10pt",
    "video_button_margins": [5, 40, 5, 35],  # Top, right, bottom, left
    "frame_control_button_margins": [5, 19, 5, 35],  # Top, right, bottom, left
    "slider_margins": [5, 40, 0, 30],
    "background_color": "#FFFFFF",
    "mission_goal_color": "#00FF00",
    "mission_goal_alpha": 0.0,
    "mission_goal_line_width": 2,
    "decimal_points": 2,
}

simulation_tile_trajectory_style: Dict[str, Any] = {
    "ego": {"line_color": "#B0E685", "line_alpha": 0.0, "line_width": 3},
    "expert_ego": {"line_color": "#de7b6c", "line_alpha": 0.0, "line_width": 0},
}

simulation_tile_agent_style: Dict[str, Any] = {
    "ego": {
        "fill_color": "#de7b6c",
        "fill_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 3,
    },
    "vehicles": {
        "fill_color": "#699cdb",
        "fill_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 3,
    },
    "pedestrians": {
        "fill_color": "#b07aa1",
        "fill_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 3,
    },
    "bicycles": {
        "fill_color": "#699CDB",
        "fill_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 3,
    },
    "genericobjects": {
        "fill_color": "#edc948",
        "fill_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 3,
    },
    "traffic_cone": {
        "fill_color": "#edc948",
        "fill_alpha": 0.5,
        "line_color": "#000000",
        "line_width": 3,
    },
    "barrier": {
        "fill_color": "#edc948",
        "fill_alpha": 0.5,
        "line_color": "#000000",
        "line_width": 3,
    },
    "czone_sign": {
        "fill_color": "#edc948",
        "fill_alpha": 0.5,
        "line_color": "#000000",
        "line_width": 3,
    },
}

simulation_map_layer_color: Dict[SemanticMapLayer, Any] = {
    SemanticMapLayer.LANE: {
        "fill_color": "#D3D3D3",
        "fill_color_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 0.0,
    },
    SemanticMapLayer.WALKWAYS: {
        "fill_color": "#699CDB",
        "fill_color_alpha": 0.0,
        "line_color": "#7e772e",
        "line_width": 0.0,
    },
    SemanticMapLayer.CARPARK_AREA: {
        "fill_color": "#ff7f00",
        "fill_color_alpha": 0.0,
        "line_color": "#ff7f00",
        "line_width": 0.0,
    },
    SemanticMapLayer.PUDO: {
        "fill_color": "#AF75A7",
        "fill_color_alpha": 0.0,
        "line_color": "#AF75A7",
        "line_width": 0.0,
    },
    SemanticMapLayer.INTERSECTION: {
        "fill_color": "#D3D3D3",
        "fill_color_alpha": 1.0,
        "line_color": "#000000",
        "line_width": 0.0,
    },
    SemanticMapLayer.STOP_LINE: {
        "fill_color": "#FF0101",
        "fill_color_alpha": 0.0,
        "line_color": "#FF0101",
        "line_width": 0.0,
    },
    SemanticMapLayer.CROSSWALK: {
        "fill_color": "#B5B5B5",
        "fill_color_alpha": 0.0,
        "line_color": "#B5B5B5",
        "line_width": 0.0,
    },
    SemanticMapLayer.ROADBLOCK: {
        "fill_color": "#0000C0",
        "fill_color_alpha": 0.0,
        "line_color": "#0000C0",
        "line_width": 0.0,
    },
    SemanticMapLayer.BASELINE_PATHS: {
        "line_color": "#666666",
        "line_color_alpha": 1.0,
        "line_width": 3.0,
    },
    SemanticMapLayer.LANE_CONNECTOR: {
        "line_color": "#666666",
        "line_color_alpha": 1.0,
        "line_width": 3.0,
    },
}

configuration_tab_style: Dict[str, Any] = {
    "file_path_input_margin": [15, 0, 50, 30],
    "folder_path_selection_margin": [12, 0, 0, 6],
    "main_board_layout_height": 600,
}

overview_tab_style: Dict[str, Any] = {
    "table_margins": [20, 0, 0, 50],
    "table_width": 800,
    "table_height": 300,
    "scatter_size": 10,
    "statistic_figure_margin": [10, 20, 20, 30],
    "plot_legend_background_fill_alpha": 0.3,
    "plot_grid_line_color": "white",
    "overview_title_div_margin": [10, 20, 20, 30],
}

scenario_tab_style: Dict[str, Any] = {
    "default_div_width": 800,
    "col_offset_width": 400,
    "ego_expert_state_figure_sizes": [500, 250],
    "scenario_metric_score_figure_sizes": [400, 350],
    "time_series_figure_margins": [10, 20, 20, 30],
    "time_series_figure_title_text_font_size": "0.9em",
    "time_series_figure_xaxis_axis_label_text_font_size": "0.8em",
    "time_series_figure_xaxis_major_label_text_font_size": "0.8em",
    "time_series_figure_yaxis_axis_label_text_font_size": "0.8em",
    "time_series_figure_yaxis_major_label_text_font_size": "0.8em",
    "plot_legend_label_text_font_size": "0.7em",
}
