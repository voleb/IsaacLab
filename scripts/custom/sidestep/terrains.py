
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for railway terrains."""

from __future__ import annotations

import numpy as np
from typing import Tuple

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass


@configclass
class HfRailwayOnHillTerrainCfg(HfTerrainBaseCfg):
    """Configuration for the railway on hill terrain."""
    
    rail_height: float = 0.174
    """Height of the rail (m). Defaults to 60kg rail height."""
    
    rail_width: float = 0.145 
    """Width of the rail base (m). Defaults to 60kg rail base width."""
    
    rail_head_width: float = 0.065
    """Width of the rail head (m). Defaults to 60kg rail head width."""
    
    track_gauge: float = 1.435
    """Distance between the inner faces of the rail heads (m). Standard gauge."""
    
    hill_height: float = 0.5
    """Height of the hill/embankment (m)."""
    
    hill_width: float = 4.0
    """Width of the top of the embankment (m)."""
    
    slope_width: float = 2.0
    """Width of the sloped part of the embankment (m)."""
    
    noise_range: Tuple[float, float] = (0.0, 0.03)
    """Range of noise to add for gravel (m)."""
    
    noise_step: float = 0.005
    """Step size for noise (m)."""

    border_width: float = 0.0
    """The width of the border around the terrain (in m). Defaults to 0.0."""


@height_field_to_mesh
def railway_on_hill_terrain(difficulty: float, cfg: HfRailwayOnHillTerrainCfg) -> np.ndarray:
    """Generate a terrain with a railway track on an embankment (hill).

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
    """
    # resolve terrain configuration
    # -- scale difficulty parameters if needed
    hill_height = cfg.hill_height 
    # Scale rail height with difficulty: 0 -> 0, 1 -> cfg.rail_height
    # We want it to be passable at 0.
    current_rail_height = cfg.rail_height * difficulty
    
    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale) # Along Y
    
    # We assume the track runs along X (width direction in IsaacLab usually means X, length means Y)
    # Actually checking hf_terrains.py:
    # width_pixels corresponds to cfg.size[0] (x-axis)
    # length_pixels corresponds to cfg.size[1] (y-axis)
    # Most terrains are laid out in a grid.
    # If we want the robot to cross it, the track should probably run along Y (length), so robot crosses in X direction?
    # Or track along X, robot crosses in Y.
    # Standard locomotion envs usually move along X.
    # If the task is "side step to cross", usually the robot faces forward (X) and moves sideways (Y).
    # So the rail should be along X. That means the robot would need to step OVER something running parallel to its heading?
    # No, "side step to cross" usually means:
    # Robot faces the rail (e.g. facing +X), rail is running along Y (perpendicular to X).
    # Robot moves sideways (Y) to cross? No, that would be moving along the rail.
    # If robot faces +X and rail is along Y, robot steps forward (X) to cross.
    # "Side step" implies lateral movement.
    # So:
    # 1. Robot faces +X. Rail runs along X. Robot steps sideways (+Y or -Y) to cross the rail.
    #    This means the rail is parallel to the robot's heading. Robot is on one side, wants to go to the other side *laterally*.
    # 2. Robot faces +Y (perpendicular to rail). Rail runs along X. Robot steps sideways (+X or -X) to cross? No.
    # "Unitree go2가 side step으로 넘어서" -> "Side step to cross over".
    # This implies the robot is facing the rail (perpendicularly) and side-steps? No, that's wandering.
    # It implies the robot is PARALLEL to the rail and steps sideways over it.
    # Correct interpretation: Rail runs along X. Robot faces X. Robot steps in Y to cross the rail from one side to the other.
    # So the rail should be longitudinal (along X).
    
    # Let's create an embankment running along X.
    # Cross-section is in Y-Z plane.
    
    # Discrete dimensions
    y_pixels = length_pixels
    x_pixels = width_pixels
    
    # Embankment profile (along Y)
    # Center of terrain is y_pixels // 2
    center_y = y_pixels // 2
    
    # Embankment top width
    embankment_top_pixels = int(cfg.hill_width / cfg.horizontal_scale)
    slope_pixels = int(cfg.slope_width / cfg.horizontal_scale)
    embankment_height = int(hill_height / cfg.vertical_scale)
    
    # Rail dimensions
    rail_base_width_pixels = int(cfg.rail_width / cfg.horizontal_scale)
    if rail_base_width_pixels < 1: rail_base_width_pixels = 1
    
    rail_head_width_pixels = int(cfg.rail_head_width / cfg.horizontal_scale)
    if rail_head_width_pixels < 1: rail_head_width_pixels = 1
    
    rail_height_pixels = int(current_rail_height / cfg.vertical_scale)
    
    gauge_pixels = int(cfg.track_gauge / cfg.horizontal_scale)
    
    # Create height field
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    # 1. Embankment (Trapezoid along Y)
    # Y-coordinates relative to center
    y_coords = np.arange(length_pixels)
    dy = np.abs(y_coords - center_y)
    
    # Flat top
    hf_profile = np.zeros(length_pixels)
    mask_top = dy <= (embankment_top_pixels / 2)
    hf_profile[mask_top] = embankment_height
    
    # Slopes
    mask_slope = (dy > (embankment_top_pixels / 2)) & (dy <= (embankment_top_pixels / 2 + slope_pixels))
    # Linear slope: h = h_max * (1 - (d - top_half) / slope_width)
    slope_dist = dy[mask_slope] - (embankment_top_pixels / 2)
    hf_profile[mask_slope] = embankment_height * (1.0 - slope_dist / slope_pixels)
    
    # Broadcast profile along X
    hf_raw[:] = hf_profile
    
    # 2. Add Rails (Two lines along X)
    # Rails are at center_y +/- (gauge / 2)
    # Since visual gauge is usually inner-to-inner, the center-to-center distance is gauge + head_width.
    # Let's assume gauge is center-to-center for simplicity here, or correct for head width.
    # Standard 1.435m is inner-to-inner. Head width ~6.5cm. Center-to-center ~ 1.5m.
    # Center offset
    rail_offset = int((cfg.track_gauge + cfg.rail_head_width) / 2 / cfg.horizontal_scale)
    
    rail_y_centers = [center_y - rail_offset, center_y + rail_offset]
    
    for y_c in rail_y_centers:
        # Add rail bump
        # We can make it a simple box for now, or a small trapezoid if resolution permits.
        # With 0.1m resolution, it's just a bump.
        y_start = y_c - rail_base_width_pixels // 2
        y_end = y_start + rail_base_width_pixels
        
        # Clip to terrain bounds
        y_start = max(0, y_start)
        y_end = min(length_pixels, y_end)
        
        if y_end > y_start:
            hf_raw[:, y_start:y_end] += rail_height_pixels

    # 3. Add Noise (Gravel)
    # Gravel noise
    noise_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    noise_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    noise_step = int(cfg.noise_step / cfg.vertical_scale)
    
    if noise_max > noise_min:
        noise_range = np.arange(noise_min, noise_max + noise_step, noise_step)
        noise = np.random.choice(noise_range, size=(width_pixels, length_pixels))
        hf_raw += noise

    return np.rint(hf_raw).astype(np.int16)
