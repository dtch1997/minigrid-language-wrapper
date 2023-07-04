import gymnasium as gym

from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.core.actions import Actions
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ObjectDescription:
    type: str = ""
    color: str = ""

    def to_string(self):
        return f"{self.color} {self.type}"

    @staticmethod
    def from_string(string):
        color, type, _, location = string.split()
        return ObjectDescription(type, color, location)

    def match(self, object: WorldObj):
        return (self.type == "any" or self.type == object.type) and (
            self.color == "any" or self.color == object.color
        )


def get_object_pos(grid: Grid, object_desc: ObjectDescription) -> Tuple[int, int]:
    """Get position of object matching description.
    If there are multiple objects matching the description,
    return the first one.
    If there are no objects matching the description,
    return (-1, -1)
    """
    for idx, object in enumerate(grid.grid):
        if object is None:
            continue
        if object_desc.match(object):
            row = idx // grid.width
            col = idx % grid.width
            return (row, col)
    return (-1, -1)


def grid_to_str(grid: Grid) -> str:
    """Get string description of grid."""
    grid_str = ""
    for idx, object in enumerate(grid.grid):
        if object is None:
            continue
        else:
            y = idx // grid.width
            x = idx % grid.width
            cell_str = f"{object.color} {object.type} at {(x,y)}\n"
            if object.type == "door":
                if object.is_locked:
                    cell_str = "locked " + cell_str
                elif not object.is_open:
                    cell_str = "closed " + cell_str
                else:
                    cell_str = "open " + cell_str
            grid_str += cell_str
    return grid_str


# TODO: Room grid to string


def env_to_str(env: "gym.Env"):
    grid_str = grid_to_str(env.grid)
    agent_pos = env.agent_pos
    agent_dir = env.agent_dir
    agent_dir_str = ["right", "down", "left", "up"][agent_dir]
    env_str = (
        f"The environment consists of:\n"
        + "\n".join([("--" + line) for line in grid_str.split("\n ")])
        + f"The overall mission is: {env.mission}\n"
        f"The agent is at: {agent_pos}\n"
        f"The agent is facing: {agent_dir_str}"
    )
    return env_str


def str_to_action(action_str: str) -> Actions:
    """Convert string description of action to action."""
    action_str = action_str.strip().lower()
    if action_str == "left":
        return Actions.left
    elif action_str == "right":
        return Actions.right
    elif action_str == "forward":
        return Actions.forward
    elif action_str == "pickup":
        return Actions.pickup
    elif action_str == "drop":
        return Actions.drop
    elif action_str == "toggle":
        return Actions.toggle
    elif action_str == "done":
        return Actions.done
    else:
        raise ValueError(f"Unrecognized action: {action_str}")
