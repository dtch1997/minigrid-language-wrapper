import numpy as np
from typing import Tuple
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

import minigrid
from minigrid.core.roomgrid import Room, RoomGrid
from minigrid.core.constants import (
    STATE_TO_IDX,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class RoomGridObservationWrapper(ObservationWrapper):

    """Replace the 'image' observation with an observation of the current room"""

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(
            self.unwrapped, minigrid.roomgrid.RoomGrid
        ), "This wrapper only works with RoomGrid environments"
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.room_size, self.env.room_size, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, observation) -> dict:
        env = self.unwrapped

        room = env.room_from_pos(*env.agent_pos)
        agentX, agentY = env.agent_pos
        topX, topY = room.top
        room_grid = env.grid.slice(topX, topY, env.room_size, env.room_size)

        # Calculate the agent's position in the room
        agentXRoom, agentYRoom = agentX - topX, agentY - topY
        room_grid_enc = room_grid.encode()

        # Replace the agent's position in the room with the agent
        # Notes:
        # - This overwrites any other object in the agent's position
        # E.g. if the agent is standing on a door, the door will be replaced with the agent
        # - The agent's direction is not encoded in the room grid
        room_grid_enc[agentXRoom][agentYRoom] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**observation, "image": room_grid_enc}


def get_relative_direction(source_pos, target_pos):
    """Get the relative direction of the target position from the source position"""
    directions = ["east", "south", "west", "north"]
    dx = target_pos[0] - source_pos[0]
    dy = target_pos[1] - source_pos[1]
    if dx == 0 and dy == 0:
        return "here"
    elif dx == 0:
        if dy > 0:
            return "south"
        else:
            return "north"
    elif dy == 0:
        if dx > 0:
            return "east"
        else:
            return "west"
    # Handle diagonals
    if dx > 0:
        if dy > 0:
            return "southeast"
        else:
            return "northeast"
    else:
        if dy > 0:
            return "southwest"
        else:
            return "northwest"


def describe_room(room: Room) -> str:
    text_obs = ""

    # Describe doors to other rooms
    directions = ("east", "south", "west", "north")
    text_obs += "The room has doors: "
    for i, door in enumerate(room.doors):
        if door:
            door_state = "closed"
            if door.is_open:
                door_state = "open"
            elif door.is_locked:
                door_state = "locked"
            text_obs += f"{door_state} {door.color} door leading {directions[i]}, "
    text_obs += "\n"

    # Describe objects in room
    text_obs += "The room has objects: "
    for obj in room.objs:
        if obj.type in ("unseen", "empty"):
            continue
        elif obj.type in ("wall", "agent"):
            # Walls, agent are (implicitly) part of the room
            continue
        elif obj.type == "door":
            # Handled above
            continue
        text_obs += f"{obj.color} {obj.type},"
    text_obs += "\n"
    return text_obs


def describe_agent(env):
    text_obs = ""

    # Describe agent
    directions = ("east", "south", "west", "north")
    text_obs += f"The agent is facing {directions[env.agent_dir]}.\n"

    # Describe objects adjacent to agent
    directions = ("east", "south", "west", "north")
    delta_pos = ((1, 0), (0, 1), (-1, 0), (0, -1))
    text_obs += "The agent is next to objects: "
    for dir, dpos in zip(directions, delta_pos):
        obj_pos = (env.agent_pos[0] + dpos[0], env.agent_pos[1] + dpos[1])
        obj = env.grid.get(*obj_pos)
        if obj is not None and not obj.type in ("unseen", "empty"):
            text_obs += f"{obj.color} {obj.type} to the {dir}, "
    text_obs += "\n"

    # TODO: describe agent field of view

    # Describe inventory
    if env.carrying:
        text_obs += (
            f"The agent is carrying: {env.carrying.color} {env.carrying.type}.\n"
        )

    return text_obs


def describe_roomgrid(env: RoomGrid) -> str:
    """Describe all rooms in a RoomGrid env"""
    text_obs = ""
    for row in range(env.num_rows):
        for col in range(env.num_cols):
            room = env.get_room(col, row)
            text_obs += "Room ({}, {}):\n".format(col, row)
            text_obs += describe_room(room)
    return text_obs


def room_idx_from_pos(env: RoomGrid, x: int, y: int) -> Tuple[int, int]:
    """Get the room a given position maps to"""

    assert x >= 0
    assert y >= 0

    i = x // (env.room_size - 1)
    j = y // (env.room_size - 1)

    assert i < env.num_cols
    assert j < env.num_rows

    return (i, j)


class RoomGridTextPartialObsWrapper(ObservationWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env, fully_obs=False):
        assert isinstance(
            env.unwrapped, minigrid.roomgrid.RoomGrid
        ), "This wrapper only works with RoomGrid environments"
        super().__init__(env)

        self.fully_obs = fully_obs
        self.observation_space = deepcopy(self.env.observation_space)
        self.observation_space.spaces["text"] = spaces.Text(max_length=4096)

    def observation(self, observation) -> dict:
        """Describes objects at a higher level than the Minigrid wrapper"""

        env = self.unwrapped
        text_obs = ""

        if self.fully_obs:
            room_idx = room_idx_from_pos(env, *env.agent_pos)
            text_obs += f"The agent is in room ({room_idx[0]}, {room_idx[1]}).\n"
            text_obs += describe_roomgrid(env)
        else:
            room = env.room_from_pos(*env.agent_pos)
            text_obs += "The agent is in a room."
            text_obs += describe_room(room)

        text_obs += describe_agent(env)

        observation["text"] = text_obs
        return observation


class RoomGridTextFullyObsWrapper(RoomGridTextPartialObsWrapper):
    def __init__(self, env):
        super().__init__(env, fully_obs=True)
