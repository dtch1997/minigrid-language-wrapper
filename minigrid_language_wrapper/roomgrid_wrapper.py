import numpy as np
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, ActionWrapper

import minigrid
from minigrid.core.actions import Actions
from minigrid.core.constants import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
    STATE_TO_IDX,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

from minigrid_language_wrapper.text_wrapper import (
    MinigridTextObservationWrapper,
)


class RoomgridObservationWrapper(ObservationWrapper):

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


class RoomgridTextObservationWrapper(ObservationWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(self.env.observation_space)
        self.observation_space.spaces["text"] = spaces.Text(max_length=4096)

    def observation(self, observation) -> dict:
        """Describes objects at a higher level than the Minigrid wrapper"""

        text_obs = ""

        agent_x, agent_y = self.unwrapped.agent_pos
        directions = ["east", "south", "west", "north"]
        agent_dir = directions[observation["direction"]]
        room = self.unwrapped.room_from_pos(*self.unwrapped.agent_pos)
        agent_pos_room = (agent_x - room.top[0], agent_y - room.top[1])
        text_obs += f"You are in a room, facing {agent_dir}.\n"

        for i, door in enumerate(room.doors):
            if door:
                door_state = "closed"
                if door.is_open:
                    door_state = "open"
                elif door.is_locked:
                    door_state = "locked"
                text_obs += f"There is a {door_state} door to the {directions[i]}.\n"

        # Describe objects in view
        image = observation["image"]
        h, w, _ = image.shape
        for i in range(h):
            for j in range(w):
                # Calculate relative direction of object from agent
                object_pos = (i, j)
                object_dir = get_relative_direction(agent_pos_room, object_pos)

                cell = image[i, j]
                object_idx, color_idx, state_idx = cell
                object_type = IDX_TO_OBJECT[object_idx]
                color = IDX_TO_COLOR[color_idx]

                if object_type == "door":
                    state = IDX_TO_STATE[state_idx] + " "
                else:
                    state = ""

                if object_type in ("unseen", "empty", "wall", "agent"):
                    continue
                if object_type == "door":
                    # Handled above
                    continue
                else:
                    text_obs += f"There is a {state}{color} {object_type} to your {object_dir}\n"

        # Describe objects in inventory
        if self.unwrapped.carrying:
            object = self.unwrapped.carrying
            text_obs += f"You are carrying a {object.color} {object.type}.\n"

        observation["text"] = text_obs
        return observation
