from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, ActionWrapper

from minigrid.core.actions import Actions
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class MinigridTextObservationWrapper(ObservationWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = deepcopy(self.env.observation_space)
        self.observation_space.spaces["text"] = spaces.Text()

    def observation(self, observation) -> dict:
        # Decode objects in the 'image' observation
        # Represent them as strings

        text_obs = ""
        image = observation["image"]
        h, w, _ = image.shape
        for i in range(h):
            for j in range(w):
                cell = image[i, j]
                object_idx, color_idx, state_idx = cell
                object_type = IDX_TO_OBJECT[object_idx]
                color = IDX_TO_COLOR[color_idx]

                if object_type == "door":
                    state = IDX_TO_STATE[state_idx] + " "
                else:
                    state = ""

                if object_type == "unseen":
                    continue
                else:
                    text_obs += f"{state}{color} {object_type} at {(i, j)}\n"

        observation["text"] = text_obs
        return observation


class MinigridTextActionWrapper(ActionWrapper):
    @property
    def spec(self):
        return self.env.spec

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Text()

    def action(self, action) -> Actions:
        action_str = action.strip().lower()
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
