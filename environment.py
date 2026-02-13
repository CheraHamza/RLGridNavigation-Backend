"""Server-side GridWorld environment – mirrors the JS GridWorld exactly."""


class GridWorld:
    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        starting_position: list[int] | None = None,
        target_position: list[int] | None = None,
        obstacles: list[list[int]] | None = None,
    ):
        self.height = height
        self.width = width
        self.starting_position = list(starting_position or [0, 0])
        self.target_position = list(target_position or [8, 8])
        self.obstacles: set[tuple[int, int]] = set()
        if obstacles:
            self.obstacles = {(o[0], o[1]) for o in obstacles}
        self.current_position = list(self.starting_position)
        self.max_steps = height * width
        self.steps = 0

    def reset(self) -> dict:
        self.current_position = list(self.starting_position)
        self.steps = 0
        return self._get_state()

    def _get_state(self) -> dict:
        return {
            "position": list(self.current_position),
            "target": list(self.target_position),
        }

    def step(self, action: str) -> dict:
        x, y = self.current_position
        new_x, new_y = x, y
        reward = -0.01
        done = False

        if action == "up":
            if y > 0:
                new_y = y - 1
            else:
                reward = -0.1
        elif action == "down":
            if y < self.height - 1:
                new_y = y + 1
            else:
                reward = -0.1
        elif action == "left":
            if x > 0:
                new_x = x - 1
            else:
                reward = -0.1
        elif action == "right":
            if x < self.width - 1:
                new_x = x + 1
            else:
                reward = -0.1

        # Check obstacle collision – treat like hitting a wall
        if (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y
            reward = -0.1

        self.current_position = [new_x, new_y]
        self.steps += 1

        if new_x == self.target_position[0] and new_y == self.target_position[1]:
            reward = 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return {
            "state": self._get_state(),
            "reward": reward,
            "done": done,
            "steps": self.steps,
            "max_steps": self.max_steps,
        }
