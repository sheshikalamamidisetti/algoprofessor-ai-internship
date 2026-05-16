import os

# Enable ANSI colors in Windows terminal
os.system("")


class TeamLogger:

    COLORS = {
        "RESET": "\033[0m",
        "RED": "\033[91m",
        "GREEN": "\033[92m",
        "YELLOW": "\033[93m",
        "BLUE": "\033[94m",
        "CYAN": "\033[96m",
        "BOLD": "\033[1m"
    }

    def __init__(self, team_name="TEAM"):
        self.team_name = team_name

    def header(self, message):
        bar = "=" * 70
        print(
            f"\n{self.COLORS['BOLD']}{self.COLORS['CYAN']}{bar}"
        )
        print(message)
        print(
            f"{bar}{self.COLORS['RESET']}"
        )

    def info(self, message):
        print(
            f"{self.COLORS['BLUE']}[INFO]{self.COLORS['RESET']} {message}"
        )

    def success(self, agent, message):
        print(
            f"{self.COLORS['GREEN']}[SUCCESS]{self.COLORS['RESET']} [{agent}] {message}"
        )

    def agent(self, agent, message):
        print(
            f"{self.COLORS['YELLOW']}[{agent}]{self.COLORS['RESET']} {message}"
        )

    def error(self, message):
        print(
            f"{self.COLORS['RED']}[ERROR]{self.COLORS['RESET']} {message}"
        )