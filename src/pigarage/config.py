from configparser import ConfigParser, SectionProxy
from pathlib import Path

VISUAL_DEBUG = False


class Config:
    def __init__(self) -> None:
        self.config = ConfigParser()
        self.config.read("pigarage.ini")

    @property
    def mqtt(self) -> SectionProxy:
        return self.config["mqtt"]

    @property
    def gpio(self) -> SectionProxy:
        return self.config["gpio"]

    @property
    def logging(self) -> SectionProxy:
        return self.config["logging"]

    @property
    def pigarage(self) -> SectionProxy:
        return self.config["pigarage"]

    @property
    def logdir(self) -> Path:
        return Path(self.logging.get("logdir", ".")).resolve()

    @property
    def server(self) -> SectionProxy:
        return (
            self.config["pigarage-server"]
            if self.config.has_section("pigarage-server")
            else {}
        )

    @property
    def client(self) -> SectionProxy:
        return (
            self.config["pigarage-client"]
            if self.config.has_section("pigarage-client")
            else {}
        )


config = Config()
