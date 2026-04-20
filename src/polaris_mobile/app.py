"""Mobile app shell for iOS/Android Briefcase packaging."""

from __future__ import annotations

import toga
from toga.style import Pack
from toga.style.pack import COLUMN


class PolarisMobile(toga.App):
    def startup(self) -> None:
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=16))
        title = toga.Label("Polaris Mobile", style=Pack(padding_bottom=12, font_size=18))
        body = toga.Label(
            "Mobile packaging is now configured with Briefcase.\n"
            "This shell is the native mobile app entrypoint.",
            style=Pack(padding_bottom=8),
        )
        status = toga.Label(
            "Next step: integrate mobile-native workflows for analysis features.",
            style=Pack(),
        )
        main_box.add(title)
        main_box.add(body)
        main_box.add(status)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()


def main() -> PolarisMobile:
    return PolarisMobile()
