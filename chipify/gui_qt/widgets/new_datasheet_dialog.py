# Copyright (c) 2026 Santiago Hofwimmer
"""
new_datasheet_dialog.py – Name a new datasheet and pick what it starts from (Qt).

"Start from" offers the blank starter template or any existing datasheet, so a
variant of a datasheet (a mismatch run with fewer seeds for a quicker check,
say) starts as a copy of the original instead of from scratch. The file itself
is written by :func:`chipify.uikit.services.yaml_editor_service.create_datasheet`.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from chipify.gui_qt.widgets.helpers import autoclose_combo

BLANK_TEMPLATE = "Blank template"


class NewDatasheetDialog(QDialog):
    """Modal prompt for a new datasheet's name and starting point."""

    def __init__(self, parent: QWidget, existing: list[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Datasheet")
        self.setMinimumWidth(420)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. mismatch_quick")
        self.source_combo = QComboBox()
        self.source_combo.addItem(BLANK_TEMPLATE)
        self.source_combo.addItems(existing)
        autoclose_combo(self.source_combo)

        form = QFormLayout()
        form.addRow("Name:", self.name_edit)
        form.addRow("Start from:", self.source_combo)

        hint = QLabel("A copy takes the saved file as it is on disk, with its "
                      "comments, equations and reports block.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)

        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self.name_edit.textChanged.connect(self._update_ok)
        self._update_ok()

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self._buttons)

    def _update_ok(self, *_a) -> None:
        self._buttons.button(QDialogButtonBox.Ok).setEnabled(
            bool(self.name_edit.text().strip()))

    def name(self) -> str:
        return self.name_edit.text().strip()

    def source(self) -> str | None:
        """File name of the datasheet to copy, or None for the blank template."""
        text = self.source_combo.currentText()
        return None if text == BLANK_TEMPLATE else text
