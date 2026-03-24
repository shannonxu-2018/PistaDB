"""
Shuckr — A visual browser for PistaDB .pst files.
Inspired by DB Browser for SQLite.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QAction, QFont, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QStatusBar, QMessageBox,
    QTabWidget, QTextEdit,
)

from pistadb_driver import (
    Driver, Database, Metric, IndexType, SearchResult,
    read_pst_header, PstHeader,
)

APP_NAME = "Shuckr"
APP_VERSION = "1.0.0"
LIB_SEARCH_DIRS = [
    str(Path(__file__).parent.parent / "build" / "Release"),
    str(Path(__file__).parent.parent / "build" / "Debug"),
    str(Path(__file__).parent.parent / "build"),
]

VECTORS_PAGE_SIZE = 200


# ── Dialogs ──────────────────────────────────────────────────────────────────

class NewDatabaseDialog(QDialog):
    """Dialog for creating a new .pst database."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create New Database")
        self.setMinimumWidth(380)

        layout = QFormLayout(self)

        self.path_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addRow("File path:", path_row)

        self.dim_spin = QSpinBox()
        self.dim_spin.setRange(1, 65536)
        self.dim_spin.setValue(128)
        layout.addRow("Dimension:", self.dim_spin)

        self.metric_combo = QComboBox()
        for m in Metric:
            self.metric_combo.addItem(m.name, m.value)
        layout.addRow("Metric:", self.metric_combo)

        self.index_combo = QComboBox()
        for idx in IndexType:
            self.index_combo.addItem(idx.name, idx.value)
        self.index_combo.setCurrentIndex(1)  # HNSW default
        layout.addRow("Index type:", self.index_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Database As", "", "PistaDB Files (*.pst);;All Files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def values(self):
        return (
            self.path_edit.text().strip(),
            self.dim_spin.value(),
            Metric(self.metric_combo.currentData()),
            IndexType(self.index_combo.currentData()),
        )


class InsertVectorDialog(QDialog):
    """Dialog for inserting or editing a single vector."""

    def __init__(self, dim: int, parent=None, vid: int = 0,
                 label: str = "", vector: Optional[np.ndarray] = None):
        super().__init__(parent)
        self.setWindowTitle("Insert Vector" if vector is None else "Edit Vector")
        self.setMinimumWidth(500)

        layout = QFormLayout(self)

        self.id_spin = QSpinBox()
        self.id_spin.setRange(0, 2**31 - 1)
        self.id_spin.setValue(vid)
        layout.addRow("ID:", self.id_spin)

        self.label_edit = QLineEdit(label)
        self.label_edit.setMaxLength(255)
        layout.addRow("Label:", self.label_edit)

        self.vector_edit = QTextEdit()
        self.vector_edit.setPlaceholderText(
            f"Enter {dim} float values, comma or space separated"
        )
        if vector is not None:
            self.vector_edit.setPlainText(
                ", ".join(f"{v:.6g}" for v in vector)
            )
        self.vector_edit.setMaximumHeight(120)
        layout.addRow(f"Vector ({dim}d):", self.vector_edit)

        self._dim = dim

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def _validate_and_accept(self):
        try:
            self.get_vector()
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Vector", str(e))

    def get_vector(self) -> np.ndarray:
        text = self.vector_edit.toPlainText().strip()
        parts = text.replace(",", " ").split()
        if len(parts) != self._dim:
            raise ValueError(
                f"Expected {self._dim} values, got {len(parts)}"
            )
        return np.array([float(x) for x in parts], dtype=np.float32)

    def values(self):
        return self.id_spin.value(), self.label_edit.text(), self.get_vector()


# ── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(960, 640)

        self._driver: Optional[Driver] = None
        self._db: Optional[Database] = None
        self._db_path: Optional[str] = None
        self._cached_ids: list[int] = []   # known-valid IDs from last scan
        self._browse_page = 0
        self._unsaved = False

        self._init_driver()
        self._build_actions()
        self._build_ui()
        self._update_title()
        self._set_db_actions_enabled(False)

        settings = QSettings(APP_NAME, APP_NAME)
        geo = settings.value("geometry")
        if geo:
            self.restoreGeometry(geo)

    # ── driver ────────────────────────────────────────────────────────────

    def _init_driver(self):
        for d in LIB_SEARCH_DIRS:
            try:
                self._driver = Driver(lib_dir=d)
                return
            except OSError:
                continue
        try:
            self._driver = Driver()
        except OSError:
            QMessageBox.critical(
                self, "Library Not Found",
                "Could not find pistadb.dll / libpistadb.so.\n"
                "Build the project first or set PISTADB_LIB_DIR.",
            )

    # ── Shared QActions ────────────────────────────────────────────────────

    def _build_actions(self):
        self._act_new = QAction("New Database", self)
        self._act_new.setShortcut(QKeySequence.StandardKey.New)
        self._act_new.triggered.connect(self._on_new)

        self._act_open = QAction("Open Database", self)
        self._act_open.setShortcut(QKeySequence.StandardKey.Open)
        self._act_open.triggered.connect(self._on_open)

        self._act_save = QAction("Save", self)
        self._act_save.setShortcut(QKeySequence.StandardKey.Save)
        self._act_save.triggered.connect(self._on_save)

        self._act_close = QAction("Close Database", self)
        self._act_close.setShortcut(QKeySequence("Ctrl+W"))
        self._act_close.triggered.connect(self._on_close_db)

        self._act_insert = QAction("Insert Vector", self)
        self._act_insert.setShortcut(QKeySequence("Ctrl+I"))
        self._act_insert.triggered.connect(self._on_insert)

        self._act_delete = QAction("Delete Selected", self)
        self._act_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self._act_delete.triggered.connect(self._on_delete_selected)

        self._act_edit = QAction("Edit Selected", self)
        self._act_edit.setShortcut(QKeySequence("Ctrl+E"))
        self._act_edit.triggered.connect(self._on_edit_selected)

        self._act_search = QAction("Search", self)
        self._act_search.setShortcut(QKeySequence("Ctrl+F"))
        self._act_search.triggered.connect(self._on_search_tab)

    # ── UI construction ──────────────────────────────────────────────────

    def _build_ui(self):
        self._build_menu()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()
        root.addWidget(self._tabs)

        self._tabs.addTab(self._build_info_tab(), "Database Info")
        self._tabs.addTab(self._build_browse_tab(), "Browse Data")
        self._tabs.addTab(self._build_search_tab(), "Search")

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Ready")
        self._status.addWidget(self._status_label, 1)

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction(self._act_new)
        file_menu.addAction(self._act_open)
        file_menu.addSeparator()
        file_menu.addAction(self._act_save)
        file_menu.addSeparator()
        file_menu.addAction(self._act_close)
        file_menu.addSeparator()
        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence.StandardKey.Quit)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction(self._act_insert)
        edit_menu.addAction(self._act_edit)
        edit_menu.addAction(self._act_delete)
        edit_menu.addSeparator()
        edit_menu.addAction(self._act_search)

        help_menu = mb.addMenu("&Help")
        act_about = QAction("&About", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ── Info tab ─────────────────────────────────────────────────────────

    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        grp = QGroupBox("Database Properties")
        form = QFormLayout(grp)

        self._info_path = QLabel("-")
        self._info_path.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("File:", self._info_path)
        self._info_dim = QLabel("-")
        form.addRow("Dimension:", self._info_dim)
        self._info_count = QLabel("-")
        form.addRow("Vector count:", self._info_count)
        self._info_metric = QLabel("-")
        form.addRow("Metric:", self._info_metric)
        self._info_index = QLabel("-")
        form.addRow("Index type:", self._info_index)
        self._info_lib_ver = QLabel("-")
        form.addRow("Library version:", self._info_lib_ver)
        layout.addWidget(grp)

        grp2 = QGroupBox("Raw .pst Header")
        hdr_layout = QVBoxLayout(grp2)
        self._header_text = QTextEdit()
        self._header_text.setReadOnly(True)
        self._header_text.setMaximumHeight(160)
        self._header_text.setFont(QFont("Consolas", 9))
        hdr_layout.addWidget(self._header_text)
        layout.addWidget(grp2)

        layout.addStretch()
        return w

    # ── Browse tab ───────────────────────────────────────────────────────

    def _build_browse_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        nav = QHBoxLayout()
        self._btn_first = QPushButton("|<")
        self._btn_prev = QPushButton("<")
        self._page_label = QLabel("Page 0 / 0")
        self._btn_next = QPushButton(">")
        self._btn_last = QPushButton(">|")
        self._btn_refresh = QPushButton("Refresh")

        for btn in (self._btn_first, self._btn_prev, self._btn_next, self._btn_last):
            btn.setFixedWidth(40)
        self._btn_refresh.setFixedWidth(80)

        self._btn_first.clicked.connect(lambda: self._go_page(0))
        self._btn_prev.clicked.connect(lambda: self._go_page(self._browse_page - 1))
        self._btn_next.clicked.connect(lambda: self._go_page(self._browse_page + 1))
        self._btn_last.clicked.connect(lambda: self._go_page(self._total_pages() - 1))
        self._btn_refresh.clicked.connect(self._full_rescan)

        nav.addWidget(self._btn_first)
        nav.addWidget(self._btn_prev)
        nav.addStretch()
        nav.addWidget(self._page_label)
        nav.addStretch()
        nav.addWidget(self._btn_next)
        nav.addWidget(self._btn_last)
        nav.addWidget(self._btn_refresh)
        layout.addLayout(nav)

        self._vec_table = QTableWidget()
        self._vec_table.setColumnCount(3)
        self._vec_table.setHorizontalHeaderLabels(["ID", "Label", "Vector (preview)"])
        self._vec_table.horizontalHeader().setStretchLastSection(True)
        self._vec_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._vec_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self._vec_table.doubleClicked.connect(self._on_edit_selected)
        layout.addWidget(self._vec_table)

        return w

    # ── Search tab ───────────────────────────────────────────────────────

    def _build_search_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        input_grp = QGroupBox("Query")
        input_layout = QFormLayout(input_grp)

        self._search_k = QSpinBox()
        self._search_k.setRange(1, 10000)
        self._search_k.setValue(10)
        input_layout.addRow("k:", self._search_k)

        self._search_vec_edit = QTextEdit()
        self._search_vec_edit.setPlaceholderText(
            "Enter query vector (comma or space separated floats)"
        )
        self._search_vec_edit.setMaximumHeight(80)
        input_layout.addRow("Vector:", self._search_vec_edit)

        self._search_random_btn = QPushButton("Random Query")
        self._search_random_btn.clicked.connect(self._fill_random_query)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._run_search)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._search_random_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._search_btn)
        input_layout.addRow(btn_row)
        layout.addWidget(input_grp)

        self._search_table = QTableWidget()
        self._search_table.setColumnCount(3)
        self._search_table.setHorizontalHeaderLabels(["ID", "Distance", "Label"])
        self._search_table.horizontalHeader().setStretchLastSection(True)
        self._search_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        layout.addWidget(self._search_table)

        return w

    # ── Actions enable/disable ───────────────────────────────────────────

    def _set_db_actions_enabled(self, enabled: bool):
        for act in (self._act_save, self._act_close, self._act_insert,
                    self._act_delete, self._act_edit, self._act_search):
            act.setEnabled(enabled)

    # ── ID scanning ──────────────────────────────────────────────────────
    # pistadb has no "list all IDs" API, so we probe sequential IDs once
    # and cache the result.  CRUD operations update the cache locally so
    # we don't rescan after every mutation.

    def _scan_all_ids(self) -> list[int]:
        """Probe IDs 0..max and return a sorted list of valid ones."""
        if not self._db:
            return []
        count = self._db.count
        if count == 0:
            return []

        ids: list[int] = []
        # Probe enough IDs to find all existing vectors.
        # After deletions there can be gaps, so probe well beyond count.
        miss_streak = 0
        vid = 0
        # Stop after we found `count` ids OR after 500 consecutive misses
        while miss_streak < 500:
            try:
                self._db.get(vid)
                ids.append(vid)
                miss_streak = 0
                if len(ids) >= count:
                    # Found all vectors reported by the library
                    break
            except RuntimeError:
                miss_streak += 1
            vid += 1
        return ids

    def _full_rescan(self):
        """Rescan all IDs from scratch and reload current page."""
        self._cached_ids = self._scan_all_ids()
        self._load_page()

    # ── File operations ──────────────────────────────────────────────────

    def _on_new(self):
        if not self._driver:
            QMessageBox.warning(self, "Error", "PistaDB library not loaded.")
            return
        if not self._confirm_close():
            return

        dlg = NewDatabaseDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        path, dim, metric, index_type = dlg.values()
        if not path:
            return

        try:
            db = self._driver.create(path, dim, metric, index_type)
            db.save()
            self._open_db(db, path)
            self._status_label.setText(f"Created: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create database:\n{e}")

    def _on_open(self):
        if not self._driver:
            QMessageBox.warning(self, "Error", "PistaDB library not loaded.")
            return
        if not self._confirm_close():
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Database", "",
            "PistaDB Files (*.pst);;All Files (*)",
        )
        if not path:
            return

        hdr = read_pst_header(path)
        if not hdr:
            QMessageBox.warning(self, "Error", "Not a valid .pst file.")
            return

        try:
            db = self._driver.open(
                path, hdr.dimension,
                Metric(hdr.metric_type),
                IndexType(hdr.index_type),
            )
            self._open_db(db, path)
            self._status_label.setText(f"Opened: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database:\n{e}")

    def _open_db(self, db: Database, path: str):
        self._close_db()
        self._db = db
        self._db_path = path
        self._unsaved = False
        self._set_db_actions_enabled(True)
        self._cached_ids = self._scan_all_ids()
        self._browse_page = 0
        self._refresh_info()
        self._load_page()
        self._update_title()

    def _on_save(self):
        if not self._db:
            return
        try:
            self._db.save()
            self._unsaved = False
            self._update_title()
            self._status_label.setText("Saved.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{e}")

    def _on_close_db(self):
        if self._confirm_close():
            self._close_db()

    def _close_db(self):
        if self._db:
            try:
                self._db.close()
            except Exception:
                pass
        self._db = None
        self._db_path = None
        self._unsaved = False
        self._cached_ids = []
        self._set_db_actions_enabled(False)
        self._clear_info()
        self._vec_table.setRowCount(0)
        self._search_table.setRowCount(0)
        self._page_label.setText("Page 0 / 0")
        self._update_title()

    def _confirm_close(self) -> bool:
        if not self._unsaved or not self._db:
            return True
        r = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Save before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Save:
            self._on_save()
            return True
        return r == QMessageBox.StandardButton.Discard

    # ── Info refresh ─────────────────────────────────────────────────────

    def _refresh_info(self):
        if not self._db:
            return
        self._info_path.setText(self._db_path or "-")
        self._info_dim.setText(str(self._db.dim))
        self._info_count.setText(str(len(self._cached_ids)))
        self._info_metric.setText(self._db.metric.name)
        self._info_index.setText(self._db.index_type.name)
        if self._driver:
            self._info_lib_ver.setText(self._driver.version)

        hdr = read_pst_header(self._db_path) if self._db_path else None
        if hdr:
            self._header_text.setPlainText(
                f"magic       = {hdr.magic}\n"
                f"version     = {hdr.version_major}.{hdr.version_minor}\n"
                f"flags       = 0x{hdr.flags:08X}\n"
                f"dimension   = {hdr.dimension}\n"
                f"metric      = {hdr.metric_name} ({hdr.metric_type})\n"
                f"index       = {hdr.index_name} ({hdr.index_type})\n"
                f"num_vectors = {hdr.num_vectors}"
            )
        else:
            self._header_text.clear()

    def _clear_info(self):
        for lbl in (self._info_path, self._info_dim, self._info_count,
                    self._info_metric, self._info_index, self._info_lib_ver):
            lbl.setText("-")
        self._header_text.clear()

    # ── Browse / pagination ──────────────────────────────────────────────

    def _total_pages(self) -> int:
        n = len(self._cached_ids)
        if n == 0:
            return 1
        return (n + VECTORS_PAGE_SIZE - 1) // VECTORS_PAGE_SIZE

    def _go_page(self, page: int):
        total = self._total_pages()
        page = max(0, min(page, total - 1))
        self._browse_page = page
        self._load_page()

    def _load_page(self):
        """Render the current page from the cached ID list."""
        self._vec_table.setRowCount(0)
        total_ids = len(self._cached_ids)
        total_pages = self._total_pages()

        # Clamp page
        if self._browse_page >= total_pages:
            self._browse_page = max(0, total_pages - 1)

        self._page_label.setText(
            f"Page {self._browse_page + 1} / {total_pages}  ({total_ids} vectors)"
        )

        if not self._db or total_ids == 0:
            return

        dim = self._db.dim
        start = self._browse_page * VECTORS_PAGE_SIZE
        end = min(start + VECTORS_PAGE_SIZE, total_ids)
        page_ids = self._cached_ids[start:end]

        # Fetch vectors for this page only
        rows: list[tuple[int, str, np.ndarray]] = []
        stale_ids: list[int] = []
        for vid in page_ids:
            try:
                vec, label = self._db.get(vid)
                rows.append((vid, label, vec))
            except RuntimeError:
                # ID no longer valid (deleted between scan and now)
                stale_ids.append(vid)

        # Remove stale IDs from cache
        if stale_ids:
            stale_set = set(stale_ids)
            self._cached_ids = [i for i in self._cached_ids if i not in stale_set]

        self._vec_table.setRowCount(len(rows))
        for i, (rid, label, vec) in enumerate(rows):
            self._vec_table.setItem(i, 0, QTableWidgetItem(str(rid)))
            self._vec_table.setItem(i, 1, QTableWidgetItem(label))
            preview = ", ".join(f"{v:.4f}" for v in vec[:8])
            if dim > 8:
                preview += f", ... ({dim}d)"
            self._vec_table.setItem(i, 2, QTableWidgetItem(preview))

        self._refresh_info()

    # ── CRUD operations ──────────────────────────────────────────────────

    def _on_insert(self):
        if not self._db:
            return
        # Suggest next available ID
        next_id = (max(self._cached_ids) + 1) if self._cached_ids else 0
        dlg = InsertVectorDialog(self._db.dim, self, vid=next_id)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vid, label, vector = dlg.values()

        if vid in self._cached_ids:
            r = QMessageBox.question(
                self, "ID Exists",
                f"Vector id={vid} already exists. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return
            try:
                self._db.delete(vid)
                self._cached_ids.remove(vid)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to remove old vector:\n{e}")
                return

        try:
            self._db.insert(vid, vector, label)
            # Update cache: insert in sorted position
            self._cached_ids.append(vid)
            self._cached_ids.sort()
            self._unsaved = True
            self._update_title()
            self._load_page()
            self._status_label.setText(f"Inserted vector id={vid}")
        except Exception as e:
            QMessageBox.critical(self, "Insert Failed", str(e))

    def _get_selected_browse_ids(self) -> list[int]:
        """Get IDs of selected rows in the browse table, switching tab if needed."""
        # Ensure we're looking at the browse tab
        if self._tabs.currentIndex() != 1:
            self._tabs.setCurrentIndex(1)

        rows = self._vec_table.selectionModel().selectedRows()
        ids = []
        for idx in rows:
            id_item = self._vec_table.item(idx.row(), 0)
            if id_item:
                ids.append(int(id_item.text()))
        return ids

    def _on_delete_selected(self):
        if not self._db:
            return

        ids = self._get_selected_browse_ids()
        if not ids:
            QMessageBox.information(self, "Delete", "No rows selected.\nSelect rows in the Browse Data tab first.")
            return

        r = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(ids)} vector(s)?\nIDs: {ids}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        deleted = []
        errors = []
        for vid in ids:
            try:
                self._db.delete(vid)
                deleted.append(vid)
            except Exception as e:
                errors.append(f"id={vid}: {e}")

        # Update cache: remove successfully deleted IDs
        if deleted:
            deleted_set = set(deleted)
            self._cached_ids = [i for i in self._cached_ids if i not in deleted_set]
            self._unsaved = True
            self._update_title()

        # Reload page (may shift to previous page if current page is now empty)
        self._load_page()

        if errors:
            QMessageBox.warning(self, "Delete Errors", "\n".join(errors))
        if deleted:
            self._status_label.setText(f"Deleted {len(deleted)} vector(s)")

    def _on_edit_selected(self):
        if not self._db:
            return

        ids = self._get_selected_browse_ids()
        if len(ids) != 1:
            QMessageBox.information(self, "Edit", "Select exactly one row to edit.")
            return

        vid = ids[0]
        try:
            vec, label = self._db.get(vid)
        except RuntimeError as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        dlg = InsertVectorDialog(
            self._db.dim, self, vid=vid, label=label, vector=vec,
        )
        dlg.id_spin.setEnabled(False)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        _, new_label, new_vec = dlg.values()
        try:
            self._db.update(vid, new_vec)
            self._unsaved = True
            self._update_title()
            self._load_page()
            self._status_label.setText(f"Updated vector id={vid}")
        except Exception as e:
            QMessageBox.critical(self, "Update Failed", str(e))

    # ── Search ───────────────────────────────────────────────────────────

    def _on_search_tab(self):
        if not self._db:
            return
        self._tabs.setCurrentIndex(2)

    def _fill_random_query(self):
        if not self._db:
            return
        dim = self._db.dim
        vec = np.random.randn(dim).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        self._search_vec_edit.setPlainText(
            ", ".join(f"{v:.6f}" for v in vec)
        )

    def _run_search(self):
        if not self._db:
            return
        dim = self._db.dim
        text = self._search_vec_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Search", "Enter a query vector.")
            return
        try:
            parts = text.replace(",", " ").split()
            if len(parts) != dim:
                QMessageBox.warning(
                    self, "Search",
                    f"Expected {dim} values, got {len(parts)}",
                )
                return
            query = np.array([float(x) for x in parts], dtype=np.float32)
        except ValueError as e:
            QMessageBox.warning(self, "Search", f"Invalid vector: {e}")
            return

        k = self._search_k.value()
        try:
            results = self._db.search(query, k)
        except Exception as e:
            QMessageBox.critical(self, "Search Failed", str(e))
            return

        self._search_table.setRowCount(len(results))
        for i, r in enumerate(results):
            self._search_table.setItem(i, 0, QTableWidgetItem(str(r.id)))
            self._search_table.setItem(
                i, 1, QTableWidgetItem(f"{r.distance:.6f}")
            )
            self._search_table.setItem(i, 2, QTableWidgetItem(r.label))

        self._status_label.setText(
            f"Search returned {len(results)} result(s)"
        )

    # ── Misc ─────────────────────────────────────────────────────────────

    def _update_title(self):
        parts = [APP_NAME]
        if self._db_path:
            name = Path(self._db_path).name
            if self._unsaved:
                name += " *"
            parts.insert(0, name)
        self.setWindowTitle(" \u2014 ".join(parts))

    def _on_about(self):
        lib_ver = self._driver.version if self._driver else "N/A"
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            f"A visual browser for PistaDB .pst vector database files.<br>"
            f"Inspired by DB Browser for SQLite.<br><br>"
            f"PistaDB library version: {lib_ver}",
        )

    def closeEvent(self, event):
        if not self._confirm_close():
            event.ignore()
            return
        settings = QSettings(APP_NAME, APP_NAME)
        settings.setValue("geometry", self.saveGeometry())
        self._close_db()
        event.accept()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
