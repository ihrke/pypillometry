"""Control panel for the interactive viewer."""

from pyqtgraph.Qt import QtCore, QtWidgets
from typing import Dict, Callable, Optional


class ControlPanel(QtWidgets.QWidget):
    """Control panel with checkboxes and buttons for viewer configuration.
    
    Signals
    -------
    modality_toggled : Signal(str, bool)
        Emitted when a data modality checkbox is toggled
    events_toggled : Signal(bool)
        Emitted when event markers checkbox is toggled
    mask_mode_changed : Signal(str)
        Emitted when mask visualization mode changes
    region_added : Signal()
        Emitted when "Add Region" button is clicked
    regions_cleared : Signal()
        Emitted when "Clear Regions" button is clicked
    accept_clicked : Signal()
        Emitted when "Accept" button is clicked
    cancel_clicked : Signal()
        Emitted when "Cancel" button is clicked
    """
    
    # Define signals
    modality_toggled = QtCore.Signal(str, bool)
    eye_toggled = QtCore.Signal(str, bool)  # 'left' or 'right', enabled
    variable_toggled = QtCore.Signal(str, bool)  # 'pupil', 'x', or 'y', enabled
    events_toggled = QtCore.Signal(bool)
    mask_mode_changed = QtCore.Signal(str)
    region_added = QtCore.Signal()
    regions_cleared = QtCore.Signal()
    accept_clicked = QtCore.Signal()
    cancel_clicked = QtCore.Signal()
    auto_y_toggled = QtCore.Signal(int, bool)  # plot_index, enabled
    time_unit_changed = QtCore.Signal(str)  # unit string ('ms', 's', 'min')
    show_whole_signal = QtCore.Signal()  # show entire signal
    
    def __init__(self, available_modalities: list, has_events: bool = False, plot_types: list = None):
        """Initialize control panel.
        
        Parameters
        ----------
        available_modalities : list of str
            List of available data modalities
        has_events : bool
            Whether event markers are available
        """
        super().__init__()
        
        self.modality_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.eye_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.variable_checkboxes: Dict[str, QtWidgets.QCheckBox] = {}
        self.auto_y_checkboxes: Dict[int, QtWidgets.QCheckBox] = {}
        self.available_modalities = available_modalities
        self.has_events = has_events
        self.plot_types = plot_types or []
        
        # Determine which eyes and variables are available
        self.available_eyes = set()
        self.available_variables = set()
        for modality in available_modalities:
            if 'left' in modality:
                self.available_eyes.add('left')
            if 'right' in modality:
                self.available_eyes.add('right')
            if 'pupil' in modality:
                self.available_variables.add('pupil')
            if '_x' in modality:
                self.available_variables.add('x')
            if '_y' in modality:
                self.available_variables.add('y')
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the control panel UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("<b>Controls</b>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        layout.addWidget(line)
        
        # Time unit selector
        layout.addWidget(QtWidgets.QLabel("<b>Time Unit:</b>"))
        self.time_unit_combo = QtWidgets.QComboBox()
        self.time_unit_combo.addItems(['Seconds (s)', 'Milliseconds (ms)', 'Minutes (min)'])
        self.time_unit_combo.setCurrentIndex(0)  # Default to seconds
        self.time_unit_combo.currentIndexChanged.connect(self._on_time_unit_changed)
        layout.addWidget(self.time_unit_combo)
        
        # Show whole signal button
        btn_show_all = QtWidgets.QPushButton("Show Whole Signal")
        btn_show_all.clicked.connect(self.show_whole_signal.emit)
        layout.addWidget(btn_show_all)
        
        layout.addWidget(self._create_separator())
        
        # Eyes section
        layout.addWidget(QtWidgets.QLabel("<b>Eyes:</b>"))
        eye_labels = {'left': 'Left Eye', 'right': 'Right Eye'}
        for eye in ['left', 'right']:
            if eye in self.available_eyes:
                checkbox = QtWidgets.QCheckBox(eye_labels[eye])
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(
                    lambda state, e=eye: self._on_eye_toggled(e, state == QtCore.Qt.Checked)
                )
                self.eye_checkboxes[eye] = checkbox
                layout.addWidget(checkbox)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Variables section
        layout.addWidget(QtWidgets.QLabel("<b>Variables:</b>"))
        var_labels = {'pupil': 'Pupil Size', 'x': 'Gaze X', 'y': 'Gaze Y'}
        for var in ['pupil', 'x', 'y']:
            if var in self.available_variables:
                checkbox = QtWidgets.QCheckBox(var_labels[var])
                checkbox.setChecked(True)
                checkbox.stateChanged.connect(
                    lambda state, v=var: self._on_variable_toggled(v, state == QtCore.Qt.Checked)
                )
                self.variable_checkboxes[var] = checkbox
                layout.addWidget(checkbox)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Auto Y-axis section
        if self.plot_types:
            layout.addWidget(QtWidgets.QLabel("<b>Auto Y-Axis:</b>"))
            for i, plot_type in enumerate(self.plot_types):
                label = {'pupil': 'Pupil', 'x': 'Gaze X', 'y': 'Gaze Y', 'all': 'All'}. get(plot_type, plot_type)
                checkbox = QtWidgets.QCheckBox(f"Auto {label}")
                checkbox.setChecked(True)  # Default to enabled
                checkbox.stateChanged.connect(
                    lambda state, idx=i: self.auto_y_toggled.emit(idx, state == QtCore.Qt.Checked)
                )
                self.auto_y_checkboxes[i] = checkbox
                layout.addWidget(checkbox)
            layout.addWidget(self._create_separator())
        
        # Event markers section
        if self.has_events:
            layout.addWidget(QtWidgets.QLabel("<b>Markers:</b>"))
            self.events_checkbox = QtWidgets.QCheckBox("Show Events")
            self.events_checkbox.setChecked(False)  # Default to hidden
            self.events_checkbox.stateChanged.connect(
                lambda state: self.events_toggled.emit(state == QtCore.Qt.Checked)
            )
            layout.addWidget(self.events_checkbox)
            layout.addWidget(self._create_separator())
        
        # Mask visualization mode
        layout.addWidget(QtWidgets.QLabel("<b>Mask Display:</b>"))
        self.mask_mode_group = QtWidgets.QButtonGroup()
        
        self.mask_shaded_radio = QtWidgets.QRadioButton("Shaded regions")
        self.mask_gaps_radio = QtWidgets.QRadioButton("Gaps in lines")
        
        self.mask_shaded_radio.setChecked(True)
        self.mask_mode_group.addButton(self.mask_shaded_radio)
        self.mask_mode_group.addButton(self.mask_gaps_radio)
        
        self.mask_shaded_radio.toggled.connect(
            lambda checked: self.mask_mode_changed.emit('shaded' if checked else 'gaps')
        )
        
        layout.addWidget(self.mask_shaded_radio)
        layout.addWidget(self.mask_gaps_radio)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Region selection section
        layout.addWidget(QtWidgets.QLabel("<b>Region Selection:</b>"))
        
        btn_add_region = QtWidgets.QPushButton("Add Region")
        btn_add_region.clicked.connect(self.region_added.emit)
        layout.addWidget(btn_add_region)
        
        btn_clear_regions = QtWidgets.QPushButton("Clear All")
        btn_clear_regions.clicked.connect(self.regions_cleared.emit)
        layout.addWidget(btn_clear_regions)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Accept/Cancel buttons
        btn_layout = QtWidgets.QVBoxLayout()
        
        btn_accept = QtWidgets.QPushButton("Accept")
        btn_accept.clicked.connect(self.accept_clicked.emit)
        btn_accept.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        btn_layout.addWidget(btn_accept)
        
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.cancel_clicked.emit)
        btn_cancel.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        btn_layout.addWidget(btn_cancel)
        
        layout.addLayout(btn_layout)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Set fixed width for control panel
        self.setFixedWidth(200)
    
    def _create_separator(self) -> QtWidgets.QFrame:
        """Create a horizontal separator line."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line
    
    def _on_eye_toggled(self, eye: str, enabled: bool):
        """Handle eye toggle - affects all modalities for that eye."""
        # Emit signal for eye toggle
        self.eye_toggled.emit(eye, enabled)
        
        # Update specific modalities
        for modality in self.available_modalities:
            if eye in modality:
                # Determine if modality should be visible based on both eye and variable state
                var = None
                if 'pupil' in modality:
                    var = 'pupil'
                elif '_x' in modality:
                    var = 'x'
                elif '_y' in modality:
                    var = 'y'
                
                if var and var in self.variable_checkboxes:
                    var_enabled = self.variable_checkboxes[var].isChecked()
                    self.modality_toggled.emit(modality, enabled and var_enabled)
    
    def _on_variable_toggled(self, var: str, enabled: bool):
        """Handle variable toggle - affects all modalities for that variable."""
        # Emit signal for variable toggle
        self.variable_toggled.emit(var, enabled)
        
        # Update specific modalities
        for modality in self.available_modalities:
            if (var == 'pupil' and 'pupil' in modality) or \
               (var == 'x' and '_x' in modality) or \
               (var == 'y' and '_y' in modality):
                # Determine if modality should be visible based on both eye and variable state
                eye = 'left' if 'left' in modality else 'right'
                if eye in self.eye_checkboxes:
                    eye_enabled = self.eye_checkboxes[eye].isChecked()
                    self.modality_toggled.emit(modality, enabled and eye_enabled)
    
    def get_mask_mode(self) -> str:
        """Get current mask visualization mode.
        
        Returns
        -------
        str
            'shaded' or 'gaps'
        """
        return 'shaded' if self.mask_shaded_radio.isChecked() else 'gaps'
    
    def get_modality_states(self) -> Dict[str, bool]:
        """Get current state of all modality checkboxes.
        
        Returns
        -------
        dict
            Dictionary mapping modality names to their checked state
        """
        states = {}
        for modality in self.available_modalities:
            # Determine eye and variable for this modality
            eye = 'left' if 'left' in modality else 'right'
            if 'pupil' in modality:
                var = 'pupil'
            elif '_x' in modality:
                var = 'x'
            elif '_y' in modality:
                var = 'y'
            else:
                var = None
            
            # Modality is enabled if both eye and variable are checked
            eye_enabled = self.eye_checkboxes.get(eye, None)
            var_enabled = self.variable_checkboxes.get(var, None) if var else None
            
            if eye_enabled is not None and var_enabled is not None:
                states[modality] = eye_enabled.isChecked() and var_enabled.isChecked()
            else:
                states[modality] = False
        
        return states
    
    def _on_time_unit_changed(self, index: int):
        """Handle time unit selection change."""
        unit_map = {0: 's', 1: 'ms', 2: 'min'}
        self.time_unit_changed.emit(unit_map[index])

