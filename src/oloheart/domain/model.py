"""Pure domain model for anatomy and the interactive heart aggregate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class HeartPartId(StrEnum):
    RIGHT_ATRIUM = "right_atrium"
    LEFT_ATRIUM = "left_atrium"
    RIGHT_VENTRICLE = "right_ventricle"
    LEFT_VENTRICLE = "left_ventricle"
    AORTA = "aorta"
    PULMONARY_ARTERY = "pulmonary_artery"
    VENA_CAVA = "vena_cava"
    PULMONARY_VEINS = "pulmonary_veins"
    TRICUSPID_VALVE = "tricuspid_valve"
    MITRAL_VALVE = "mitral_valve"
    AORTIC_VALVE = "aortic_valve"
    PULMONARY_VALVE = "pulmonary_valve"
    INTERVENTRICULAR_SEPTUM = "interventricular_septum"
    INTERATRIAL_SEPTUM = "interatrial_septum"
    CORONARY_ARTERIES = "coronary_arteries"
    CORONARY_VEINS = "coronary_veins"
    PERICARDIUM = "pericardium"
    EPICARDIUM = "epicardium"
    MYOCARDIUM = "myocardium"
    ENDOCARDIUM = "endocardium"
    PAPILLARY_MUSCLES = "papillary_muscles"
    CHORDAE_TENDINEAE = "chordae_tendineae"
    CARDIAC_CONDUCTION = "cardiac_conduction"
    CORONARY_SINUS = "coronary_sinus"


class AnatomyCategory(StrEnum):
    CHAMBER = "CÁMARA CARDÍACA"
    GREAT_VESSEL = "GRAN VASO"
    VALVE = "VÁLVULA"
    INTERNAL_STRUCTURE = "ESTRUCTURA INTERNA"
    CORONARY_SYSTEM = "SISTEMA CORONARIO"
    TISSUE_LAYER = "CAPA CARDÍACA"
    SUBVALVULAR_SYSTEM = "APARATO SUBVALVULAR"
    ELECTRICAL_SYSTEM = "SISTEMA DE CONDUCCIÓN"


class AnatomySystem(StrEnum):
    """Seven clinically meaningful visibility groups used by the workstation."""

    PERICARDIUM = "Pericardio"
    MYOCARDIUM = "Miocardio"
    CHAMBERS_AND_SEPTA = "Cámaras y tabiques"
    VALVES_AND_SUBVALVULAR = "Válvulas y aparato subvalvular"
    GREAT_VESSELS = "Grandes vasos"
    CORONARY_CIRCULATION = "Circulación coronaria"
    CONDUCTION_SYSTEM = "Sistema de conducción"


_PART_SYSTEM: dict[HeartPartId, AnatomySystem] = {
    HeartPartId.PERICARDIUM: AnatomySystem.PERICARDIUM,
    HeartPartId.EPICARDIUM: AnatomySystem.PERICARDIUM,
    HeartPartId.MYOCARDIUM: AnatomySystem.MYOCARDIUM,
    HeartPartId.ENDOCARDIUM: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.RIGHT_ATRIUM: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.LEFT_ATRIUM: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.RIGHT_VENTRICLE: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.LEFT_VENTRICLE: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.INTERVENTRICULAR_SEPTUM: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.INTERATRIAL_SEPTUM: AnatomySystem.CHAMBERS_AND_SEPTA,
    HeartPartId.TRICUSPID_VALVE: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.MITRAL_VALVE: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.AORTIC_VALVE: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.PULMONARY_VALVE: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.PAPILLARY_MUSCLES: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.CHORDAE_TENDINEAE: AnatomySystem.VALVES_AND_SUBVALVULAR,
    HeartPartId.AORTA: AnatomySystem.GREAT_VESSELS,
    HeartPartId.PULMONARY_ARTERY: AnatomySystem.GREAT_VESSELS,
    HeartPartId.VENA_CAVA: AnatomySystem.GREAT_VESSELS,
    HeartPartId.PULMONARY_VEINS: AnatomySystem.GREAT_VESSELS,
    HeartPartId.CORONARY_ARTERIES: AnatomySystem.CORONARY_CIRCULATION,
    HeartPartId.CORONARY_VEINS: AnatomySystem.CORONARY_CIRCULATION,
    HeartPartId.CORONARY_SINUS: AnatomySystem.CORONARY_CIRCULATION,
    HeartPartId.CARDIAC_CONDUCTION: AnatomySystem.CONDUCTION_SYSTEM,
}


def anatomy_system_for(identifier: HeartPartId) -> AnatomySystem:
    """Resolve a selectable structure to its seven-system visibility group."""

    return _PART_SYSTEM[identifier]


class CardiacPhase(StrEnum):
    ATRIAL_SYSTOLE = "Sístole auricular"
    ISOVOLUMETRIC_CONTRACTION = "Contracción isovolumétrica"
    VENTRICULAR_EJECTION = "Eyección ventricular"
    ISOVOLUMETRIC_RELAXATION = "Relajación isovolumétrica"
    RAPID_FILLING = "Llenado ventricular rápido"
    DIASTASIS = "Diástasis"


class GestureKind(StrEnum):
    NONE = "None"
    OPEN_PALM = "Open_Palm"
    CLOSED_FIST = "Closed_Fist"
    POINT = "Pointing_Up"
    THUMB_UP = "Thumb_Up"
    THUMB_DOWN = "Thumb_Down"
    VICTORY = "Victory"
    THREE_FINGERS = "Three_Fingers"
    PINKY = "Pinky"
    PINCH = "Pinch"
    FLOW = "Index_Pinky"


@dataclass(frozen=True, slots=True)
class ClinicalMetric:
    label: str
    value: str
    reference: str


@dataclass(frozen=True, slots=True)
class HeartPart:
    identifier: HeartPartId
    display_name: str
    latin_name: str
    category: AnatomyCategory
    summary: str
    function: str
    location: str
    metrics: tuple[ClinicalMetric, ...]

    @property
    def system(self) -> AnatomySystem:
        return anatomy_system_for(self.identifier)


@dataclass(frozen=True, slots=True)
class HandObservation:
    """Camera-derived spatial data that deliberately excludes image pixels."""

    gesture: GestureKind
    confidence: float
    palm_x: float
    palm_y: float
    pointer_x: float
    pointer_y: float
    span: float
    hands: int
    landmarks: tuple[tuple[float, float], ...]
    captured_at: float

    @classmethod
    def empty(cls, captured_at: float = 0.0) -> "HandObservation":
        return cls(GestureKind.NONE, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0, (), captured_at)


@dataclass(frozen=True, slots=True)
class AnatomicalElement:
    """Identity of one manipulable structure, independent of its catalog category."""

    part_id: HeartPartId
    key: str
    name: str


@dataclass(frozen=True, slots=True)
class FragmentOffset:
    """Persistent model-space displacement of one explicitly acquired element."""

    part_id: HeartPartId
    horizontal: float
    vertical: float
    depth: float
    element_key: str = ""


@dataclass(frozen=True, slots=True)
class CardiacCycleState:
    """Educational phase schedule with pressure-compatible valve states."""

    phase: CardiacPhase
    progress: float
    atrial_contraction: float
    ventricular_contraction: float
    atrioventricular_valves_open: bool
    semilunar_valves_open: bool
    coronary_perfusion: float
    electrical_progress: float

    @property
    def atrioventricular_opening(self) -> float:
        """Continuous leaflet excursion; AV valves close before ejection."""
        if not self.atrioventricular_valves_open:
            return 0.0
        if 0.58 <= self.progress < 0.63:
            return HeartExperience._smoothstep((self.progress - 0.58) / 0.05)
        if 0.085 <= self.progress < 0.12:
            return 1.0 - HeartExperience._smoothstep((self.progress - 0.085) / 0.035)
        return 1.0

    @property
    def semilunar_opening(self) -> float:
        """Semilunar cusps fold toward the vessel wall during ventricular ejection."""
        if not self.semilunar_valves_open:
            return 0.0
        opening = HeartExperience._smoothstep((self.progress - 0.20) / 0.025)
        closing = 1.0 - HeartExperience._smoothstep((self.progress - 0.455) / 0.025)
        return min(opening, closing)

    @property
    def ventricular_emptying(self) -> float:
        """Hold chamber dimensions during both isovolumetric phases."""
        if self.phase == CardiacPhase.VENTRICULAR_EJECTION:
            t = (self.progress - 0.20) / 0.28
        elif self.phase == CardiacPhase.ISOVOLUMETRIC_RELAXATION:
            return 1.0
        elif self.phase == CardiacPhase.RAPID_FILLING:
            t = 1.0 - (self.progress - 0.58) / 0.20
        else:
            return 0.0
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True, slots=True)
class HeartSnapshot:
    rotation_x: float
    rotation_y: float
    rotation_z: float
    zoom: float
    explosion: float
    selected_part: HeartPartId | None
    realistic: bool
    beating: bool
    beat_scale: float
    bpm: int
    spatial_mode: bool
    fragment_offsets: tuple[FragmentOffset, ...]
    grabbed_fragment: HeartPartId | None
    cycle: CardiacCycleState
    visible_systems: frozenset[AnatomySystem]
    interior_view: bool = False
    selected_element: AnatomicalElement | None = None
    grabbed_element: AnatomicalElement | None = None
    flow_enabled: bool = False
    flow_times: tuple[float, float, float] = (0.0, 0.0, 0.0)


class HeartExperience:
    """Aggregate root that owns all transformations and presentation-independent state."""

    MIN_ZOOM = 0.58
    MAX_ZOOM = 2.15

    def __init__(self) -> None:
        self._rotation = [-0.08, 0.34, -0.05]
        self._target_rotation = self._rotation.copy()
        self._zoom = self._target_zoom = 1.0
        self._explosion = self._target_explosion = 0.0
        self._selected_part: HeartPartId | None = None
        self._selected_element: AnatomicalElement | None = None
        self._realistic = self._beating = False
        self._spatial_mode = False
        self._interior_view = False
        self._fragment_offsets: dict[tuple[HeartPartId, str], tuple[float, float, float]] = {}
        self._grabbed_fragment: HeartPartId | None = None
        self._grabbed_element: AnatomicalElement | None = None
        self._flow_enabled = False
        self._flow_times = [0.0, 0.0, 0.0]
        self._bpm = 72
        self._beat_clock = 0.0
        self._visible_systems = set(AnatomySystem)
        self._visible_systems.discard(AnatomySystem.PERICARDIUM)

    def select(self, part: HeartPartId, element: AnatomicalElement | None = None) -> None:
        self._selected_part = part
        self._selected_element = element
        self._visible_systems.add(anatomy_system_for(part))
        if part in {
            HeartPartId.INTERATRIAL_SEPTUM, HeartPartId.INTERVENTRICULAR_SEPTUM,
            HeartPartId.ENDOCARDIUM, HeartPartId.PAPILLARY_MUSCLES,
            HeartPartId.CHORDAE_TENDINEAE, HeartPartId.CARDIAC_CONDUCTION,
            HeartPartId.MITRAL_VALVE, HeartPartId.TRICUSPID_VALVE,
            HeartPartId.AORTIC_VALVE, HeartPartId.PULMONARY_VALVE,
        }:
            self._interior_view = True

    def focus_whole_heart(self) -> None:
        self._selected_part = None
        self._selected_element = None

    def rotate_by(self, horizontal: float, vertical: float) -> None:
        self._target_rotation[1] += horizontal
        self._target_rotation[0] = self._clamp(self._target_rotation[0] + vertical, -1.35, 1.35)

    def zoom_by(self, factor: float) -> None:
        self._target_zoom = self._clamp(self._target_zoom * factor, self.MIN_ZOOM, self.MAX_ZOOM)

    def set_zoom(self, value: float) -> None:
        self._target_zoom = self._clamp(value, self.MIN_ZOOM, self.MAX_ZOOM)

    def toggle_explosion(self) -> None:
        self._target_explosion = 0.0 if self._target_explosion > 0.5 else 1.0
        if self._target_explosion < 0.5:
            self._fragment_offsets.clear()
            self._grabbed_fragment = None
            self._grabbed_element = None

    def begin_fragment_drag(self, part: HeartPartId, element: AnatomicalElement | None = None) -> bool:
        """Acquire one picked element without moving adjacent category members."""

        if self._target_explosion < 0.5 or anatomy_system_for(part) not in self._visible_systems:
            return False
        self._grabbed_fragment = part
        self._grabbed_element = element
        self._selected_part, self._selected_element = part, element
        self._fragment_offsets.setdefault((part, element.key if element else ""), (0.0, 0.0, 0.0))
        return True

    def move_grabbed_fragment(self, horizontal: float, vertical: float, depth: float) -> bool:
        """Move the acquired group in normalized camera space."""

        if self._grabbed_fragment is None or self._target_explosion < 0.5:
            return False
        key = (self._grabbed_fragment, self._grabbed_element.key if self._grabbed_element else "")
        current = self._fragment_offsets.get(key, (0.0, 0.0, 0.0))
        self._fragment_offsets[key] = (
            self._clamp(current[0] + horizontal, -0.42, 0.42),
            self._clamp(current[1] + vertical, -0.42, 0.42),
            self._clamp(current[2] + depth, -0.50, 0.50),
        )
        return True

    def release_fragment(self) -> None:
        self._grabbed_fragment = None
        self._grabbed_element = None

    def toggle_blood_flow(self) -> None:
        """Reveal the educational circulation through the anatomical section."""
        self._flow_enabled = not self._flow_enabled
        if self._flow_enabled:
            self._interior_view = True
            self._beating = self._realistic = True

    def toggle_realistic_heartbeat(self) -> None:
        self._realistic = not self._realistic
        self._beating = self._realistic

    def toggle_spatial_mode(self) -> None:
        """Toggle the distraction-free, full-canvas anatomical workspace."""

        self._spatial_mode = not self._spatial_mode

    def toggle_interior_view(self) -> None:
        """Change the anatomical section without changing selection or placement."""

        self._interior_view = not self._interior_view

    def toggle_system(self, system: AnatomySystem) -> None:
        """Toggle one anatomical system without leaking presentation state."""

        if system in self._visible_systems:
            self._visible_systems.remove(system)
            if self._selected_part is not None and anatomy_system_for(self._selected_part) == system:
                self._selected_part = None
                self._selected_element = None
            if self._grabbed_fragment is not None and anatomy_system_for(self._grabbed_fragment) == system:
                self._grabbed_fragment = None
                self._grabbed_element = None
        else:
            self._visible_systems.add(system)

    def reset_view(self) -> None:
        self._target_rotation[:] = [-0.08, 0.34, -0.05]
        self._target_zoom = 1.0
        self._target_explosion = 0.0
        self._selected_part = None
        self._spatial_mode = False
        self._selected_element = self._grabbed_element = None
        self._flow_enabled = False
        self._flow_times[:] = [0.0, 0.0, 0.0]
        self._interior_view = False
        self._fragment_offsets.clear()
        self._grabbed_fragment = None
        self._visible_systems = set(AnatomySystem)
        self._visible_systems.discard(AnatomySystem.PERICARDIUM)

    def tick(self, delta_seconds: float) -> None:
        delta = self._clamp(delta_seconds, 0.0, 0.08)
        smoothing = 1.0 - math.exp(-8.5 * delta)
        for index in range(3):
            self._rotation[index] += (self._target_rotation[index] - self._rotation[index]) * smoothing
        self._zoom += (self._target_zoom - self._zoom) * smoothing
        self._explosion += (self._target_explosion - self._explosion) * smoothing
        if self._beating:
            # Small substeps avoid carrying flow across a closed-valve phase
            # when a slow display frame straddles two phases of the cycle.
            steps = max(1, math.ceil(delta * 240))
            dt = delta / steps
            for _ in range(steps):
                self._beat_clock = (self._beat_clock + dt * self._bpm / 60.0) % 1.0
                cycle = self._cycle_state()
                for index, allowed in enumerate((True, cycle.atrioventricular_valves_open,
                                                 cycle.semilunar_valves_open)):
                    if allowed:
                        self._flow_times[index] += dt

    def snapshot(self) -> HeartSnapshot:
        offsets = tuple(
            FragmentOffset(part, *values, key)
            for (part, key), values in sorted(self._fragment_offsets.items())
        )
        cycle = self._cycle_state()
        return HeartSnapshot(*self._rotation, self._zoom, self._explosion, self._selected_part,
                             self._realistic, self._beating, self._beat_scale(cycle), self._bpm,
                             self._spatial_mode, offsets, self._grabbed_fragment, cycle,
                             frozenset(self._visible_systems), self._interior_view,
                             self._selected_element, self._grabbed_element,
                             self._flow_enabled, tuple(self._flow_times))

    def _beat_scale(self, cycle: CardiacCycleState) -> float:
        if not self._beating:
            return 1.0
        return 1.0 + cycle.atrial_contraction * 0.012 + cycle.ventricular_contraction * 0.045

    def _cycle_state(self) -> CardiacCycleState:
        if not self._beating:
            return CardiacCycleState(
                CardiacPhase.DIASTASIS, 0.0, 0.0, 0.0, True, False, 0.85, 0.0
            )

        progress = self._beat_clock
        atrial = math.exp(-((progress - 0.065) / 0.045) ** 2)
        if progress < 0.12:
            phase = CardiacPhase.ATRIAL_SYSTOLE
            ventricular = 0.0
            av_open, semilunar_open = True, False
        elif progress < 0.20:
            phase = CardiacPhase.ISOVOLUMETRIC_CONTRACTION
            ventricular = self._smoothstep((progress - 0.12) / 0.08)
            av_open, semilunar_open = False, False
        elif progress < 0.48:
            phase = CardiacPhase.VENTRICULAR_EJECTION
            ventricular = 1.0 - 0.12 * self._smoothstep((progress - 0.20) / 0.28)
            av_open, semilunar_open = False, True
        elif progress < 0.58:
            phase = CardiacPhase.ISOVOLUMETRIC_RELAXATION
            ventricular = 0.88 * (1.0 - self._smoothstep((progress - 0.48) / 0.10))
            av_open, semilunar_open = False, False
        elif progress < 0.78:
            phase = CardiacPhase.RAPID_FILLING
            ventricular = 0.0
            av_open, semilunar_open = True, False
        else:
            phase = CardiacPhase.DIASTASIS
            ventricular = 0.0
            av_open, semilunar_open = True, False

        coronary = 0.22 if semilunar_open else 0.92
        if phase in {CardiacPhase.ISOVOLUMETRIC_CONTRACTION,
                     CardiacPhase.ISOVOLUMETRIC_RELAXATION}:
            coronary = 0.48
        electrical = min(1.0, progress / 0.24) if progress < 0.24 else 0.0
        return CardiacCycleState(
            phase,
            progress,
            atrial,
            ventricular,
            av_open,
            semilunar_open,
            coronary,
            electrical,
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = HeartExperience._clamp(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def build_anatomy_catalog() -> tuple[HeartPart, ...]:
    """Return educational anatomy in the same order used by the native navigator."""

    m = ClinicalMetric
    p = HeartPart
    c = AnatomyCategory
    i = HeartPartId
    return (
        p(i.RIGHT_ATRIUM, "Aurícula derecha", "Atrium dextrum", c.CHAMBER,
          "Cámara superior que recibe la sangre venosa sistémica.",
          "Conduce la sangre desoxigenada hacia el ventrículo derecho.",
          "Superior y derecha; comunica con las venas cavas y la válvula tricúspide.",
          (m("PRESIÓN MEDIA", "2–6 mmHg", "Referencia fisiológica"), m("PARED", "≈2 mm", "Pared auricular"))),
        p(i.LEFT_ATRIUM, "Aurícula izquierda", "Atrium sinistrum", c.CHAMBER,
          "Cámara posterior que recibe sangre oxigenada de los pulmones.",
          "Completa el llenado del ventrículo izquierdo durante la diástole.",
          "Posterior y superior; recibe normalmente cuatro venas pulmonares.",
          (m("PRESIÓN MEDIA", "6–12 mmHg", "Referencia fisiológica"), m("VOLUMEN", "≈22–50 ml", "Variable con el ciclo"))),
        p(i.RIGHT_VENTRICLE, "Ventrículo derecho", "Ventriculus dexter", c.CHAMBER,
          "Cámara anterior de pared delgada y geometría semilunar.",
          "Impulsa sangre venosa hacia la circulación pulmonar.",
          "Anterior al ventrículo izquierdo; continúa con el tracto de salida pulmonar.",
          (m("PARED", "3–5 mm", "Adulto, aproximado"), m("PRESIÓN", "15–30 / 2–8", "mmHg"))),
        p(i.LEFT_VENTRICLE, "Ventrículo izquierdo", "Ventriculus sinister", c.CHAMBER,
          "Cámara muscular que forma el ápex y gran parte del borde izquierdo.",
          "Genera la presión para impulsar sangre a la circulación sistémica.",
          "Inferior a la aurícula izquierda y conectado a la aorta.",
          (m("PARED", "8–15 mm", "Diástole, aproximado"), m("FRACCIÓN DE EYECCIÓN", "55–70%", "Rango habitual"), m("VOLUMEN DIASTÓLICO", "≈120 ml", "Valor ilustrativo"))),
        p(i.AORTA, "Aorta", "Aorta", c.GREAT_VESSEL,
          "La arteria de mayor calibre del organismo.", "Distribuye sangre oxigenada desde el ventrículo izquierdo.",
          "Nace en la raíz aórtica, asciende y forma un arco antes de descender.",
          (m("DIÁMETRO RAÍZ", "≈30 mm", "Depende de talla y edad"), m("PRESIÓN", "120/80 mmHg", "Ejemplo educativo"))),
        p(i.PULMONARY_ARTERY, "Arteria pulmonar", "Truncus pulmonalis", c.GREAT_VESSEL,
          "Tronco vascular que se bifurca hacia ambos pulmones.", "Transporta sangre desoxigenada para el intercambio gaseoso.",
          "Emerge del ventrículo derecho y pasa delante de la aorta.",
          (m("DIÁMETRO", "≈25–30 mm", "Tronco principal"), m("PRESIÓN", "15–30 / 4–12", "mmHg"))),
        p(i.VENA_CAVA, "Venas cavas", "Venae cavae", c.GREAT_VESSEL,
          "Grandes venas que completan el retorno sistémico.", "Llevan sangre a la aurícula derecha.",
          "Entran desde arriba y desde abajo.", (m("FLUJO", "Continuo y fásico", "Modulado por respiración"),)),
        p(i.PULMONARY_VEINS, "Venas pulmonares", "Venae pulmonales", c.GREAT_VESSEL,
          "Vasos de retorno de la circulación pulmonar.", "Llevan sangre oxigenada a la aurícula izquierda.",
          "Desembocan en su pared posterior.", (m("NÚMERO HABITUAL", "4", "Dos por pulmón"),)),
        p(i.TRICUSPID_VALVE, "Válvula tricúspide", "Valva atrioventricularis dextra", c.VALVE,
          "Válvula auriculoventricular derecha de tres valvas.", "Evita reflujo hacia la aurícula derecha.",
          "Entre la aurícula y el ventrículo derechos.", (m("VALVAS", "3", "Anterior, posterior y septal"),)),
        p(i.MITRAL_VALVE, "Válvula mitral", "Valva atrioventricularis sinistra", c.VALVE,
          "Válvula auriculoventricular izquierda de dos valvas.", "Sella el ventrículo durante la sístole.",
          "Entre la aurícula y el ventrículo izquierdos.", (m("VALVAS", "2", "Anterior y posterior"), m("ÁREA", "4–6 cm²", "Referencia anatómica"))),
        p(i.AORTIC_VALVE, "Válvula aórtica", "Valva aortae", c.VALVE,
          "Válvula semilunar de la salida ventricular izquierda.", "Evita retorno desde la aorta.",
          "En la raíz aórtica.", (m("CÚSPIDES", "3", "Derecha, izquierda y no coronaria"),)),
        p(i.PULMONARY_VALVE, "Válvula pulmonar", "Valva trunci pulmonalis", c.VALVE,
          "Válvula semilunar del tracto de salida derecho.", "Evita retorno desde el tronco pulmonar.",
          "Entre el infundíbulo y el tronco pulmonar.", (m("CÚSPIDES", "3", "Semilunares"),)),
        p(i.INTERVENTRICULAR_SEPTUM, "Tabique interventricular", "Septum interventriculare", c.INTERNAL_STRUCTURE,
          "Pared muscular y membranosa que separa ambos ventrículos.", "Separa circuitos y participa en la conducción.",
          "Desde la base hasta el ápex.", (m("COMPONENTE", "Muscular + membranoso", "Estructura mixta"),)),
        p(i.INTERATRIAL_SEPTUM, "Tabique interauricular", "Septum interatriale", c.INTERNAL_STRUCTURE,
          "Pared que separa ambas aurículas e incluye la fosa oval en su cara derecha.",
          "Mantiene separados los retornos venosos sistémico y pulmonar tras el nacimiento.",
          "Entre las aurículas, posterior a la raíz aórtica.",
          (m("REFERENCIA", "Fosa oval", "Remanente del foramen oval"),)),
        p(i.CORONARY_ARTERIES, "Arterias coronarias", "Arteriae coronariae", c.CORONARY_SYSTEM,
          "Red arterial visible sobre la superficie cardíaca.", "Aporta oxígeno y nutrientes al miocardio.",
          "Nace en la raíz aórtica y recorre los surcos epicárdicos.",
          (m("ORIGEN", "Senos aórticos", "Coronaria derecha e izquierda"), m("FLUJO", "Predomina en diástole", "Especialmente a la izquierda"))),
        p(i.CORONARY_VEINS, "Venas cardíacas", "Venae cordis", c.CORONARY_SYSTEM,
          "Red venosa formada principalmente por las venas cardíacas magna, media y menor.",
          "Recoge la sangre desoxigenada del miocardio y la dirige al seno coronario.",
          "Acompaña a las arterias en los surcos interventriculares y coronario.",
          (m("DRENAJE", "Seno coronario", "Retorno principal a aurícula derecha"),)),
        p(i.PERICARDIUM, "Pericardio", "Pericardium", c.TISSUE_LAYER,
          "Saco protector de doble pared que rodea al corazón.",
          "Limita la sobredistensión y reduce la fricción durante el movimiento.",
          "Envuelve el corazón y contiene una fina cavidad con líquido pericárdico.",
          (m("ORGANIZACIÓN", "Fibroso + seroso", "Capa protectora"),
           m("CAVIDAD", "≈20–25 mL", "Líquido seroso fisiológico"))),
        p(i.EPICARDIUM, "Epicardio", "Epicardium", c.TISSUE_LAYER,
          "Capa externa de la pared cardíaca y hoja visceral del pericardio seroso.",
          "Protege la superficie y aloja vasos coronarios y tejido adiposo.",
          "Recubre externamente aurículas, ventrículos y surcos cardíacos.",
          (m("POSICIÓN", "Capa externa", "Pared cardíaca"),)),
        p(i.MYOCARDIUM, "Miocardio", "Myocardium", c.TISSUE_LAYER,
          "Capa muscular principal organizada en haces helicoidales entrecruzados.",
          "Genera la contracción que impulsa la sangre.",
          "Entre epicardio y endocardio; es más grueso en el ventrículo izquierdo.",
          (m("CAPAS FUNCIONALES", "3 orientaciones", "Fibras helicoidales"),
           m("VI / VD", "≈2–3× más grueso", "Relación ilustrativa"))),
        p(i.ENDOCARDIUM, "Endocardio", "Endocardium", c.TISSUE_LAYER,
          "Revestimiento liso de las cámaras y válvulas cardíacas.",
          "Reduce la resistencia al flujo y separa la sangre del miocardio.",
          "Tapiza cavidades, tabiques y superficies valvulares.",
          (m("POSICIÓN", "Capa interna", "Superficie lisa"),)),
        p(i.PAPILLARY_MUSCLES, "Músculos papilares", "Musculi papillares", c.SUBVALVULAR_SYSTEM,
          "Proyecciones musculares cónicas dentro de ambos ventrículos.",
          "Tensan las cuerdas tendinosas durante la sístole.",
          "Nacen de la pared ventricular y se conectan a las valvas auriculoventriculares.",
          (m("FUNCIÓN", "Estabilidad valvular", "Durante sístole"),)),
        p(i.CHORDAE_TENDINEAE, "Cuerdas tendinosas", "Chordae tendineae", c.SUBVALVULAR_SYSTEM,
          "Fibras resistentes que conectan valvas y músculos papilares.",
          "Evitan que las valvas mitral y tricúspide prolapsen hacia las aurículas.",
          "Se distribuyen como abanicos dentro de los ventrículos.",
          (m("ASPECTO", "Cordones fibrosos", "Ramificación fina"),)),
        p(i.CARDIAC_CONDUCTION, "Sistema de conducción", "Systema conducens cordis", c.ELECTRICAL_SYSTEM,
          "Red especializada que coordina aurículas y ventrículos.",
          "Inicia y distribuye el impulso eléctrico de cada latido.",
          "Incluye nodo SA, nodo AV, haz de His, ramas y fibras de Purkinje.",
          (m("NODO SA", "60–100 impulsos/min", "Marcapasos fisiológico"),
           m("RETARDO AV", "≈0.1 s", "Llenado ventricular"))),
        p(i.CORONARY_SINUS, "Seno coronario", "Sinus coronarius", c.CORONARY_SYSTEM,
          "Principal colector venoso del músculo cardíaco.",
          "Devuelve la sangre del miocardio a la aurícula derecha.",
          "Recorre la cara posterior en el surco auriculoventricular.",
          (m("DESEMBOCADURA", "Aurícula derecha", "Retorno venoso cardíaco"),)),
    )
