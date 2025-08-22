import threading
import math
import numpy as np
import sounddevice as sd
import tkinter as tk
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# --------------------------------
# Конфигурация
# --------------------------------
SAMPLE_RATE = 44100
BLOCKSIZE = 1024

A4_KEY_NUMBER = 49
A4_FREQ = 440.0

# Базовая длительность ноты при клике мышью без удержания сустейна (сек)
NOTE_DURATION = 0.7

# ADSR-параметры (сек)
ATTACK = 0.02
DECAY = 0.10
SUSTAIN_LEVEL = 0.75
RELEASE = 0.35

# Гармоники (отношение частоты, амплитуда)
HARMONICS = [
    (1.0, 1.0),
    (2.0, 0.6),
    (3.0, 0.35),
    (4.0, 0.25),
    (5.0, 0.18),
    (6.0, 0.12),
    (7.0, 0.09),
    (8.0, 0.07),
]

# Глобальная громкость частичных и мастера
PARTIALS_AMPLITUDE = 0.35
MASTER_GAIN = 0.95

# Софт-клиппинг (антиклиппинг)
SOFT_CLIP_DRIVE = 1.6  # 1.2..2.0 — мягкая сатурация

# Полифония
MAX_POLYPHONY = 16

# Клавиатура (диапазон)
START_KEY = 36     # C3
NUM_OCTAVES = 2    # количество октав

# Рисование клавиатуры
WHITE_W, WHITE_H = 50, 160
BLACK_W, BLACK_H = 32, 100
BLACK_OFFSET = 0.68  # смещение чёрных между белыми (в долях ширины белой)

# ПК-клавиатура: 25 клавиш (2 октавы + верхний C)
# ВАЖНО: в Tk keysym для запятой — 'comma'
PC_KEYS_SEQUENCE = [
    'z', 's', 'x', 'd', 'c', 'v', 'g', 'b', 'h', 'n', 'j', 'm', 'comma',
    'q', '2', 'w', '3', 'e', 'r', '5', 't', '6', 'y', '7', 'u'
]

TWO_PI = 2.0 * math.pi


# --------------------------------
# Вспомогательные функции
# --------------------------------
def key_to_freq(key_number: int) -> Optional[float]:
    if not 1 <= key_number <= 88:
        return None
    n = key_number - A4_KEY_NUMBER
    return A4_FREQ * (2 ** (n / 12.0))


def soft_clip(x: np.ndarray, drive: float = SOFT_CLIP_DRIVE) -> np.ndarray:
    # Мягкая сатурация: нормализованный tanh
    return np.tanh(drive * x) / np.tanh(drive)


@dataclass
class KeyInfo:
    full: str
    freq: float
    is_black: bool
    white_index: Optional[int] = None
    white_before_index: Optional[int] = None
    canvas_id: Optional[int] = None


def build_keys(start_key=36, num_octaves=2) -> List[KeyInfo]:
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    num_keys = num_octaves * 12 + 1

    keys: List[KeyInfo] = []
    white_count = 0
    for i in range(num_keys):
        key_number = start_key + i
        note_index = key_number % 12
        note_name = note_names[note_index]
        is_black = ('#' in note_name)
        octave = (key_number + 8) // 12
        full_name = f"{note_name}{octave}"
        freq = key_to_freq(key_number)

        if not is_black:
            keys.append(KeyInfo(full=full_name, freq=freq, is_black=False, white_index=white_count))
            white_count += 1
        else:
            keys.append(KeyInfo(full=full_name, freq=freq, is_black=True, white_before_index=max(0, white_count - 1)))
    return keys


@dataclass
class ActiveNote:
    key_full: str
    freq: float
    start_frame: int
    velocity: float = 0.85
    phases: List[float] = field(default_factory=list)
    release_frame: Optional[int] = None
    release_amp: Optional[float] = None
    auto_release_frame: Optional[int] = None
    source_tag: Optional[str] = None  # например, 'pc:z' или 'mouse'

    def __post_init__(self):
        if not self.phases:
            self.phases = [0.0 for _ in HARMONICS]


# ----------- ADSR -----------
def env_level_held_scalar(t: float) -> float:
    if t < 0:
        return 0.0
    if ATTACK > 0 and t < ATTACK:
        return t / ATTACK
    t2 = t - ATTACK
    if DECAY > 0 and t2 < DECAY:
        return 1.0 - (1.0 - SUSTAIN_LEVEL) * (t2 / DECAY)
    return SUSTAIN_LEVEL


def env_held_vector(t: np.ndarray) -> np.ndarray:
    env = np.full_like(t, SUSTAIN_LEVEL, dtype=np.float64)
    m_neg = t < 0
    if np.any(m_neg):
        env[m_neg] = 0.0
    if ATTACK > 0:
        m_a = (t >= 0) & (t < ATTACK)
        if np.any(m_a):
            env[m_a] = t[m_a] / ATTACK
    else:
        m_a0 = (t >= 0) & (t < DECAY)
        if np.any(m_a0):
            env[m_a0] = 1.0
    if DECAY > 0:
        m_d = (t >= ATTACK) & (t < ATTACK + DECAY)
        if np.any(m_d):
            env[m_d] = 1.0 - (1.0 - SUSTAIN_LEVEL) * ((t[m_d] - ATTACK) / DECAY)
    return env


def env_vector_for_note(note: ActiveNote, frames_idx: np.ndarray) -> np.ndarray:
    t = (frames_idx - note.start_frame) / SAMPLE_RATE
    env = env_held_vector(t)
    if note.release_frame is not None:
        rel_t = (frames_idx - note.release_frame) / SAMPLE_RATE
        base = note.release_amp if note.release_amp is not None else env_level_held_scalar(
            (note.release_frame - note.start_frame) / SAMPLE_RATE
        )
        m_rel = rel_t >= 0.0
        if np.any(m_rel):
            env_rel = base * (1.0 - (rel_t / RELEASE))
            env_rel[rel_t >= RELEASE] = 0.0
            env[m_rel] = np.maximum(0.0, env_rel[m_rel])
    return env


class PianoApp:
    def __init__(self):
        # GUI
        self.root = tk.Tk()
        self.root.title("Python Piano — RT Synth (Sustain, Velocity, Anti-Clip, PC Keyboard)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        info = (
            "Клик по клавишам — звук (velocity зависит от высоты клика). "
            "ПК-клавиатура: z s x d c v g b h n j m comma q 2 w 3 e r 5 t 6 y 7 u. "
            "Пробел — Sustain. Стрелки ↑/↓ — velocity для ПК."
        )
        tk.Label(self.root, text=info, fg="#555", justify="left", wraplength=800).pack(padx=8, pady=(8, 2), anchor="w")

        self.status_var = tk.StringVar()
        self.status_label = tk.Label(self.root, textvariable=self.status_var)
        self.status_label.pack(padx=8, pady=(2, 8), anchor="w")

        self.keys_data: List[KeyInfo] = build_keys(START_KEY, NUM_OCTAVES)
        white_count = sum(1 for k in self.keys_data if not k.is_black)
        width = white_count * WHITE_W
        height = WHITE_H
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#ddd", highlightthickness=0)
        self.canvas.pack(padx=8, pady=8)
        self.canvas.focus_set()

        self.draw_keyboard()

        # Состояние
        self.sustain_event = threading.Event()  # True -> sustain ON
        self.notes_lock = threading.Lock()
        self.active_notes: List[ActiveNote] = []
        self.global_frame = 0  # счётчик воспроизведённых сэмплов
        self.running = True

        self.current_pc_velocity = 0.85  # регулируется стрелками
        self.held_pc_notes: Dict[str, ActiveNote] = {}  # keysym -> ActiveNote

        # Карта ПК-клавиш -> индекс ноты
        self.pc_key_to_index = self.build_pc_keymap()

        # Аудио
        try:
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,                  # при проблемах смените на 2
                dtype='float32',
                blocksize=BLOCKSIZE,
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            tk.messagebox.showerror("Audio error", f"Не удалось открыть аудио-устройство:\n{e}")
            raise

        # События
        self.root.bind("<KeyPress-space>", self.on_space_down)
        self.root.bind("<KeyRelease-space>", self.on_space_up)

        self.root.bind("<KeyPress-Up>", self.on_vel_up)
        self.root.bind("<KeyPress-Down>", self.on_vel_down)
        self.root.bind("<KeyPress>", self.on_key_down)
        self.root.bind("<KeyRelease>", self.on_key_up)

        # Начальный статус
        self.set_status()

    # -------------------- GUI --------------------
    def draw_keyboard(self):
        # Белые
        for k in self.keys_data:
            if not k.is_black:
                x = k.white_index * WHITE_W
                k.canvas_id = self.canvas.create_rectangle(
                    x, 0, x + WHITE_W, WHITE_H, fill="#ffffff", outline="#000000"
                )
                self.canvas.tag_bind(k.canvas_id, "<Button-1>", lambda e, kk=k: self.handle_key_click(kk, e))

        # Чёрные
        for k in self.keys_data:
            if k.is_black:
                left_white = k.white_before_index
                x = (left_white + BLACK_OFFSET) * WHITE_W - (BLACK_W / 2)
                k.canvas_id = self.canvas.create_rectangle(
                    x, 0, x + BLACK_W, BLACK_H, fill="#333333", outline="#000000"
                )
                self.canvas.tag_bind(k.canvas_id, "<Button-1>", lambda e, kk=k: self.handle_key_click(kk, e))

    def set_status(self):
        # важно: не брать здесь lock, чтобы не создать дедлок
        poly = len(self.active_notes)
        sustain = "ON" if self.sustain_event.is_set() else "OFF"
        self.status_var.set(
            f"Sustain: {sustain} | Velocity(PC): {self.current_pc_velocity:.2f} | Poly: {poly}/{MAX_POLYPHONY}"
        )

    def handle_key_click(self, k: KeyInfo, event=None):
        # Velocity от высоты клика (сверху громче)
        if event is not None:
            y = float(event.y)
            y = min(max(y, 0.0), WHITE_H)
            velocity = 0.3 + 0.7 * (1.0 - (y / WHITE_H))
        else:
            velocity = 0.7

        self.canvas.focus_set()
        self.start_note(k, velocity=velocity, source_tag="mouse", auto_release=not self.sustain_event.is_set())

    # -------------------- ПК-клавиатура --------------------
    def build_pc_keymap(self) -> Dict[str, int]:
        mapping = {}
        n = min(len(PC_KEYS_SEQUENCE), len(self.keys_data))
        for i in range(n):
            mapping[PC_KEYS_SEQUENCE[i]] = i
        return mapping

    def on_key_down(self, event):
        ks = event.keysym.lower()
        if ks == "space" or ks in ("up", "down"):
            return  # уже обрабатывается отдельными хендлерами

        idx = self.pc_key_to_index.get(ks)
        if idx is None:
            return

        # Не запускать повторно при авторепите
        if ks in self.held_pc_notes:
            return

        k = self.keys_data[idx]
        if k.freq is None:
            return

        note = self.start_note(k, velocity=self.current_pc_velocity, source_tag=f"pc:{ks}", auto_release=False)
        if note is not None:
            self.held_pc_notes[ks] = note

    def on_key_up(self, event):
        ks = event.keysym.lower()
        if ks == "space" or ks in ("up", "down"):
            return

        note = self.held_pc_notes.pop(ks, None)
        if note is None:
            return

        # Если сустейн активен — не отпускаем ноту (как педаль)
        if self.sustain_event.is_set():
            note.auto_release_frame = None
        else:
            self.release_note(note, self.global_frame)

    def on_vel_up(self, event=None):
        self.current_pc_velocity = min(1.0, self.current_pc_velocity + 0.05)
        self.set_status()

    def on_vel_down(self, event=None):
        self.current_pc_velocity = max(0.2, self.current_pc_velocity - 0.05)
        self.set_status()

    # -------------------- Sustain --------------------
    def on_space_down(self, event=None):
        if not self.sustain_event.is_set():
            self.sustain_event.set()
            # отменяем авто-релиз для уже звучащих
            with self.notes_lock:
                for n in self.active_notes:
                    n.auto_release_frame = None
            self.set_status()

    def on_space_up(self, event=None):
        if self.sustain_event.is_set():
            self.sustain_event.clear()
            self.release_all_notes()
            self.set_status()

    # -------------------- Закрытие --------------------
    def on_close(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        self.root.destroy()

    # -------------------- Логика нот --------------------
    def start_note(self, k: KeyInfo, velocity: float = 0.85, source_tag: Optional[str] = None, auto_release: bool = False) -> Optional[ActiveNote]:
        if k.freq is None:
            return None

        # Критическая секция минимальна
        with self.notes_lock:
            # Voice stealing
            if len(self.active_notes) >= MAX_POLYPHONY:
                cur_frame = self.global_frame
                envs = [(i, self._env_level_safe(n, cur_frame)) for i, n in enumerate(self.active_notes)]
                idx_min = min(envs, key=lambda p: p[1])[0]
                self.release_note(self.active_notes[idx_min], cur_frame)

            start_frame = self.global_frame
            note = ActiveNote(
                key_full=k.full,
                freq=k.freq,
                start_frame=start_frame,
                velocity=float(max(0.0, min(1.0, velocity))),
                source_tag=source_tag or "unknown"
            )
            if auto_release and not self.sustain_event.is_set():
                rel_delay = max(0.0, NOTE_DURATION - RELEASE)
                note.auto_release_frame = start_frame + int(rel_delay * SAMPLE_RATE)
            self.active_notes.append(note)

        # Важно: обновлять статус уже после выхода из lock
        self.set_status()
        return note

    def _env_level_safe(self, note: ActiveNote, frame: int) -> float:
        # Оценка env в один момент (для voice stealing)
        t = (frame - note.start_frame) / SAMPLE_RATE
        if note.release_frame is None:
            return env_level_held_scalar(t)
        if frame < note.release_frame:
            return env_level_held_scalar(t)
        rel_t = (frame - note.release_frame) / SAMPLE_RATE
        base = note.release_amp if note.release_amp is not None else env_level_held_scalar(
            (note.release_frame - note.start_frame) / SAMPLE_RATE
        )
        if rel_t >= RELEASE:
            return 0.0
        return max(0.0, base * (1.0 - (rel_t / RELEASE)))

    def release_note(self, note: ActiveNote, frame_now: int):
        if note.release_frame is not None:
            return
        note.release_frame = frame_now
        t_rel = (frame_now - note.start_frame) / SAMPLE_RATE
        note.release_amp = env_level_held_scalar(t_rel)

    def release_all_notes(self):
        # Берём снимок списка под lock, отпускаем lock и делаем работу вне
        with self.notes_lock:
            notes_snapshot = list(self.active_notes)
            frame_now = self.global_frame
        for n in notes_snapshot:
            if n.release_frame is None:
                self.release_note(n, frame_now)

    # -------------------- Аудио коллбэк --------------------
    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            # Можно логировать underrun/overrun
            pass

        frames_idx = np.arange(frames, dtype=np.int64)

        # Берём снимок состояния под коротким lock
        with self.notes_lock:
            notes_snapshot = list(self.active_notes)
            gf = self.global_frame

        buf = np.zeros(frames, dtype=np.float64)
        to_remove_objs = []

        # Обрабатываем события авторелиза, попадающие в текущий блок (без lock)
        for note in notes_snapshot:
            if (note.auto_release_frame is not None
                and not self.sustain_event.is_set()
                and note.release_frame is None
                and note.auto_release_frame >= gf
                and note.auto_release_frame < gf + frames):
                self.release_note(note, note.auto_release_frame)

        # Микширование нот (векторно)
        for note in notes_snapshot:
            # Если релиз завершился до начала блока — отметим на удаление
            if (note.release_frame is not None
                and gf >= note.release_frame + int(RELEASE * SAMPLE_RATE) + 1):
                to_remove_objs.append(note)
                continue

            idx = gf + frames_idx
            env = env_vector_for_note(note, idx)
            if not np.any(env > 0.0):
                if note.release_frame is not None and (gf + frames) >= note.release_frame + int(RELEASE * SAMPLE_RATE) + 1:
                    to_remove_objs.append(note)
                continue

            sig = np.zeros(frames, dtype=np.float64)
            for h_idx, (ratio, h_amp) in enumerate(HARMONICS):
                phi0 = note.phases[h_idx]
                inc = TWO_PI * (note.freq * ratio) / SAMPLE_RATE
                ph = phi0 + inc * frames_idx
                sig += np.sin(ph) * (h_amp * PARTIALS_AMPLITUDE)
                note.phases[h_idx] = (phi0 + inc * frames) % TWO_PI

            sig *= env * note.velocity
            buf += sig

        # Мастер и софт-клип
        buf *= MASTER_GAIN
        buf = soft_clip(buf, SOFT_CLIP_DRIVE)

        # Синхронизация: удаление нот и продвижение счётчика — под lock
        with self.notes_lock:
            # Удаляем завершённые по объектной идентичности
            if to_remove_objs:
                self.active_notes = [n for n in self.active_notes if n not in to_remove_objs]
            self.global_frame = gf + frames

        # Выход
        outdata[:, 0] = buf.astype(np.float32)

    # -------------------- Запуск --------------------
    def run(self):
        self.set_status()
        self.root.mainloop()


if __name__ == "__main__":
    app = PianoApp()
    print("🎹 Python Piano — RT Synth (Sustain, Velocity, Anti-Clip, PC Keyboard)")
    print("Подсказка: раскладка клавиатуры — English (US).")
    app.run()