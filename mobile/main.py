"""
AI Attendance — Kivy + KivyMD 2.0 Android App

A mobile client that connects to the FastAPI server for face recognition,
registration, sample addition, and profile management.
"""

import base64
import io
import os
import threading

import requests
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.lang import Builder
from kivy.metrics import dp

from kivymd.app import MDApp
from kivymd.uix.appbar import MDTopAppBar, MDTopAppBarTitle
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDButton, MDButtonText
from kivymd.uix.dialog import (
    MDDialog,
    MDDialogHeadlineText,
    MDDialogSupportingText,
    MDDialogButtonContainer,
    MDDialogContentContainer,
)
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.list import (
    MDList,
    MDListItem,
    MDListItemHeadlineText,
    MDListItemSupportingText,
    MDListItemTertiaryText,
    MDListItemLeadingIcon,
)
from kivymd.uix.navigationbar import (
    MDNavigationBar,
    MDNavigationItem,
    MDNavigationItemIcon,
    MDNavigationItemLabel,
)
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.textfield import MDTextField, MDTextFieldHintText

# ---------------------------------------------------------------------------
# Server configuration — change to your PC's local IP
# ---------------------------------------------------------------------------
SERVER_URL = "http://10.245.56.14:8000"


def api_url(path: str) -> str:
    return f"{SERVER_URL}{path}"


# ---------------------------------------------------------------------------
# KV Layout
# ---------------------------------------------------------------------------
KV = """
MDScreen:
    md_bg_color: app.theme_cls.backgroundColor

    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            MDTopAppBarTitle:
                text: "AI Attendance"

        MDScreenManager:
            id: screen_manager

            MDScreen:
                name: "attendance"
                AttendanceScreen:
                    id: attendance_screen

            MDScreen:
                name: "register"
                RegisterScreen:
                    id: register_screen

            MDScreen:
                name: "samples"
                SamplesScreen:
                    id: samples_screen

            MDScreen:
                name: "profiles"
                ProfilesScreen:
                    id: profiles_screen

        MDNavigationBar:
            on_switch_tabs: app.on_tab_switch(*args)

            MDNavigationItem:
                MDNavigationItemIcon:
                    icon: "camera"
                MDNavigationItemLabel:
                    text: "Attendance"

            MDNavigationItem:
                MDNavigationItemIcon:
                    icon: "account-plus"
                MDNavigationItemLabel:
                    text: "Register"

            MDNavigationItem:
                MDNavigationItemIcon:
                    icon: "image-plus"
                MDNavigationItemLabel:
                    text: "Samples"

            MDNavigationItem:
                MDNavigationItemIcon:
                    icon: "account-group"
                MDNavigationItemLabel:
                    text: "Profiles"


<AttendanceScreen>:
    orientation: "vertical"
    padding: "16dp"
    spacing: "12dp"

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.pick_image()
        MDButtonText:
            text: "Select Group Photo"

    MDLabel:
        id: att_status
        text: "Select a group photo to begin"
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1] + dp(16)

    Image:
        id: att_image
        size_hint_y: 0.35
        allow_stretch: True

    ScrollView:
        MDList:
            id: att_results


<RegisterScreen>:
    orientation: "vertical"
    padding: "16dp"
    spacing: "12dp"

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.pick_image()
        MDButtonText:
            text: "Select Group Photo"

    MDLabel:
        id: reg_status
        text: "Select a photo to detect unregistered faces"
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1] + dp(16)

    ScrollView:
        size_hint_y: 0.5
        MDList:
            id: reg_faces

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.save_registrations()
        MDButtonText:
            text: "Save Registrations"


<SamplesScreen>:
    orientation: "vertical"
    padding: "16dp"
    spacing: "12dp"

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.pick_images()
        MDButtonText:
            text: "Select Images"

    MDLabel:
        id: samp_status
        text: "Select images to add face samples"
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1] + dp(16)

    ScrollView:
        MDList:
            id: samp_results

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.commit_samples()
        MDButtonText:
            text: "Add Selected Samples"


<ProfilesScreen>:
    orientation: "vertical"
    padding: "16dp"
    spacing: "12dp"

    MDButton:
        style: "filled"
        pos_hint: {"center_x": 0.5}
        on_release: root.load_profiles()
        MDButtonText:
            text: "Refresh Profiles"

    MDLabel:
        id: prof_status
        text: ""
        halign: "center"
        size_hint_y: None
        height: self.texture_size[1] + dp(16)

    ScrollView:
        MDList:
            id: prof_list


<FaceRegRow>:
    orientation: "vertical"
    size_hint_y: None
    height: dp(130)
    padding: "8dp"
    spacing: "4dp"

    MDBoxLayout:
        spacing: "8dp"

        Image:
            id: face_img
            size_hint_x: 0.3
            allow_stretch: True

        MDBoxLayout:
            orientation: "vertical"
            spacing: "4dp"
            size_hint_x: 0.7

            MDTextField:
                id: sid_input
                mode: "outlined"
                size_hint_y: None
                height: dp(48)
                MDTextFieldHintText:
                    text: "Student ID"

            MDTextField:
                id: name_input
                mode: "outlined"
                size_hint_y: None
                height: dp(48)
                MDTextFieldHintText:
                    text: "Name"
"""


# ---------------------------------------------------------------------------
# Helper: load base64 image into Kivy texture
# ---------------------------------------------------------------------------

def b64_to_texture(b64_string: str):
    """Convert a base64 JPEG string into a Kivy texture."""
    img_bytes = base64.b64decode(b64_string)
    buf = io.BytesIO(img_bytes)
    core_img = CoreImage(buf, ext="jpg")
    return core_img.texture


# ---------------------------------------------------------------------------
# Reusable face registration row widget
# ---------------------------------------------------------------------------

class FaceRegRow(MDBoxLayout):
    pass


# ---------------------------------------------------------------------------
# Screen: Attendance
# ---------------------------------------------------------------------------

class AttendanceScreen(MDBoxLayout):
    _file_manager = None
    _recognition_data = []

    def pick_image(self):
        if not self._file_manager:
            self._file_manager = MDFileManager(
                select_path=self._on_file_selected,
                exit_manager=self._close_manager,
                ext=[".jpg", ".jpeg", ".png"],
            )
        self._file_manager.show(os.path.expanduser("~"))

    def _close_manager(self, *args):
        if self._file_manager:
            self._file_manager.close()

    def _on_file_selected(self, path):
        self._close_manager()
        self.ids.att_status.text = "Processing..."
        threading.Thread(target=self._recognize, args=(path,), daemon=True).start()

    def _recognize(self, path):
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    api_url("/recognize"),
                    files={"image": ("photo.jpg", f, "image/jpeg")},
                    timeout=60,
                )
            data = resp.json()
            self._recognition_data = data.get("results", [])
            Clock.schedule_once(lambda dt: self._show_results(data))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.att_status, "text", f"Error: {e}")
            )

    def _show_results(self, data):
        counts = data.get("counts", {})
        total = data.get("total_faces", 0)
        self.ids.att_status.text = (
            f"{total} faces | "
            f"{counts.get('MATCH', 0)} recognized | "
            f"{counts.get('REVIEW', 0)} review | "
            f"{counts.get('UNKNOWN', 0)} unknown"
        )

        # Show annotated image
        b64 = data.get("annotated_b64", "")
        if b64:
            self.ids.att_image.texture = b64_to_texture(b64)

        # Show result list
        result_list = self.ids.att_results
        result_list.clear_widgets()
        for r in data.get("results", []):
            status_icons = {
                "MATCH": "check-circle",
                "REVIEW": "help-circle",
                "UNKNOWN": "close-circle",
            }
            item = MDListItem(
                MDListItemLeadingIcon(
                    icon=status_icons.get(r["status"], "help"),
                ),
                MDListItemHeadlineText(
                    text=f"Face {r['index']}: {r['name']}",
                ),
                MDListItemSupportingText(
                    text=f"ID: {r['student_id']}  |  {r['status']}",
                ),
                MDListItemTertiaryText(
                    text=f"Confidence: {r['score']}",
                ),
            )
            result_list.add_widget(item)


# ---------------------------------------------------------------------------
# Screen: Register
# ---------------------------------------------------------------------------

class RegisterScreen(MDBoxLayout):
    _file_manager = None
    _unregistered = []
    _face_rows = []

    def pick_image(self):
        if not self._file_manager:
            self._file_manager = MDFileManager(
                select_path=self._on_file_selected,
                exit_manager=self._close_manager,
                ext=[".jpg", ".jpeg", ".png"],
            )
        self._file_manager.show(os.path.expanduser("~"))

    def _close_manager(self, *args):
        if self._file_manager:
            self._file_manager.close()

    def _on_file_selected(self, path):
        self._close_manager()
        self.ids.reg_status.text = "Detecting faces..."
        threading.Thread(target=self._detect, args=(path,), daemon=True).start()

    def _detect(self, path):
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    api_url("/register"),
                    files={"image": ("photo.jpg", f, "image/jpeg")},
                    timeout=60,
                )
            data = resp.json()
            self._unregistered = data.get("unregistered", [])
            Clock.schedule_once(lambda dt: self._show_faces(data))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.reg_status, "text", f"Error: {e}")
            )

    def _show_faces(self, data):
        total = data.get("total_faces", 0)
        reg = data.get("registered_count", 0)
        unreg = len(self._unregistered)
        self.ids.reg_status.text = (
            f"{total} faces | {reg} registered | {unreg} new"
        )

        face_list = self.ids.reg_faces
        face_list.clear_widgets()
        self._face_rows = []

        for i, face_data in enumerate(self._unregistered):
            row = FaceRegRow()
            b64 = face_data.get("crop_b64", "")
            if b64:
                row.ids.face_img.texture = b64_to_texture(b64)
            self._face_rows.append(row)
            face_list.add_widget(row)

    def save_registrations(self):
        if not self._unregistered:
            self.ids.reg_status.text = "No faces to register."
            return

        assignments = []
        for face_data, row in zip(self._unregistered, self._face_rows):
            sid = row.ids.sid_input.text.strip()
            name = row.ids.name_input.text.strip()
            if sid and name:
                assignments.append({
                    "embedding": face_data["embedding"],
                    "student_id": sid,
                    "name": name,
                })

        if not assignments:
            self.ids.reg_status.text = "Enter ID and name for at least one face."
            return

        self.ids.reg_status.text = "Saving..."
        threading.Thread(
            target=self._save, args=(assignments,), daemon=True
        ).start()

    def _save(self, assignments):
        try:
            resp = requests.post(
                api_url("/register/save"),
                json={"assignments": assignments},
                timeout=30,
            )
            data = resp.json()
            if resp.status_code == 200:
                msg = f"Saved {data.get('saved', 0)} profile(s)."
            else:
                msg = f"Error: {data.get('detail', 'Unknown error')}"
            Clock.schedule_once(
                lambda dt: setattr(self.ids.reg_status, "text", msg)
            )
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.reg_status, "text", f"Error: {e}")
            )


# ---------------------------------------------------------------------------
# Screen: Add Samples
# ---------------------------------------------------------------------------

class SamplesScreen(MDBoxLayout):
    _file_manager = None
    _proposals = []
    _selected = set()

    def pick_images(self):
        if not self._file_manager:
            self._file_manager = MDFileManager(
                select_path=self._on_path_selected,
                exit_manager=self._close_manager,
                ext=[".jpg", ".jpeg", ".png"],
            )
        self._file_manager.show(os.path.expanduser("~"))

    def _close_manager(self, *args):
        if self._file_manager:
            self._file_manager.close()

    def _on_path_selected(self, path):
        self._close_manager()
        if os.path.isdir(path):
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        else:
            files = [path]

        if not files:
            self.ids.samp_status.text = "No images found."
            return

        self.ids.samp_status.text = f"Processing {len(files)} image(s)..."
        threading.Thread(
            target=self._upload, args=(files,), daemon=True
        ).start()

    def _upload(self, file_paths):
        try:
            file_handles = []
            files_param = []
            for fp in file_paths:
                fh = open(fp, "rb")
                file_handles.append(fh)
                files_param.append(
                    ("images", (os.path.basename(fp), fh, "image/jpeg"))
                )
            resp = requests.post(
                api_url("/add-samples"), files=files_param, timeout=120
            )
            for fh in file_handles:
                fh.close()
            data = resp.json()
            self._proposals = data.get("proposals", [])
            self._selected = {
                i for i, p in enumerate(self._proposals) if p.get("auto_add")
            }
            Clock.schedule_once(lambda dt: self._show_proposals())
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.samp_status, "text", f"Error: {e}")
            )

    def _show_proposals(self):
        auto = sum(1 for p in self._proposals if p.get("auto_add"))
        review = len(self._proposals) - auto
        self.ids.samp_status.text = (
            f"{len(self._proposals)} matches | {auto} confident | {review} review"
        )

        result_list = self.ids.samp_results
        result_list.clear_widgets()

        for i, p in enumerate(self._proposals):
            selected = i in self._selected
            icon_name = "checkbox-marked" if selected else "checkbox-blank-outline"
            tag = "[AUTO]" if p.get("auto_add") else "[REVIEW]"

            item = MDListItem(
                MDListItemLeadingIcon(icon=icon_name),
                MDListItemHeadlineText(
                    text=f"{tag} {p['name']} (ID: {p['student_id']})",
                ),
                MDListItemSupportingText(
                    text=f"Score: {p['score']}",
                ),
                on_release=lambda x, idx=i: self._toggle(idx),
            )
            result_list.add_widget(item)

    def _toggle(self, idx):
        if idx in self._selected:
            self._selected.discard(idx)
        else:
            self._selected.add(idx)
        self._show_proposals()

    def commit_samples(self):
        if not self._selected:
            self.ids.samp_status.text = "No samples selected."
            return

        samples = []
        for i in self._selected:
            p = self._proposals[i]
            samples.append({
                "student_id": p["student_id"],
                "embedding": p["embedding"],
            })

        self.ids.samp_status.text = "Adding samples..."
        threading.Thread(
            target=self._commit, args=(samples,), daemon=True
        ).start()

    def _commit(self, samples):
        try:
            resp = requests.post(
                api_url("/add-samples/commit"),
                json={"samples": samples},
                timeout=30,
            )
            data = resp.json()
            msg = f"Added {data.get('added', 0)} sample(s) to gallery."
            self._proposals = []
            self._selected = set()
            Clock.schedule_once(lambda dt: self._clear_and_msg(msg))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.samp_status, "text", f"Error: {e}")
            )

    def _clear_and_msg(self, msg):
        self.ids.samp_status.text = msg
        self.ids.samp_results.clear_widgets()


# ---------------------------------------------------------------------------
# Screen: Profiles
# ---------------------------------------------------------------------------

class ProfilesScreen(MDBoxLayout):
    _dialog = None

    def on_kv_post(self, *args):
        self.load_profiles()

    def load_profiles(self):
        self.ids.prof_status.text = "Loading..."
        threading.Thread(target=self._fetch, daemon=True).start()

    def _fetch(self):
        try:
            resp = requests.get(api_url("/profiles"), timeout=15)
            data = resp.json()
            Clock.schedule_once(lambda dt: self._show(data))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.prof_status, "text", f"Error: {e}")
            )

    def _show(self, data):
        profiles = data.get("profiles", [])
        self.ids.prof_status.text = f"{len(profiles)} registered student(s)"
        prof_list = self.ids.prof_list
        prof_list.clear_widgets()

        for p in profiles:
            item = MDListItem(
                MDListItemLeadingIcon(icon="account"),
                MDListItemHeadlineText(text=p["name"]),
                MDListItemSupportingText(
                    text=f"ID: {p['student_id']}",
                ),
                MDListItemTertiaryText(
                    text=f"Samples: {p['samples']}",
                ),
                on_release=lambda x, sid=p["student_id"], nm=p["name"]: self._show_actions(sid, nm),
            )
            prof_list.add_widget(item)

    def _show_actions(self, student_id, name):
        self._dialog = MDDialog(
            MDDialogHeadlineText(text=f"{name} ({student_id})"),
            MDDialogSupportingText(text="Choose an action for this profile."),
            MDDialogButtonContainer(
                MDButton(
                    MDButtonText(text="DELETE"),
                    style="text",
                    on_release=lambda x: self._delete(student_id),
                ),
                MDButton(
                    MDButtonText(text="CLOSE"),
                    style="text",
                    on_release=lambda x: self._dialog.dismiss(),
                ),
                spacing="8dp",
            ),
        )
        self._dialog.open()

    def _delete(self, student_id):
        if self._dialog:
            self._dialog.dismiss()
        threading.Thread(
            target=self._do_delete, args=(student_id,), daemon=True
        ).start()

    def _do_delete(self, student_id):
        try:
            resp = requests.delete(
                api_url(f"/profiles/{student_id}"), timeout=15
            )
            data = resp.json()
            if resp.status_code == 200:
                msg = f"Deleted {data.get('name', '')} ({student_id})"
            else:
                msg = f"Error: {data.get('detail', 'Unknown')}"
            Clock.schedule_once(lambda dt: self._refresh_msg(msg))
        except Exception as e:
            Clock.schedule_once(
                lambda dt: setattr(self.ids.prof_status, "text", f"Error: {e}")
            )

    def _refresh_msg(self, msg):
        self.ids.prof_status.text = msg
        self.load_profiles()


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class AttendanceApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Teal"
        return Builder.load_string(KV)

    def on_tab_switch(self, bar, item, item_icon, item_text):
        tab_map = {
            "Attendance": "attendance",
            "Register": "register",
            "Samples": "samples",
            "Profiles": "profiles",
        }
        screen_name = tab_map.get(item_text, "attendance")
        self.root.ids.screen_manager.current = screen_name


if __name__ == "__main__":
    AttendanceApp().run()
